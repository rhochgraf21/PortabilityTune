
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Routine base class (see the header for information about the class).
//
// =================================================================================================

#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// For each kernel this map contains a list of routines it is used in
const std::vector<std::string> Routine::routines_axpy = {"AXPY", "COPY", "SCAL", "SWAP"};
const std::vector<std::string> Routine::routines_dot = {"AMAX", "ASUM", "DOT", "DOTC", "DOTU", "MAX", "MIN", "NRM2", "SUM"};
const std::vector<std::string> Routine::routines_ger = {"GER", "GERC", "GERU", "HER", "HER2", "HPR", "HPR2", "SPR", "SPR2", "SYR", "SYR2"};
const std::vector<std::string> Routine::routines_gemv = {"GBMV", "GEMV", "HBMV", "HEMV", "HPMV", "SBMV", "SPMV", "SYMV", "TMBV", "TPMV", "TRMV", "TRSV"};
const std::vector<std::string> Routine::routines_gemm = {"GEMM", "HEMM", "SYMM", "TRMM"};
const std::vector<std::string> Routine::routines_gemm_syrk = {"GEMM", "HEMM", "HER2K", "HERK", "SYMM", "SYR2K", "SYRK", "TRMM", "TRSM"};
const std::vector<std::string> Routine::routines_trsm = {"TRSM"};
const std::unordered_map<std::string, const std::vector<std::string>> Routine::routines_by_kernel = {
  {"Xaxpy", routines_axpy},
  {"Xdot", routines_dot},
  {"Xgemv", routines_gemv},
  {"XgemvFast", routines_gemv},
  {"XgemvFastRot", routines_gemv},
  {"Xtrsv", routines_gemv},
  {"Xger", routines_ger},
  {"Copy", routines_gemm_syrk},
  {"Pad", routines_gemm_syrk},
  {"Transpose", routines_gemm_syrk},
  {"Padtranspose", routines_gemm_syrk},
  {"Xgemm", routines_gemm_syrk},
  {"XgemmDirect", routines_gemm},
  {"GemmRoutine", routines_gemm},
  {"Invert", routines_trsm},
};
// =================================================================================================

// The constructor does all heavy work, errors are returned as exceptions
Routine::Routine(Queue &queue, EventPointer event, const std::string &name,
                 const std::vector<std::string> &routines, const Precision precision,
                 const std::vector<database::DatabaseEntry> &userDatabase,
                 std::unordered_map<std::string, std::vector<const char *>> source):
    precision_(precision),
    routine_name_(name),
    kernel_names_(routines),
    queue_(queue),
    event_(event),
    context_(queue_.GetContext()),
    device_(queue_.GetDevice()),
    db_(routines) {
  
  log_debug("Initializing dbs");
  InitDatabase(device_, routines, precision, userDatabase, db_);
  log_debug("Initializing program");
  InitProgram(source);
}

// Temporary overloaded constructor for testing
// The constructor does all heavy work, errors are returned as exceptions
Routine::Routine(Queue &queue, EventPointer event, const std::string &name,
                 const std::vector<std::string> &routines, 
                 const Precision precision,
                 const std::vector<database::DatabaseEntry> &userDatabase,
                 std::initializer_list<const char *> source):
    precision_(precision),
    routine_name_(name),
    kernel_names_(routines),
    queue_(queue),
    event_(event),
    context_(queue_.GetContext()),
    device_(queue_.GetDevice()),
    db_(routines) {

  // log_debug("initializing routine " + name);
  InitDatabase(device_, routines, precision, userDatabase, db_);
  // log_debug("done");
  // InitProgram(source);
}

void Routine::InitProgram(std::unordered_map<std::string, std::vector<const char *>> source) {
  // printf("INIT PROGRAM");
  // Determines the identifier for this particular routine call
  auto routine_info = routine_name_;
  for (const auto &kernel_name : kernel_names_) {
    for(const auto &defines_string: db_(kernel_name).GetValuesString()) {
      // log_debug("db has values string: " + defines_string + " for kernel " + kernel_name);
      routine_info += "_" + kernel_name + defines_string;
    }
  }
  // log_debug(routine_info);

  // Queries the cache to see whether or not the program (context-specific) is already there
  bool has_program;
  program_ = ProgramCache::Instance().Get(ProgramKeyRef{ context_(), device_(), precision_, routine_info },
                                          &has_program);
  log_debug("has_program: " + ToString(static_cast<int>(has_program)+0) + " for program " + routine_info);
  if (has_program) { return; }

  // Sets the build options from an environmental variable (if set)
  auto options = std::vector<std::string>();
  const auto environment_variable = std::getenv("CLBLAST_BUILD_OPTIONS");
  if (environment_variable != nullptr) {
    options.push_back(std::string(environment_variable));
  }

  // Queries the cache to see whether or not the binary (device-specific) is already there. If it
  // is, a program is created and stored in the cache
  const auto device_name = GetDeviceName(device_);
  const auto platform_id = device_.PlatformID();
  bool has_binary;
  auto binary = BinaryCache::Instance().Get(BinaryKeyRef{platform_id,  precision_, routine_info, device_name },
                                            &has_binary);
  if (has_binary) {
    program_ = std::make_shared<Program>(device_, context_, binary);
    SetOpenCLKernelStandard(device_, options);
    program_->Build(device_, options);
    ProgramCache::Instance().Store(ProgramKey{ context_(), device_(), precision_, routine_info },
                                    std::shared_ptr<Program>{program_});
    return;
  }

  // Otherwise, the kernel will be compiled and program will be built. Both the binary and the
  // program will be added to the cache.

  // Inspects whether or not FP64 is supported in case of double precision
  if ((precision_ == Precision::kDouble && !PrecisionSupported<double>(device_)) ||
      (precision_ == Precision::kComplexDouble && !PrecisionSupported<double2>(device_))) {
    throw RuntimeErrorCode(StatusCode::kNoDoublePrecision);
  }

  // As above, but for FP16 (half precision)
  if (precision_ == Precision::kHalf && !PrecisionSupported<half>(device_)) {
    throw RuntimeErrorCode(StatusCode::kNoHalfPrecision);
  }

  // Collect the default defines for this routine, place them in the source string
  auto source_string = std::string{""};

  source_string += "\n #define CONCATENATE_DETAIL(x, y) x##y \n #define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y) \n";

  auto source_string_default_kernel = std::string{""};
  for (const char *s: source["default"]) {
      source_string_default_kernel += s;
  }
  source_string += source_string_default_kernel;

  // Collect the defines for each kernel parameters.
  for (auto &kernel_name: kernel_names_) {

    // Get all of the parameters for this kernel
    auto parameter_sets_v = db_(kernel_name).GetValuesString();
    auto defines_v = db_(kernel_name).GetDefines();

    // get the source kernels
    std::vector<const char *> kernel_source = source[kernel_name];

    // combine source kernels into single source kernel
    auto source_string_kernel = std::string{""};
    for (const char *s: kernel_source) {
      source_string_kernel += s;
    }

    /*
      for every set of parameters, duplicate the source code for the kernel
      add the defines for that set of parameters,
      and add macro to rename cl functions with parameters and avoid collisions.
    */
    
    for(size_t i = 0; i < parameter_sets_v.size(); i++) {

      // get the defines for that kernel
      source_string += defines_v[i];

      // add the parameters string
      // printf("defining parameters: %s, kernel_name: %s \n", parameter_sets_v[i].c_str(), kernel_name.c_str());
      source_string += "#define PARAMS " + parameter_sets_v[i] + "\n";

      // duplicate the source code for the kernel
      source_string += source_string_kernel;

      // undefine parameters string
      source_string += "\n#ifdef PARAMS\n#undef PARAMS\n#endif\n";
    }
  }

  // printf("%s", source_string.c_str());

  // Completes the source and compiles the kernel
  program_ = CompileFromSource(source_string, precision_, routine_name_,
                               device_, context_, options, 0);


  // printf("compilation successful");


  // Store the compiled binary and program in the cache
  BinaryCache::Instance().Store(BinaryKey{platform_id, precision_, routine_info, device_name},
                                program_->GetIR());

  // printf("store in program cache with routine info: %s", routine_info.c_str());

  ProgramCache::Instance().Store(ProgramKey{context_(), device_(), precision_, routine_info},
                                 std::shared_ptr<Program>{program_});
}

// =================================================================================================
} // namespace clblast
