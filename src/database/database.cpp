
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Database class (see the header for information about the class).
//
// =================================================================================================

#include <list>
#include <set>
#include <unistd.h>

#include "utilities/utilities.hpp"

#include "database/database.hpp"

#include "database/kernels/xaxpy/xaxpy.hpp"
#include "database/kernels/xdot/xdot.hpp"
#include "database/kernels/xgemv/xgemv.hpp"
#include "database/kernels/xgemv_fast/xgemv_fast.hpp"
#include "database/kernels/xgemv_fast_rot/xgemv_fast_rot.hpp"
#include "database/kernels/xger/xger.hpp"
#include "database/kernels/xgemm/xgemm.hpp"
#include "database/kernels/xgemm_direct/xgemm_direct.hpp"
#include "database/kernels/xconvgemm/xconvgemm.hpp"
#include "database/kernels/copy/copy.hpp"
#include "database/kernels/pad/pad.hpp"
#include "database/kernels/transpose/transpose.hpp"
#include "database/kernels/padtranspose/padtranspose.hpp"
#include "database/kernels/invert/invert.hpp"

#include "database/kernels/gemm_routine/gemm_routine.hpp"
#include "database/kernels/trsv_routine/trsv_routine.hpp"

#include "database/apple_cpu_fallback.hpp"

namespace clblast {
// =================================================================================================

std::vector<database::DatabaseEntry> Database::database = std::vector<database::DatabaseEntry>{};
const std::vector<database::DatabaseEntry> Database::apple_cpu_fallback = std::vector<database::DatabaseEntry>{
  database::XaxpyApple, database::XdotApple,
  database::XgemvApple, database::XgemvFastApple, database::XgemvFastRotApple, database::XgerApple, database::XtrsvApple,
  database::XgemmApple, database::XgemmDirectApple, database::XconvgemmApple,
  database::CopyApple, database::PadApple, database::TransposeApple, database::PadtransposeApple,
  database::InvertApple,
  database::TrsvRoutineApple
};

// The default values
const std::string Database::kDeviceVendorAll = "default";

// =================================================================================================

// Constructor, computing device properties and populating the parameter-vector from the database.
// This takes an optional overlay database in case of custom tuning or custom kernels.
Database::Database(const Device &device, const std::string &kernel_name,
                   const Precision precision, const std::vector<database::DatabaseEntry> &overlay) {
  entries_ = std::shared_ptr<std::vector<database::Entry>>(new std::vector<database::Entry>());
  // Initializes the static variable on first use. At this point we are sure all global variables are initialized
  if (database.size() == 0) {
    database = std::vector<database::DatabaseEntry>{
        database::XaxpyHalf, database::XaxpySingle, database::XaxpyDouble, database::XaxpyComplexSingle, database::XaxpyComplexDouble,
        database::XdotHalf, database::XdotSingle, database::XdotDouble, database::XdotComplexSingle, database::XdotComplexDouble,
        database::XgemvHalf, database::XgemvSingle, database::XgemvDouble, database::XgemvComplexSingle, database::XgemvComplexDouble,
        database::XgemvFastHalf, database::XgemvFastSingle, database::XgemvFastDouble, database::XgemvFastComplexSingle, database::XgemvFastComplexDouble,
        database::XgemvFastRotHalf, database::XgemvFastRotSingle, database::XgemvFastRotDouble, database::XgemvFastRotComplexSingle, database::XgemvFastRotComplexDouble,
        database::XgerHalf, database::XgerSingle, database::XgerDouble, database::XgerComplexSingle, database::XgerComplexDouble,
        database::XgemmHalf, database::XgemmSingle, database::XgemmDouble, database::XgemmComplexSingle, database::XgemmComplexDouble,
        database::XgemmDirectHalf, database::XgemmDirectSingle, database::XgemmDirectDouble, database::XgemmDirectComplexSingle, database::XgemmDirectComplexDouble,
        database::XconvgemmHalf, database::XconvgemmSingle, database::XconvgemmDouble, database::XconvgemmComplexSingle, database::XconvgemmComplexDouble,
        database::CopyHalf, database::CopySingle, database::CopyDouble, database::CopyComplexSingle, database::CopyComplexDouble,
        database::PadHalf, database::PadSingle, database::PadDouble, database::PadComplexSingle, database::PadComplexDouble,
        database::TransposeHalf, database::TransposeSingle, database::TransposeDouble, database::TransposeComplexSingle, database::TransposeComplexDouble,
        database::PadtransposeHalf, database::PadtransposeSingle, database::PadtransposeDouble, database::PadtransposeComplexSingle, database::PadtransposeComplexDouble,
        database::InvertHalf, database::InvertSingle, database::InvertDouble, database::InvertComplexSingle, database::InvertComplexDouble,
        database::GemmRoutineHalf, database::GemmRoutineSingle, database::GemmRoutineDouble, database::GemmRoutineComplexSingle, database::GemmRoutineComplexDouble,
        database::TrsvRoutineHalf, database::TrsvRoutineSingle, database::TrsvRoutineDouble, database::TrsvRoutineComplexSingle, database::TrsvRoutineComplexDouble
    };
  }

  // Finds device information
  const auto device_type = GetDeviceType(device);
  const auto device_vendor = GetDeviceVendor(device);
  const auto device_architecture = GetDeviceArchitecture(device);
  const auto device_name = GetDeviceName(device);

  // Prints the obtained information in verbose mode
  log_debug("Device type '" + device_type + "'; vendor '" + device_vendor + "'");
  log_debug("Device name '" + device_name + "'; architecture '" + device_architecture + "'");

  // Sets the databases to search through
  auto databases = std::list<std::vector<database::DatabaseEntry>>{overlay, database};

  // Special case: modifies the database if the device is a CPU with Apple OpenCL
  #if defined(__APPLE__) || defined(__MACOSX)
    if (device.Type() == "CPU") {
      const auto extensions = device.Capabilities();
      const auto is_apple = (extensions.find("cl_APPLE_SetMemObjectDestructor") == std::string::npos) ? false : true;
      const auto is_likely_apple = device.MaxWorkGroupSize() <= 32;
      if (is_apple || is_likely_apple) {
        databases.push_front(apple_cpu_fallback);
      }
    }
  #endif

  // Searches potentially multiple databases
  auto search_result = std::vector<database::Entry>();
  for (auto &db: databases) {
    search_result = Search(kernel_name, device_vendor, device_type,
                           device_name, device_architecture, precision, db);
    if (search_result.size() != 0) {
      entries_->insert(std::end(*entries_), std::begin(search_result), std::end(search_result));
      break;
    }
  }

  if (search_result.size() == 0) { throw RuntimeErrorCode(StatusCode::kDatabaseError); }
}

// =================================================================================================

// Returns a vector of OpenCL pre-processor defines in string form
std::vector<std::string> Database::GetDefines() const {
  // for each entry in the database, get the defines from the parameters in the entry
  auto defines = std::vector<std::string>();
  for (auto &entry: *entries_) {
    defines.push_back(GetDefines(entry.second));
  }
  return defines;
}

// Returns OpenCL pre-processor defines in string form
std::string Database::GetDefines(const database::Parameters &parameters) const {
  std::string defines{};
  for (auto &parameter: parameters) {
    defines += "#ifdef " + parameter.first + "\n\t#undef " + parameter.first + " \n\t#define " + 
    parameter.first + " "+ToString(parameter.second) + "\n#endif\n" + "#ifndef " + parameter.first +
    "\n\t#define " + parameter.first + " "+ToString(parameter.second) + "\n#endif\n";
  }
  return defines;
}

// ... or just the values as string
std::vector<std::string> Database::GetValuesString() const {
  auto defines = std::vector<std::string>();
  for (auto &entry: *entries_) {
    defines.push_back(GetValuesString(entry.second));
  }
  return defines;
}

// ... or just the values as string
std::string Database::GetValuesString(const database::Parameters &parameters) const {
  std::string defines{};
  for (auto &parameter: parameters) {
    defines += "_"+ToString(parameter.second);
  }
  return defines;
}

// Retrieves the names of all the parameters
std::vector<std::string> Database::GetParameterNames() const {
  auto parameter_names = std::set<std::string>();
  for (auto &entry: *entries_) {
    for (auto &parameter: entry.second) {
      parameter_names.insert(parameter.first);
    }
  }
  return std::vector<std::string>(parameter_names.begin(), parameter_names.end());
}

// =================================================================================================

// Searches a particular database for the right kernel and precision
std::vector<database::Entry> Database::Search(const std::string &this_kernel,
                                      const std::string &this_vendor, const std::string &this_type,
                                      const std::string &this_device, const std::string &this_architecture,
                                      const Precision this_precision,
                                      const std::vector<database::DatabaseEntry> &this_database) const {

  // Selects the right kernel
  for (auto &db: this_database) {
    if ((db.kernel == this_kernel) &&
        (db.precision == this_precision || db.precision == Precision::kAny)) {

      // Searches for the right vendor and device type, or selects the default if unavailable
      const auto parameters = SearchVendorAndType(this_vendor, this_type, this_device, this_architecture,
                                                  db.vendors, db.parameter_names);
      if (parameters.size() != 0) { return parameters; }
      return SearchVendorAndType(kDeviceVendorAll, database::kDeviceTypeAll, this_device, this_architecture,
                                 db.vendors, db.parameter_names);
    }
  }

  // If we reached this point, the entry was not found in this database
  return std::vector<database::Entry>();
}

std::vector<database::Entry> Database::SearchVendorAndType(const std::string &target_vendor, const std::string &target_type,
                                                   const std::string &this_device, const std::string &this_architecture,
                                                   const std::vector<database::DatabaseVendor> &vendors,
                                                   const std::vector<std::string> &parameter_names) const {
  for (auto &vendor: vendors) {
    if ((vendor.name == target_vendor) && (vendor.type == target_type)) {
      log_debug("Found architectures of vendor '" + target_vendor + "' and type '" + target_type + "'");

      // Searches the architecture; if unavailable returns the vendor's default parameters
      auto parameters = SearchArchitecture(this_architecture, this_device, vendor.architectures, parameter_names);
      if (parameters.size() != 0) { return parameters; }
      return SearchArchitecture("default", this_device, vendor.architectures, parameter_names);
    }
  }
  return std::vector<database::Entry>();
}

std::vector<database::Entry> Database::SearchArchitecture(const std::string &target_architecture,
                                                  const std::string &this_device,
                                                  const std::vector<database::DatabaseArchitecture> &architectures,
                                                  const std::vector<std::string> &parameter_names) const {
  for (auto &architecture: architectures) {
    if (architecture.name == target_architecture) {
      log_debug("Found devices of architecture type '" + target_architecture + "'");

      // Searches the device; if unavailable returns the architecture's default parameters
      auto parameters = SearchDevice(this_device, architecture.devices, parameter_names);
      if (parameters.size() != 0) { return parameters; }
      return SearchDevice("default", architecture.devices, parameter_names);
    }
  }
  return std::vector<database::Entry>();
}

std::vector<database::Entry> Database::SearchDevice(const std::string &target_device,
                                            const std::vector<database::DatabaseDevice> &devices,
                                            const std::vector<std::string> &parameter_names) const {
  for (auto &device: devices) {
    const auto device_name = CharArrayToString(device.name);
    // Cuts off 'target_device' string at 50 since the database cuts off as well
    const auto target_device_cut_off = (target_device.length() > 50) ? target_device.substr(0, 50) : target_device;
    if (device_name == target_device_cut_off) {
      log_debug("Found parameters for device type '" + target_device_cut_off + "'");
      auto entries = std::vector<database::Entry>();
      for (auto &db_input: device.inputs) {
          // Sets the input sizes
          auto input = database::Input{};
          for (size_t i = 0; i < db_input.value.size(); ++i) {
            input[i] = static_cast<int>(db_input.value[i]);
          }

          // Sets the parameters accordingly
          auto parameters = database::Parameters();
          if (parameter_names.size() > db_input.parameters.size()) { return std::vector<database::Entry>(); } // ERROR
          for (size_t i = 0; i < parameter_names.size(); ++i) {
            parameters[parameter_names[i]] = static_cast<size_t>(db_input.parameters[i]);
          }

          // Adds the entry to the list
          entries.push_back(std::make_pair(input, parameters));
      }
      return entries;
    }
  }
  return std::vector<database::Entry>();
}

// Helper to convert from database format to proper types
std::string Database::CharArrayToString(const database::Name char_array) const {
  auto result = std::string(char_array.data());
  result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
  return result;
}

size_t Database::GetParameterValue(const database::Input& input, const std::string& parameter_name) const {
  // get the closest entry
  database::Entry& entry = GetEntryForInput(input);
  // then, given this entry
  // get the parameter value
  auto parameter_value = entry.second[parameter_name];
  return parameter_value;
}

std::string Database::GetValuesStringForInput(const database::Input& input) const {
  return GetValuesString(GetParameters(input));
}

// Select the entry for a given input, return parameters.
database::Parameters& Database::GetParameters(const database::Input& input) const {
  return GetEntryForInput(input).second;
}

// Get the entry for a given input
database::Entry& Database::GetEntryForInput(const database::Input& input) const {
  auto& entry = GetClosestEntry(input); // get the closest entry by euclidean distance
  return entry;
}

// Get the closest entry, by euclidean distance
database::Entry& Database::GetClosestEntry(const database::Input& target_input) const {

  // ensure database has at least one element
  auto v = *entries_;
  if (v.size() == 0) {
    throw RuntimeErrorCode(StatusCode::kDatabaseError);
  }

  database::Entry* closest_entry = &((*entries_)[0]); // the closest entry by euclidean distance
  int closest_distance = std::numeric_limits<int>::max(); // the closest distance found

  for (auto &pair: *entries_) {
    log_debug("pair before: " + GetValuesString(pair.second));
  }
  for (auto &pair: *entries_) {
    const auto input = pair.first;
    if(input.size() > 0 && target_input.size() > 0) {
      // TODO: undo
      double distance = EuclideanDistance(input, target_input);
      // double distance = 0;

      // if distance is smaller than the current closest distance, update closest_input
      if (distance < closest_distance) {
        closest_distance = distance;
        closest_entry = &pair;
      }
      if (input == target_input) {
        break;
      }
    }
    for (auto &pair: *entries_) {
      log_debug("pair during: " + GetValuesString(pair.second));
    }
  }
  for (auto &pair: *entries_) {
    log_debug("pair after: " + GetValuesString(pair.second));
  }
  return *closest_entry;
}

// Helper to calculate the Euclidean distance between two vectors
double Database::EuclideanDistance(const database::Input & v1, const database::Input & v2) const {
    if (v1.size() != v2.size()) {
        log_debug("vector 1 is of size " + ToString(v1.size()) + " and vector 2 is of size" + ToString(v2.size()));
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    double distance = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double v1_i = v1[i];
        double v2_i = v2[i];
        distance += std::pow(v1_i - v2_i, 2);
    }

    return std::sqrt(distance);
}

// =================================================================================================
} // namespace clblast
