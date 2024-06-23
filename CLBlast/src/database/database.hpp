
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Database class, providing a static variable holding the actual database
// information. The class also provides utility functions to search the database and to access a
// found entry by parameter-key. The database itself is filled in the corresponding source-file and
// partially also by the database/xxxxx.h files, in which kernel-specific parameters are found.
//
// =================================================================================================

#ifndef CLBLAST_DATABASE_H_
#define CLBLAST_DATABASE_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "utilities/utilities.hpp"
#include "database/database_structure.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Database {
 public:

  // The OpenCL device vendors
  static const std::string kDeviceVendorAll;

  // The database consists of separate database entries, stored together in a vector
  static std::vector<database::DatabaseEntry> database;

  // Database for a special case: Apple CPUs support limited number of threads
  static const std::vector<database::DatabaseEntry> apple_cpu_fallback;

  Database() = default;

  // The constructor with a user-provided database overlay (potentially an empty vector)
  explicit Database(const Device &device, const std::string &kernel_name,
                    const Precision precision, const std::vector<database::DatabaseEntry> &overlay);

  // Obtain a list of OpenCL pre-processor defines based on the parameters
  std::vector<std::string> GetDefines() const;
  std::string GetDefines(const database::Parameters &parameters) const;

  // Retrieves the values or names of all the parameters
  std::vector<std::string> GetValuesString() const;
  std::string GetValuesString(const database::Parameters &parameters) const;

  std::vector<std::string> GetParameterNames() const;
  std::vector<database::Entry>& GetEntries() const { return *entries_; }

  size_t GetParameterValue(const database::Input& input, const std::string& parameter_name) const;

  // Select the entry for a given input, returning string of parameters.
  std::string GetValuesStringForInput(const database::Input& input) const;

  // Select the entry for a given input, return parameters.
  database::Parameters& GetParameters(const database::Input& input) const;

 private:
  // Search method functions, returning a set of parameters (possibly empty)
  std::vector<database::Entry> Search(const std::string &this_kernel,
                              const std::string &this_vendor, const std::string &this_type,
                              const std::string &this_device, const std::string &this_architecture,
                              const Precision this_precision,
                              const std::vector<database::DatabaseEntry> &db) const;
  std::vector<database::Entry> SearchDevice(const std::string &target_device,
                        const std::vector<database::DatabaseDevice> &devices,
                        const std::vector<std::string> &parameter_names) const;
  std::vector<database::Entry> SearchArchitecture(const std::string &target_architecture,
                                          const std::string &this_device,
                                          const std::vector<database::DatabaseArchitecture> &architectures,
                                          const std::vector<std::string> &parameter_names) const;
  std::vector<database::Entry> SearchVendorAndType(const std::string &target_vendor,
                                           const std::string &target_type,
                                           const std::string &this_device, const std::string &this_architecture,
                                           const std::vector<database::DatabaseVendor> &vendors,
                                           const std::vector<std::string> &parameter_names) const;

  // Helper to convert from database format to proper types
  std::string CharArrayToString(const database::Name char_array) const;

  // Found parameters suitable for this device/kernel
  std::shared_ptr<std::vector<database::Entry>> entries_;

  // Get the entry for a given input
  database::Entry& GetEntryForInput(const database::Input& input) const;

  // Get the closest entry to a given input, by euclidean distance
  database::Entry& GetClosestEntry(const database::Input& target_input) const;

  // Helper to find the euclidean distance between two vectors
  double EuclideanDistance(const database::Input & v1, const database::Input & v2) const;

};

// =================================================================================================

// Multiple databases together in a map
class Databases {
 public:

  explicit Databases(const std::vector<std::string> &kernel_names): kernel_names_(kernel_names) { }

  // Database accessor
  const Database& operator()(const std::string &kernel_name) const { return databases_.at(kernel_name);  }
  Database& operator()(const std::string &kernel_name) { return databases_[kernel_name]; }

  // Parameter value accessor
  size_t GetParameterValue(const database::Input& input, const std::string &parameter_name, const std::string &kernel_name) {
    return databases_[kernel_name].GetParameterValue(input, parameter_name);
  }

  // fallback: Retrieves a parameter from the database (for not-yet-implemented feature testing)
  size_t operator[](const std::string &key) const {
    log_debug("searching database for parameter " + key);
    auto input = database::Input{0, 0, 0};
    for (const auto &kernel_name : kernel_names_) {
        log_debug("searching " + kernel_name + " database for parameter " + key);
        const auto itr = databases_.find(key);
        log_debug("result calculated for  " + kernel_name + " database for parameter " + key);
        if (itr == databases_.end()) {
            log_debug("no value in" + kernel_name + " database for parameter " + key);
            continue;
        }
        log_debug("found " + kernel_name + " database for parameter " + key);
        const auto& kernel_input_db = itr->second;
        log_debug(kernel_name);
        log_debug(key);
        return kernel_input_db.GetParameterValue(input, key);
    }
    log_debug("throwing db error");
    printf("invalid db access for %s", key.c_str());
    throw RuntimeErrorCode(StatusCode::kDatabaseError);
  }

 private:
  const std::vector<std::string> kernel_names_;
  std::unordered_map<std::string, Database> databases_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_DATABASE_H_
#endif
