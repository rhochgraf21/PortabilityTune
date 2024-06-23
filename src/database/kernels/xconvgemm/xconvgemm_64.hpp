
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xconvgemm64' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry XconvgemmDouble = {
  "Xconvgemm", Precision::kDouble, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "default", {
          { Name{"AMD Radeon Pro 450 Compute Engine                 "} , {
 { Input{0, 0, 0}, Params{1, 8, 32, 8, 8, 0, 0, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 16, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700 XT                             "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 8, 0, 0, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 8, 0, 0, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "} , {
 { Input{0, 0, 0}, Params{1, 16, 8, 8, 16, 0, 0, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 16, 8, 8, 16, 0, 0, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1031", {
          { Name{"AMD Radeon RX 6700 XT                             "} , {
 { Input{0, 0, 0}, Params{1, 8, 8, 16, 16, 0, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 8, 16, 16, 0, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1032", {
          { Name{"AMD Radeon RX 6600 XT                             "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1035", {
          { Name{"AMD Radeon Graphics                               "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 8, 0, 0, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 8, 0, 0, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1100", {
          { Name{"Radeon RX 7900 XTX                                "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 16, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 8, 16, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx906:sramecc+:xnack-", {
          { Name{"AMD Radeon VII                                    "} , {
 { Input{0, 0, 0}, Params{1, 16, 32, 8, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 16, 32, 8, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz         "} , {
 { Input{0, 0, 0}, Params{1, 16, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 16, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Gen9 HD Graphics NEO                     "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 16, 8, 0, 0, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) UHD Graphics 620                         "} , {
 { Input{0, 0, 0}, Params{1, 8, 8, 32, 8, 0, 0, 4, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 16, 8, 0, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "SM7.0", {
          { Name{"Quadro GV100                                      "} , {
 { Input{0, 0, 0}, Params{1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla V100-PCIE-16GB                              "} , {
 { Input{0, 0, 0}, Params{1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM7.5", {
          { Name{"NVIDIA GeForce RTX 2080 Ti                        "} , {
 { Input{0, 0, 0}, Params{1, 16, 8, 16, 8, 0, 0, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Quadro T2000                                      "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla T4                                          "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.0", {
          { Name{"A100-PCIE-40GB                                    "} , {
 { Input{0, 0, 0}, Params{1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.6", {
          { Name{"NVIDIA GeForce RTX 3090                           "} , {
 { Input{0, 0, 0}, Params{1, 8, 16, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 8, 16, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{0, 0, 0}, Params{1, 16, 16, 32, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
