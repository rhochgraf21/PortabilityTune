
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Padtranspose64' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry PadtransposeDouble = {
  "Padtranspose", Precision::kDouble, {"PADTRA_PAD", "PADTRA_TILE", "PADTRA_WPT"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere", {
          { Name{"AMD Radeon RX 480                                 "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 580 2048SP                          "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX590 GME                              "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Fiji", {
          { Name{"AMD Radeon R9 Fury X                              "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon R9 M370X Compute Engine                "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Hawaii", {
          { Name{"AMD FirePro W8100                                 "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon R9 290X                                "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Oland", {
          { Name{"Oland                                             "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Pitcairn", {
          { Name{"AMD Radeon R9 270X                                "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Tahiti", {
          { Name{"AMD Radeon HD 7970                                "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Tonga", {
          { Name{"AMD Radeon R9 380                                 "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Vega", {
          { Name{"Radeon RX Vega                                    "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { Name{"AMD Radeon Pro 450 Compute Engine                 "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon Pro 580 Compute Engine                 "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700                                "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 5700 XT                             "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 6900 XT                             "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1031", {
          { Name{"AMD Radeon RX 6700 XT                             "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1032", {
          { Name{"AMD Radeon RX 6600 XT                             "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1035", {
          { Name{"AMD Radeon Graphics                               "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1100", {
          { Name{"Radeon RX 7900 XTX                                "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1102", {
          { Name{"AMD Radeon RX 7600                                "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx902", {
          { Name{"AMD Radeon(TM) Graphics                           "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon(TM) RX Vega 10 Graphics                "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx906:sramecc+:xnack-", {
          { Name{"AMD Radeon VII                                    "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx90c", {
          { Name{"AMD Radeon(TM) Graphics                           "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "default", {
          { Name{"Mali-T760                                         "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz         "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz           "} , {
 { Input{1024, 1024, 0}, Params{0, 32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-4590S CPU @ 3.00GHz          "} , {
 { Input{1024, 1024, 0}, Params{1, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz          "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz   "} , {
 { Input{1024, 1024, 0}, Params{0, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz           "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz          "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz          "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz         "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz         "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz         "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz         "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) HD Graphics 620                          "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) UHD Graphics 620                         "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "default", {
          { Name{"Intel(R) Many Integrated Core Acceleration Card   "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "SM2.0", {
          { Name{"GeForce GTX 480                                   "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 580                                   "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM3.0", {
          { Name{"GRID K520                                         "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 670                                   "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 680                                   "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 760 Ti OEM                            "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM3.5", {
          { Name{"GeForce 920A                                      "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX TITAN                                 "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX TITAN Black                           "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla K20m                                        "} , {
 { Input{1024, 1024, 0}, Params{0, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla K40m                                        "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM5.0", {
          { Name{"GeForce 920MX                                     "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 750                                   "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 750 Ti                                "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 32, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM5.2", {
          { Name{"GeForce GTX 970                                   "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 980                                   "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX TITAN X                               "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM6.0", {
          { Name{"Tesla P100-PCIE-16GB                              "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM6.1", {
          { Name{"GeForce GTX 1070                                  "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 1070 Ti                               "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 1080                                  "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 1080 Ti                               "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce MX150                              "} , {
 { Input{1024, 1024, 0}, Params{1, 32, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"TITAN X (Pascal)                                  "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM7.0", {
          { Name{"Quadro GV100                                      "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla V100-PCIE-16GB                              "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM7.5", {
          { Name{"GeForce GTX 1650                                  "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce GTX 1650 SUPER                     "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce GTX 1650 Ti                        "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2060                           "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 SUPER                     "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 Super                     "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 with Max-Q Design         "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2080 Ti                        "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2080 with Max-Q Design         "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Quadro T2000                                      "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"TITAN RTX                                         "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla T4                                          "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.0", {
          { Name{"A100-PCIE-40GB                                    "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.6", {
          { Name{"NVIDIA GeForce RTX 3050 Ti Laptop GPU             "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3060 Laptop GPU                "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3070                           "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3070 Ti Laptop GPU             "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080                           "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080 Laptop GPU                "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080 Ti                        "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3090                           "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.9", {
          { Name{"NVIDIA GeForce RTX 4070 Laptop GPU                "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4070 Ti                        "} , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4080                           "} , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4090                           "} , {
 { Input{1024, 1024, 0}, Params{0, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{1024, 1024, 0}, Params{1, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
