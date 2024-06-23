
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xgemv_Fast_Rot3232' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry XgemvFastRotComplexSingle = {
  "XgemvFastRot", Precision::kComplexSingle, {"VW3", "WGS3", "WPT3"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere", {
          { Name{"AMD Radeon RX 480                                 "} , {
 { Input{2048, 2048, 0}, Params{2, 32, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 580 2048SP                          "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX590 GME                              "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Fiji", {
          { Name{"AMD Radeon R9 Fury X                              "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon R9 M370X Compute Engine                "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Hawaii", {
          { Name{"AMD FirePro W8100                                 "} , {
 { Input{2048, 2048, 0}, Params{2, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Tonga", {
          { Name{"AMD Radeon R9 380                                 "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Turks", {
          { Name{"AMD Radeon HD 6770M                               "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{4, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Vancouver", {
          { Name{"ATI Radeon HD 6750M                               "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { Name{"AMD Radeon Pro 450 Compute Engine                 "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon Pro 580 Compute Engine                 "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700                                "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 5700 XT                             "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 6900 XT                             "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1031", {
          { Name{"AMD Radeon RX 6700 XT                             "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1032", {
          { Name{"AMD Radeon RX 6600 XT                             "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "} , {
 { Input{2048, 2048, 0}, Params{4, 64, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{4, 64, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1035", {
          { Name{"AMD Radeon Graphics                               "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1100", {
          { Name{"Radeon RX 7900 XTX                                "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1102", {
          { Name{"AMD Radeon RX 7600                                "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx902", {
          { Name{"AMD Radeon(TM) Graphics                           "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx906:sramecc+:xnack-", {
          { Name{"AMD Radeon VII                                    "} , {
 { Input{2048, 2048, 0}, Params{2, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx90c", {
          { Name{"AMD Radeon(TM) Graphics                           "} , {
 { Input{2048, 2048, 0}, Params{1, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{1, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "default", {
          { Name{"Mali-T760                                         "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{4, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Apple GPUs
      kDeviceTypeGPU, "Apple", {
        { "default", {
          { Name{"Apple M1                                          "} , {
 { Input{2048, 2048, 0}, Params{4, 64, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Apple M2 Max                                      "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 64, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Imagination Technologies GPUs
      kDeviceTypeGPU, "Imagination Technologies", {
        { "default", {
          { Name{"PowerVR B-Series BXE-4-32                         "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz         "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz           "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-4590S CPU @ 3.00GHz          "} , {
 { Input{2048, 2048, 0}, Params{1, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz   "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz          "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz          "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz         "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz         "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz         "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz         "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) HD Graphics 5500 BroadWell U-Processor GT"} , {
 { Input{2048, 2048, 0}, Params{2, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile "} , {
 { Input{2048, 2048, 0}, Params{4, 128, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) HD Graphics IvyBridge M GT2              "} , {
 { Input{2048, 2048, 0}, Params{2, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) HD Graphics Skylake ULT GT2              "} , {
 { Input{2048, 2048, 0}, Params{4, 64, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Iris(R) Xe Graphics                      "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) RaptorLake-S Mobile Graphics Controller  "} , {
 { Input{2048, 2048, 0}, Params{1, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) UHD Graphics 620                         "} , {
 { Input{2048, 2048, 0}, Params{2, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) UHD Graphics 770                         "} , {
 { Input{2048, 2048, 0}, Params{1, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Iris Pro                                          "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "default", {
          { Name{"Intel(R) FPGA Emulation Device                    "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "SM2.0", {
          { Name{"GeForce GTX 580                                   "} , {
 { Input{2048, 2048, 0}, Params{1, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{1, 32, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM3.0", {
          { Name{"GeForce GTX 760 Ti OEM                            "} , {
 { Input{2048, 2048, 0}, Params{2, 32, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 32, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM3.5", {
          { Name{"GeForce 920A                                      "} , {
 { Input{2048, 2048, 0}, Params{2, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM6.0", {
          { Name{"Tesla P100-PCIE-16GB                              "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM6.1", {
          { Name{"GeForce GTX 1070 Ti                               "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 1080 Ti                               "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce MX150                              "} , {
 { Input{2048, 2048, 0}, Params{1, 64, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM7.0", {
          { Name{"Quadro GV100                                      "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla V100-PCIE-16GB                              "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM7.5", {
          { Name{"GeForce GTX 1650                                  "} , {
 { Input{2048, 2048, 0}, Params{2, 32, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce GTX 1650 SUPER                     "} , {
 { Input{2048, 2048, 0}, Params{4, 32, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce GTX 1650 Ti                        "} , {
 { Input{2048, 2048, 0}, Params{2, 128, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2060                           "} , {
 { Input{2048, 2048, 0}, Params{2, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 SUPER                     "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 Super                     "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 with Max-Q Design         "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2080 Ti                        "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2080 with Max-Q Design         "} , {
 { Input{2048, 2048, 0}, Params{1, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Quadro T2000                                      "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"TITAN RTX                                         "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla T4                                          "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{1, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.0", {
          { Name{"A100-PCIE-40GB                                    "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.6", {
          { Name{"NVIDIA GeForce RTX 3050 Ti Laptop GPU             "} , {
 { Input{2048, 2048, 0}, Params{2, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3060 Laptop GPU                "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3070                           "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3070 Ti Laptop GPU             "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080                           "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080 Laptop GPU                "} , {
 { Input{2048, 2048, 0}, Params{2, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080 Ti                        "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3090                           "} , {
 { Input{2048, 2048, 0}, Params{8, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.9", {
          { Name{"NVIDIA GeForce RTX 4070 Laptop GPU                "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4070 Ti                        "} , {
 { Input{2048, 2048, 0}, Params{4, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4080                           "} , {
 { Input{2048, 2048, 0}, Params{2, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4090                           "} , {
 { Input{2048, 2048, 0}, Params{8, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{1, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "OpenCL C 2.0 Adreno(TM) 640", {
          { Name{"QUALCOMM Adreno(TM)                               "} , {
 { Input{2048, 2048, 0}, Params{2, 32, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 32, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 32, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{2048, 2048, 0}, Params{2, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
