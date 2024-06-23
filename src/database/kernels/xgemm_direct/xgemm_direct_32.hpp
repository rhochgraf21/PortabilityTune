
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xgemm_Direct32' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry XgemmDirectSingle = {
  "XgemmDirect", Precision::kSingle, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere", {
          { Name{"AMD Radeon RX 480                                 "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 32, 32, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 580 2048SP                          "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX590 GME                              "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 32, 32, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 32, 32, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Fiji", {
          { Name{"AMD Radeon R9 Fury X                              "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon R9 M370X Compute Engine                "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Hawaii", {
          { Name{"AMD FirePro W8100                                 "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 32, 32, 1, 1, 4, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 32, 32, 1, 1, 4, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Tonga", {
          { Name{"AMD Radeon R9 380                                 "} , {
 { Input{256, 256, 256}, Params{16, 16, 16, 32, 8, 0, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{16, 16, 16, 32, 8, 0, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Turks", {
          { Name{"AMD Radeon HD 6770M                               "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "Vancouver", {
          { Name{"ATI Radeon HD 6750M                               "} , {
 { Input{256, 256, 256}, Params{8, 8, 16, 8, 8, 1, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{8, 8, 16, 8, 8, 1, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { Name{"AMD Radeon Pro 450 Compute Engine                 "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon Pro 580 Compute Engine                 "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 2, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 2, 2, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700                                "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 5700 XT                             "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon RX 6900 XT                             "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 2, 2, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1031", {
          { Name{"AMD Radeon RX 6700 XT                             "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1032", {
          { Name{"AMD Radeon RX 6600 XT                             "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "} , {
 { Input{256, 256, 256}, Params{8, 16, 16, 32, 16, 1, 1, 4, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{8, 16, 16, 32, 16, 1, 1, 4, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1035", {
          { Name{"AMD Radeon Graphics                               "} , {
 { Input{256, 256, 256}, Params{8, 8, 8, 8, 8, 1, 1, 8, 4, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{8, 8, 8, 8, 8, 1, 1, 8, 4, 64, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1100", {
          { Name{"Radeon RX 7900 XTX                                "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx1102", {
          { Name{"AMD Radeon RX 7600                                "} , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx902", {
          { Name{"AMD Radeon(TM) Graphics                           "} , {
 { Input{256, 256, 256}, Params{8, 16, 8, 16, 8, 1, 0, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"AMD Radeon(TM) RX Vega 10 Graphics                "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 32, 8, 1, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx906:sramecc+:xnack-", {
          { Name{"AMD Radeon VII                                    "} , {
 { Input{256, 256, 256}, Params{16, 16, 16, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{16, 16, 16, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "gfx90c", {
          { Name{"AMD Radeon(TM) Graphics                           "} , {
 { Input{256, 256, 256}, Params{8, 16, 8, 16, 8, 1, 0, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "default", {
          { Name{"Mali-T628                                         "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Mali-T760                                         "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Apple GPUs
      kDeviceTypeGPU, "Apple", {
        { "default", {
          { Name{"Apple M1                                          "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 16, 16, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
 // KWID=2 MDIMAD=8 MDIMCD=8 NDIMBD=16 NDIMCD=16 PADA=1 PADB=1 PRECISION=32 VWMD=2 VWND=1 WGD=16
          { Name{"Apple M1 Max                                      "} , {
   // { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
   // { Input{256, 256, 256}, Params{16, 16, 8, 8, 8, 0, 0, 1, 8, 64, 0, 0, 0, 0, 0, 0 } },
   //{ Input{513, 256, 256}, Params{2, 16, 16, 4, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
   //{ Input{514, 256, 256}, Params{2, 16, 16, 2, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
   //{ Input{515, 256, 256}, Params{2, 16, 16, 1, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
   // { Input{512, 512, 512}, Params{2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0 } },
   { Input{513, 512, 512}, Params{2, 8, 8, 16, 16, 1, 1, 2, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Apple M2 Max                                      "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Imagination Technologies GPUs
      kDeviceTypeGPU, "Imagination Technologies", {
        { "default", {
          { Name{"PowerVR B-Series BXE-4-32                         "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz         "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 0, 0, 1, 8, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz           "} , {
 { Input{256, 256, 256}, Params{8, 16, 16, 16, 16, 0, 0, 1, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-4590S CPU @ 3.00GHz          "} , {
 { Input{256, 256, 256}, Params{8, 8, 8, 8, 8, 0, 0, 8, 4, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz          "} , {
 { Input{256, 256, 256}, Params{2, 32, 32, 32, 32, 0, 0, 1, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7 CPU         920  @ 2.67GHz   "} , {
 { Input{256, 256, 256}, Params{16, 16, 8, 8, 8, 0, 0, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz          "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 0, 0, 2, 2, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz         "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 8, 0, 0, 4, 4, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz         "} , {
 { Input{256, 256, 256}, Params{8, 32, 32, 32, 16, 0, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz         "} , {
 { Input{256, 256, 256}, Params{16, 8, 16, 16, 16, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz         "} , {
 { Input{256, 256, 256}, Params{16, 8, 16, 16, 16, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) HD Graphics 6000 BroadWell U-Processor GT"} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) HD Graphics 620                          "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) HD Graphics IvyBridge M GT2              "} , {
 { Input{256, 256, 256}, Params{8, 16, 32, 16, 8, 1, 0, 1, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) HD Graphics Skylake ULT GT2              "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) Iris(R) Xe Graphics                      "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) RaptorLake-S Mobile Graphics Controller  "} , {
 { Input{256, 256, 256}, Params{8, 16, 16, 8, 8, 0, 0, 4, 2, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) UHD Graphics 620                         "} , {
 { Input{256, 256, 256}, Params{8, 32, 16, 8, 8, 1, 0, 2, 8, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Intel(R) UHD Graphics 770                         "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Iris Pro                                          "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "default", {
          { Name{"Intel(R) FPGA Emulation Device                    "} , {
 { Input{256, 256, 256}, Params{2, 8, 32, 8, 8, 0, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 32, 8, 8, 0, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "SM2.0", {
          { Name{"GeForce GTX 580                                   "} , {
 { Input{256, 256, 256}, Params{2, 16, 8, 32, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 8, 32, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM3.0", {
          { Name{"GeForce GT 650M                                   "} , {
 { Input{256, 256, 256}, Params{16, 16, 16, 8, 16, 1, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 760 Ti OEM                            "} , {
 { Input{256, 256, 256}, Params{16, 32, 8, 16, 16, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM3.5", {
          { Name{"GeForce 920A                                      "} , {
 { Input{256, 256, 256}, Params{16, 32, 8, 32, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX TITAN Black                           "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM5.0", {
          { Name{"GeForce 920MX                                     "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 750 Ti                                "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM5.2", {
          { Name{"GeForce GTX 970                                   "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 32, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 32, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM6.0", {
          { Name{"Tesla P100-PCIE-16GB                              "} , {
 { Input{256, 256, 256}, Params{16, 8, 8, 16, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{16, 8, 8, 16, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM6.1", {
          { Name{"GeForce GTX 1070 Ti                               "} , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 1080                                  "} , {
 { Input{256, 256, 256}, Params{16, 16, 8, 16, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"GeForce GTX 1080 Ti                               "} , {
 { Input{256, 256, 256}, Params{16, 8, 8, 16, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce MX150                              "} , {
 { Input{256, 256, 256}, Params{8, 32, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"TITAN X (Pascal)                                  "} , {
 { Input{256, 256, 256}, Params{8, 32, 8, 8, 16, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM7.0", {
          { Name{"Quadro GV100                                      "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 16, 16, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla V100-PCIE-16GB                              "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 16, 16, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 16, 16, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM7.5", {
          { Name{"GeForce GTX 1650                                  "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce GTX 1650 SUPER                     "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce GTX 1650 Ti                        "} , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 16, 1, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2060                           "} , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 16, 1, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 SUPER                     "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 Super                     "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2070 with Max-Q Design         "} , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 16, 1, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2080 Ti                        "} , {
 { Input{256, 256, 256}, Params{8, 16, 8, 16, 8, 1, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 2080 with Max-Q Design         "} , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 16, 1, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Quadro T2000                                      "} , {
 { Input{256, 256, 256}, Params{8, 16, 8, 16, 8, 1, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"TITAN RTX                                         "} , {
 { Input{256, 256, 256}, Params{16, 8, 16, 32, 8, 1, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"Tesla T4                                          "} , {
 { Input{256, 256, 256}, Params{8, 16, 8, 16, 8, 1, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 8, 8, 16, 1, 1, 2, 1, 64, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.0", {
          { Name{"A100-PCIE-40GB                                    "} , {
 { Input{256, 256, 256}, Params{8, 8, 16, 32, 16, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{8, 8, 16, 32, 16, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.6", {
          { Name{"NVIDIA GeForce RTX 3050 Ti Laptop GPU             "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3060 Laptop GPU                "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3070                           "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3070 Ti Laptop GPU             "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080                           "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080 Laptop GPU                "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3080 Ti                        "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 3090                           "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "SM8.9", {
          { Name{"NVIDIA GeForce RTX 4070 Laptop GPU                "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4070 Ti                        "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4080                           "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { Name{"NVIDIA GeForce RTX 4090                           "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "default", {
          { Name{"QUALCOMM Adreno(TM)                               "} , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 4, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "OpenCL C 2.0 Adreno(TM) 640", {
          { Name{"QUALCOMM Adreno(TM)                               "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 730", {
          { Name{"QUALCOMM Adreno(TM)                               "} , {
 { Input{256, 256, 256}, Params{8, 16, 16, 8, 8, 0, 0, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{8, 16, 16, 8, 8, 0, 0, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 740", {
          { Name{"QUALCOMM Adreno(TM)                               "} , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 16, 16, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                         , {
 { Input{256, 256, 256}, Params{2, 8, 8, 8, 8, 1, 1, 2, 2, 16, 0, 0, 0, 0, 0, 0 } },
 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
