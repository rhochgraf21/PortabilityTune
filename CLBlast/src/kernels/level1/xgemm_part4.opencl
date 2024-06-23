// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define XGEMMUPPER2 XGEMMUPPER##PARAMS
#define XGEMMLOWER2 XGEMMLOWER##PARAMS
#define XGEMM2 XGEMM##PARAMS
#define XGEMMUPPER22 XGEMMUPPER2##PARAMS
#define XGEMMLOWER22 XGEMMLOWER2##PARAMS
#define XGEMM22 XGEMM2##PARAMS
#define GETGROUPID02 GETGROUPID0##PARAMS
#define GETGROUPID12 GETGROUPID1##PARAMS
#define XGEMMBODY2 XGEMMBODY##PARAMS
#define XGEMM2 XGEMM##PARAMS
#define XGEMMLOWER2 XGEMMLOWER##PARAMS
#define XGEMMUPPER2 XGEMMUPPER##PARAMS
#define XGEMMUPPER XGEMMUPPER##PARAMS
#define XGEMMLOWER XGEMMLOWER##PARAMS
#define XGEMM XGEMM##PARAMS
#define XGEMMUPPER XGEMMUPPER##PARAMS
#define XGEMMLOWER XGEMMLOWER##PARAMS
#define XGEMM XGEMM##PARAMS
#define XGEMM XGEMM##PARAMS
#define XGEMMLOWER XGEMMLOWER##PARAMS
#define XGEMMUPPER XGEMMUPPER##PARAMS
#define XGEMMUPPER2 XGEMMUPPER##PARAMS
#define XGEMMLOWER2 XGEMMLOWER##PARAMS
#define XGEMM2 XGEMM##PARAMS
#define XGEMMUPPER2 CONCATENATE(XGEMMUPPER,PARAMS)
#define XGEMMLOWER2 CONCATENATE(XGEMMLOWER,PARAMS)
#define XGEMM2 CONCATENATE(XGEMM,PARAMS)
#define XGEMMUPPER2 CONCATENATE(XGEMMUPPER,PARAMS)
#define XGEMMLOWER2 CONCATENATE(XGEMMLOWER,PARAMS)
#define XGEMM2 CONCATENATE(XGEMM,PARAMS)
// The upper-triangular and lower-triangular kernels are only used in special cases
#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
// Main entry point of the kernel. This is the upper-triangular version.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
void #define XGEMMUPPER XGEMMUPPER##PARAMS
#define XGEMMUPPER XGEMMUPPER##PARAMS
XGEMMUPPER2(const int kSizeN, const int kSizeK,
                const real_arg arg_alpha,
                const real_arg arg_beta,
                const __global realM* restrict agm,
                const __global realN* restrict bgm,
                __global realM* cgm) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);
  // Skip these threads if they do not contain threads contributing to the upper-triangle
  if ((GETGROUPID12() + 1)*NWG < GETGROUPID02()*MWG) {
    return;
  }
  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif
  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}
// Main entry point of the kernel. This is the lower-triangular version.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
void #define XGEMMLOWER XGEMMLOWER##PARAMS
#define XGEMMLOWER XGEMMLOWER##PARAMS
XGEMMLOWER2(const int kSizeN, const int kSizeK,
                const real_arg arg_alpha,
                const real_arg arg_beta,
                const __global realM* restrict agm,
                const __global realN* restrict bgm,
                __global realM* cgm) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);
  // Skip these threads if they do not contain threads contributing to the lower-triangle
  if (GETGROUPID12()*NWG > (GETGROUPID02() + 1)*MWG) {
    return;
  }
  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif
  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XGEMMBODY2(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}
// =================================================================================================
// If not using a triangular version, include the regular kernel
#else
// Main entry point of the kernel. This is the regular full version.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
#endif
void #define XGEMM XGEMM##PARAMS
#define XGEMM XGEMM##PARAMS
XGEMM2(const int kSizeM, const int kSizeN, const int kSizeK,
           const real_arg arg_alpha,
           const real_arg arg_beta,
           const __global realM* restrict agm,
           const __global realN* restrict bgm,
           __global realM* cgm,
           const int b_offset, const int c_offset) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);
  // Adds the offsets (in case of use of a single temporary buffer for A, B, and C)
  bgm = &bgm[b_offset];
  cgm = &cgm[c_offset];
  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif
  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XGEMMBODY2(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XGEMMBODY2(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XGEMMBODY2(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XGEMMBODY2(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}
#endif
)"
// End of the C++11 raw string literal
// =================================================================================================
