// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is a generic GEMM kernel that works for all sizes and configurations: it doesn't require any
// pre and and post-processing kernels.
//
// This kernel is seperated into three files. This is part 1 out of 3.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define GLOBALTOPRIVATEDIRECTA2 GLOBALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATEDIRECTB2 GLOBALTOPRIVATEDIRECTB##PARAMS
#define GLOBALTOPRIVATECHECKEDA2 GLOBALTOPRIVATECHECKEDA##PARAMS
#define GLOBALTOPRIVATECHECKEDB2 GLOBALTOPRIVATECHECKEDB##PARAMS
#define LOCALTOPRIVATEDIRECTA2 LOCALTOPRIVATEDIRECTA##PARAMS
#define LOCALTOPRIVATEDIRECTB2 LOCALTOPRIVATEDIRECTB##PARAMS
#define STORERESULTSDIRECT2 STORERESULTSDIRECT##PARAMS
#define STORERESULTSCHECKED2 STORERESULTSCHECKED##PARAMS
#define GLOBALTOPRIVATEDIRECTA22 GLOBALTOPRIVATEDIRECTA2##PARAMS
#define GLOBALTOPRIVATEDIRECTB22 GLOBALTOPRIVATEDIRECTB2##PARAMS
#define GLOBALTOPRIVATECHECKEDA22 GLOBALTOPRIVATECHECKEDA2##PARAMS
#define GLOBALTOPRIVATECHECKEDB22 GLOBALTOPRIVATECHECKEDB2##PARAMS
#define LOCALTOPRIVATEDIRECTA22 LOCALTOPRIVATEDIRECTA2##PARAMS
#define LOCALTOPRIVATEDIRECTB22 LOCALTOPRIVATEDIRECTB2##PARAMS
#define STORERESULTSDIRECT22 STORERESULTSDIRECT2##PARAMS
#define STORERESULTSCHECKED22 STORERESULTSCHECKED2##PARAMS
#define GET_LOCAL_ID2 GET_LOCAL_ID##PARAMS
#define STORERESULTSCHECKED2 STORERESULTSCHECKED##PARAMS
#define STORERESULTSDIRECT2 STORERESULTSDIRECT##PARAMS
#define LOCALTOPRIVATEDIRECTB2 LOCALTOPRIVATEDIRECTB##PARAMS
#define LOCALTOPRIVATEDIRECTA2 LOCALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATECHECKEDB2 GLOBALTOPRIVATECHECKEDB##PARAMS
#define GLOBALTOPRIVATECHECKEDA2 GLOBALTOPRIVATECHECKEDA##PARAMS
#define GLOBALTOPRIVATEDIRECTB2 GLOBALTOPRIVATEDIRECTB##PARAMS
#define GLOBALTOPRIVATEDIRECTA2 GLOBALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATEDIRECTA GLOBALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATEDIRECTB GLOBALTOPRIVATEDIRECTB##PARAMS
#define GLOBALTOPRIVATECHECKEDA GLOBALTOPRIVATECHECKEDA##PARAMS
#define GLOBALTOPRIVATECHECKEDB GLOBALTOPRIVATECHECKEDB##PARAMS
#define LOCALTOPRIVATEDIRECTA LOCALTOPRIVATEDIRECTA##PARAMS
#define LOCALTOPRIVATEDIRECTB LOCALTOPRIVATEDIRECTB##PARAMS
#define STORERESULTSDIRECT STORERESULTSDIRECT##PARAMS
#define STORERESULTSCHECKED STORERESULTSCHECKED##PARAMS
#define GLOBALTOPRIVATEDIRECTA GLOBALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATEDIRECTB GLOBALTOPRIVATEDIRECTB##PARAMS
#define GLOBALTOPRIVATECHECKEDA GLOBALTOPRIVATECHECKEDA##PARAMS
#define GLOBALTOPRIVATECHECKEDB GLOBALTOPRIVATECHECKEDB##PARAMS
#define LOCALTOPRIVATEDIRECTA LOCALTOPRIVATEDIRECTA##PARAMS
#define LOCALTOPRIVATEDIRECTB LOCALTOPRIVATEDIRECTB##PARAMS
#define STORERESULTSDIRECT STORERESULTSDIRECT##PARAMS
#define STORERESULTSCHECKED STORERESULTSCHECKED##PARAMS
#define STORERESULTSCHECKED STORERESULTSCHECKED##PARAMS
#define STORERESULTSDIRECT STORERESULTSDIRECT##PARAMS
#define LOCALTOPRIVATEDIRECTB LOCALTOPRIVATEDIRECTB##PARAMS
#define LOCALTOPRIVATEDIRECTA LOCALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATECHECKEDB GLOBALTOPRIVATECHECKEDB##PARAMS
#define GLOBALTOPRIVATECHECKEDA GLOBALTOPRIVATECHECKEDA##PARAMS
#define GLOBALTOPRIVATEDIRECTB GLOBALTOPRIVATEDIRECTB##PARAMS
#define GLOBALTOPRIVATEDIRECTA GLOBALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATEDIRECTA2 GLOBALTOPRIVATEDIRECTA##PARAMS
#define I2 i##PARAMS
#define GLOBALTOPRIVATEDIRECTB2 GLOBALTOPRIVATEDIRECTB##PARAMS
#define GLOBALTOPRIVATECHECKEDA2 GLOBALTOPRIVATECHECKEDA##PARAMS
#define GLOBALTOPRIVATECHECKEDB2 GLOBALTOPRIVATECHECKEDB##PARAMS
#define LOCALTOPRIVATEDIRECTA2 LOCALTOPRIVATEDIRECTA##PARAMS
#define LOCALTOPRIVATEDIRECTB2 LOCALTOPRIVATEDIRECTB##PARAMS
#define STORERESULTSDIRECT2 STORERESULTSDIRECT##PARAMS
#define STORERESULTSCHECKED2 STORERESULTSCHECKED##PARAMS
#define GLOBALTOPRIVATEDIRECTA2 CONCATENATE(GLOBALTOPRIVATEDIRECTA,PARAMS)
#define GLOBALTOPRIVATEDIRECTB2 CONCATENATE(GLOBALTOPRIVATEDIRECTB,PARAMS)
#define GLOBALTOPRIVATECHECKEDA2 CONCATENATE(GLOBALTOPRIVATECHECKEDA,PARAMS)
#define GLOBALTOPRIVATECHECKEDB2 CONCATENATE(GLOBALTOPRIVATECHECKEDB,PARAMS)
#define LOCALTOPRIVATEDIRECTA2 CONCATENATE(LOCALTOPRIVATEDIRECTA,PARAMS)
#define LOCALTOPRIVATEDIRECTB2 CONCATENATE(LOCALTOPRIVATEDIRECTB,PARAMS)
#define STORERESULTSDIRECT2 CONCATENATE(STORERESULTSDIRECT,PARAMS)
#define STORERESULTSCHECKED2 CONCATENATE(STORERESULTSCHECKED,PARAMS)
#define GLOBALTOPRIVATEDIRECTA2 CONCATENATE(GLOBALTOPRIVATEDIRECTA,PARAMS)
#define I2 CONCATENATE(i,PARAMS)
#define GLOBALTOPRIVATEDIRECTB2 CONCATENATE(GLOBALTOPRIVATEDIRECTB,PARAMS)
#define GLOBALTOPRIVATECHECKEDA2 CONCATENATE(GLOBALTOPRIVATECHECKEDA,PARAMS)
#define GLOBALTOPRIVATECHECKEDB2 CONCATENATE(GLOBALTOPRIVATECHECKEDB,PARAMS)
#define LOCALTOPRIVATEDIRECTA2 CONCATENATE(LOCALTOPRIVATEDIRECTA,PARAMS)
#define LOCALTOPRIVATEDIRECTB2 CONCATENATE(LOCALTOPRIVATEDIRECTB,PARAMS)
#define STORERESULTSDIRECT2 CONCATENATE(STORERESULTSDIRECT,PARAMS)
#define STORERESULTSCHECKED2 CONCATENATE(STORERESULTSCHECKED,PARAMS)
// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef WGD
  #define WGD 8      // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
#endif
#ifndef MDIMCD
  #define MDIMCD 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMCD
  #define NDIMCD 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMAD
  #define MDIMAD 8    // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#endif
#ifndef NDIMBD
  #define NDIMBD 8    // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#endif
#ifndef KWID
  #define KWID 1      // Unroll factor of the WGD loop (smaller or equal than WGD)
#endif
#ifndef VWMD
  #define VWMD 1      // Vector width of matrices A and C
#endif
#ifndef VWND
  #define VWND 1      // Vector width of matrix B
#endif
#ifndef PADA
  #define PADA 1      // Local memory padding for matrix A
#endif
#ifndef PADB
  #define PADB 1      // Local memory padding for matrix B
#endif
// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)
// =================================================================================================
// Data-widths in dimension M
#if VWMD == 1
	#define realMD real
#elif VWMD == 2
	#define realMD real2
#elif VWMD == 4
	#define realMD real4
#elif VWMD == 8
	#define realMD real8
#elif VWMD == 16
	#define realMD real16
#endif
// Data-widths in dimension N
#if VWND == 1
	#define realND real
#elif VWND == 2
	#define realND real2
#elif VWND == 4
	#define realND real4
#elif VWND == 8
	#define realND real8
#elif VWND == 16
	#define realND real16
#endif
// =================================================================================================
// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix.
INLINE_FUNC real #define GLOBALTOPRIVATEDIRECTA GLOBALTOPRIVATEDIRECTA##PARAMS
#define GLOBALTOPRIVATEDIRECTA GLOBALTOPRIVATEDIRECTA##PARAMS
GLOBALTOPRIVATEDIRECTA2(const __global real* restrict agms, const int _mi,
                                        const int a_ld, const int a_offset, const int idm, const int idk,
                                        const int a_transpose, const int a_conjugate) {
  const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
  real result = agms[a_index + a_offset];
  if (a_conjugate) { COMPLEX_CONJUGATE(result); }
  return result;
}
// Same as above, but now for the B input matrix
INLINE_FUNC real #define GLOBALTOPRIVATEDIRECTB GLOBALTOPRIVATEDIRECTB##PARAMS
#define GLOBALTOPRIVATEDIRECTB GLOBALTOPRIVATEDIRECTB##PARAMS
GLOBALTOPRIVATEDIRECTB2(const __global real* restrict bgms, const int _ni,
                                        const int b_ld, const int b_offset, const int idn, const int idk,
                                        const int b_transpose, const int b_conjugate) {
  const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
  real result = bgms[b_index + b_offset];
  if (b_conjugate) { COMPLEX_CONJUGATE(result); }
  return result;
}
// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
INLINE_FUNC real #define GLOBALTOPRIVATECHECKEDA GLOBALTOPRIVATECHECKEDA##PARAMS
#define GLOBALTOPRIVATECHECKEDA GLOBALTOPRIVATECHECKEDA##PARAMS
GLOBALTOPRIVATECHECKEDA2(const __global real* restrict agms, const int _mi,
                                         const int a_ld, const int a_offset, const int idm, const int idk,
                                         const int a_transpose, const int a_conjugate,
                                         const int kSizeM) {
  real result;
  if (idm + _mi < kSizeM) {
    const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
    result = agms[a_index + a_offset];
    if (a_conjugate) { COMPLEX_CONJUGATE(result); }
  }
  else {
    SetToZero(result);
  }
  return result;
}
// Same as above, but now for the B input matrix
INLINE_FUNC real #define GLOBALTOPRIVATECHECKEDB GLOBALTOPRIVATECHECKEDB##PARAMS
#define GLOBALTOPRIVATECHECKEDB GLOBALTOPRIVATECHECKEDB##PARAMS
GLOBALTOPRIVATECHECKEDB2(const __global real* restrict bgms, const int _ni,
                                         const int b_ld, const int b_offset, const int idn, const int idk,
                                         const int b_transpose, const int b_conjugate,
                                         const int kSizeN) {
  real result;
  if (idn + _ni < kSizeN) {
    const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
    result = bgms[b_index + b_offset];
    if (b_conjugate) { COMPLEX_CONJUGATE(result); }
  }
  else {
    SetToZero(result);
  }
  return result;
}
// =================================================================================================
// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
INLINE_FUNC real #define LOCALTOPRIVATEDIRECTA LOCALTOPRIVATEDIRECTA##PARAMS
#define LOCALTOPRIVATEDIRECTA LOCALTOPRIVATEDIRECTA##PARAMS
LOCALTOPRIVATEDIRECTA2(LOCAL_PTR real* alm, const int _mi, const int kg,
                                       const int a_transpose) {
  const int mg = _mi + GET_LOCAL_ID2(0)*MWID;
  const int index = (a_transpose) ? mg*(WGD + PADA) + kg : kg*(WGD + PADA) + mg;
  return alm[index];
}
// Same as above, but now for the B input matrix
INLINE_FUNC real #define LOCALTOPRIVATEDIRECTB LOCALTOPRIVATEDIRECTB##PARAMS
#define LOCALTOPRIVATEDIRECTB LOCALTOPRIVATEDIRECTB##PARAMS
LOCALTOPRIVATEDIRECTB2(LOCAL_PTR real* blm, const int _ni, const int kg,
                                       const int b_transpose) {
  const int ng = _ni + GET_LOCAL_ID2(1)*NWID;
  const int index = (b_transpose) ? ng*(WGD + PADB) + kg : kg*(WGD + PADB) + ng;
  return blm[index];
}
// =================================================================================================
// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void #define STORERESULTSDIRECT STORERESULTSDIRECT##PARAMS
#define STORERESULTSDIRECT STORERESULTSDIRECT##PARAMS
STORERESULTSDIRECT2(__global real* cgm, const real c_value,
                                    const int _mi, const int _ni, const int idm, const int idn,
                                    const real alpha, const real beta,
                                    const int c_ld, const int c_offset, const int c_transpose) {
  // Determines the destination index
  int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);
  // The final multiplication with alpha (in case beta == 0)
  real result;
  if (IsZero(beta)) {
    Multiply(result, alpha, c_value);
  }
  // The final multiplication with alpha and the addition with beta*C
  else {
    AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);
  }
  cgm[c_index + c_offset] = result;
}
// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void #define STORERESULTSCHECKED STORERESULTSCHECKED##PARAMS
#define STORERESULTSCHECKED STORERESULTSCHECKED##PARAMS
STORERESULTSCHECKED2(__global real* cgm, const real c_value,
                                     const int _mi, const int _ni, const int idm, const int idn,
                                     const int kSizeM, const int kSizeN,
                                     const real alpha, const real beta,
                                     const int c_ld, const int c_offset, const int c_transpose) {
  if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN) {
    // Deter_mines the destination index
    int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);
    // The final multiplication with alpha (in case beta == 0)
    real result;
    if (IsZero(beta)) {
      Multiply(result, alpha, c_value);
    }
    // The final multiplication with alpha and the addition with beta*C
    else {
      AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);
    }
    cgm[c_index + c_offset] = result;
  }
}
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================
