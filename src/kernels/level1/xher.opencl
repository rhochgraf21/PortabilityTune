// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xher kernels for rank-1 matrix update.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define XHE2 XHE##PARAMS
#define XHER2 XHER##PARAMS
#define LOADVECTOR2 LOADVECTOR##PARAMS
#define GET_GLOBAL_ID2 GET_GLOBAL_ID##PARAMS
#define GET_GLOBAL_SIZE2 GET_GLOBAL_SIZE##PARAMS
#define XHER XHER##PARAMS
#define XHER XHER##PARAMS
#define XHER XHER##PARAMS
#define XHE2 CONCATENATE(XHE,PARAMS)
// =================================================================================================
// Symmetric version of the rank-1 matrix update kernel (HER, HPR, SYR, SPR)
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(WGS1, WGS2, 1)))
#endif
void #define XHER XHER##PARAMS
#define XHER XHER##PARAMS
XHER(const int n,
          const real_arg arg_alpha,
          const __global real* restrict xgm, const int x_offset, const int x_inc,
          __global real* restrict agm, const int a_offset, const int a_ld,
          const int is_upper, const int is_rowmajor) {
  const real alpha = GetRealArg(arg_alpha);
  // Register storage for X and XT
  #pragma promote_to_registers
  real xvalues[WPT];
  #pragma promote_to_registers
  real xtvalues[WPT];
  // Loads the X-vector
  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id2 = _w*GET_GLOBAL_SIZE2(1) + GET_GLOBAL_ID2(1);
    xvalues[_w] = LOADVECTOR2(id2, n, xgm, x_offset, x_inc, !is_rowmajor);
  }
  // Loads the X-transposed-vector
  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id1 = _w*GET_GLOBAL_SIZE2(0) + GET_GLOBAL_ID2(0);
    xtvalues[_w] = LOADVECTOR2(id1, n, xgm, x_offset, x_inc, is_rowmajor);
  }
  // Loops over the work per thread twice
  #pragma unroll
  for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
    #pragma unroll
    for (int _w2 = 0; _w2 < WPT; _w2 += 1) {
      // Global thread IDs
      const int id1 = _w1*GET_GLOBAL_SIZE2(0) + GET_GLOBAL_ID2(0);
      const int id2 = _w2*GET_GLOBAL_SIZE2(1) + GET_GLOBAL_ID2(1);
      // Skip these threads if they do not contain threads contributing to the matrix-triangle
      if ((is_upper && (id1 > id2)) || (!is_upper && (id2 > id1))) {
        // Do nothing
      }
      // Loads A, performs the operation, and stores the result into A
      else {
        MATRIXUPDATE(id1, id2, n, n, agm, a_offset, a_ld, alpha, xvalues[_w2], xtvalues[_w1], is_upper);
      }
    }
  }
}
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================
