// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to convert symmetric matrices to/from general matrices.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define SYMMLOWERTOSQUARED2 SYMMLOWERTOSQUARED##PARAMS
#define SYMMUPPERTOSQUARED2 SYMMUPPERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED22 SYMMLOWERTOSQUARED2##PARAMS
#define SYMMUPPERTOSQUARED22 SYMMUPPERTOSQUARED2##PARAMS
#define GET_GROUP_ID2 GET_GROUP_ID##PARAMS
#define SYMMUPPERTOSQUARED2 SYMMUPPERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED2 SYMMLOWERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED SYMMLOWERTOSQUARED##PARAMS
#define SYMMUPPERTOSQUARED SYMMUPPERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED SYMMLOWERTOSQUARED##PARAMS
#define SYMMUPPERTOSQUARED SYMMUPPERTOSQUARED##PARAMS
#define SYMMUPPERTOSQUARED SYMMUPPERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED SYMMLOWERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED2 SYMMLOWERTOSQUARED##PARAMS
#define SYMMUPPERTOSQUARED2 SYMMUPPERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED2 CONCATENATE(SYMMLOWERTOSQUARED,PARAMS)
#define SYMMUPPERTOSQUARED2 CONCATENATE(SYMMUPPERTOSQUARED,PARAMS)
#define SYMMLOWERTOSQUARED2 CONCATENATE(SYMMLOWERTOSQUARED,PARAMS)
#define SYMMUPPERTOSQUARED2 CONCATENATE(SYMMUPPERTOSQUARED,PARAMS)
// =================================================================================================
#if defined(ROUTINE_SYMM)
// Kernel to populate a squared symmetric matrix, given that the triangle which holds the data is
// stored as the lower-triangle of the input matrix. This uses the padding kernel's parameters.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
#endif
void #define SYMMLOWERTOSQUARED SYMMLOWERTOSQUARED##PARAMS
#define SYMMLOWERTOSQUARED SYMMLOWERTOSQUARED##PARAMS
SYMMLOWERTOSQUARED2(const int src_dim,
                        const int src_ld, const int src_offset,
                        __global const real* restrict src,
                        const int dest_dim,
                        const int dest_ld, const int dest_offset,
                        __global real* dest) {
  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
    const int id_one = (GET_GROUP_ID2(0)*PAD_WPTX + _w_one) * PAD_DIMX + GET_LOCAL_ID2(0);
    #pragma unroll
    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
      const int id_two = (GET_GROUP_ID2(1)*PAD_WPTY + _w_two) * PAD_DIMY + GET_LOCAL_ID2(1);
      if (id_two < dest_dim && id_one < dest_dim) {
        // Loads data from the lower-symmetric matrix
        real result;
        SetToZero(result);
        if (id_two < src_dim && id_one < src_dim) {
          if (id_two <= id_one) { result = src[id_two*src_ld + id_one + src_offset]; }
          else                  { result = src[id_one*src_ld + id_two + src_offset]; }
        }
        // Stores the result in the destination matrix
        dest[id_two*dest_ld + id_one + dest_offset] = result;
      }
    }
  }
}
// Same as above, but now the matrix' data is stored in the upper-triangle
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
#endif
void #define SYMMUPPERTOSQUARED SYMMUPPERTOSQUARED##PARAMS
#define SYMMUPPERTOSQUARED SYMMUPPERTOSQUARED##PARAMS
SYMMUPPERTOSQUARED2(const int src_dim,
                        const int src_ld, const int src_offset,
                        __global const real* restrict src,
                        const int dest_dim,
                        const int dest_ld, const int dest_offset,
                        __global real* dest) {
  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
    const int id_one = (GET_GROUP_ID2(0)*PAD_WPTX + _w_one) * PAD_DIMX + GET_LOCAL_ID2(0);
    #pragma unroll
    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
      const int id_two = (GET_GROUP_ID2(1)*PAD_WPTY + _w_two) * PAD_DIMY + GET_LOCAL_ID2(1);
      if (id_two < dest_dim && id_one < dest_dim) {
        // Loads data from the upper-symmetric matrix
        real result;
        SetToZero(result);
        if (id_two < src_dim && id_one < src_dim) {
          if (id_one <= id_two) { result = src[id_two*src_ld + id_one + src_offset]; }
          else                  { result = src[id_one*src_ld + id_two + src_offset]; }
        }
        // Stores the result in the destination matrix
        dest[id_two*dest_ld + id_one + dest_offset] = result;
      }
    }
  }
}
#endif
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================
