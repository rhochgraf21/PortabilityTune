// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the batched version of the direct GEMM kernels. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define XGEMMDIRECTBATCHEDNN2 XGEMMDIRECTBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDNT2 XGEMMDIRECTBATCHEDNT##PARAMS
#define XGEMMDIRECTBATCHEDTN2 XGEMMDIRECTBATCHEDTN##PARAMS
#define XGEMMDIRECTBATCHEDTT2 XGEMMDIRECTBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN2 XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT2 XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN2 XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT2 XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
#define XGEMMDIRECTBATCHEDNN22 XGEMMDIRECTBATCHEDNN2##PARAMS
#define XGEMMDIRECTBATCHEDNT22 XGEMMDIRECTBATCHEDNT2##PARAMS
#define XGEMMDIRECTBATCHEDTN22 XGEMMDIRECTBATCHEDTN2##PARAMS
#define XGEMMDIRECTBATCHEDTT22 XGEMMDIRECTBATCHEDTT2##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN22 XGEMMDIRECTSTRIDEDBATCHEDNN2##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT22 XGEMMDIRECTSTRIDEDBATCHEDNT2##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN22 XGEMMDIRECTSTRIDEDBATCHEDTN2##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT22 XGEMMDIRECTSTRIDEDBATCHEDTT2##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT2 XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN2 XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT2 XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN2 XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDTT2 XGEMMDIRECTBATCHEDTT##PARAMS
#define XGEMMDIRECTBATCHEDTN2 XGEMMDIRECTBATCHEDTN##PARAMS
#define XGEMMDIRECTBATCHEDNT2 XGEMMDIRECTBATCHEDNT##PARAMS
#define XGEMMDIRECTBATCHEDNN2 XGEMMDIRECTBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDNN XGEMMDIRECTBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDNT XGEMMDIRECTBATCHEDNT##PARAMS
#define XGEMMDIRECTBATCHEDTN XGEMMDIRECTBATCHEDTN##PARAMS
#define XGEMMDIRECTBATCHEDTT XGEMMDIRECTBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
#define XGEMMDIRECTBATCHEDNN XGEMMDIRECTBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDNT XGEMMDIRECTBATCHEDNT##PARAMS
#define XGEMMDIRECTBATCHEDTN XGEMMDIRECTBATCHEDTN##PARAMS
#define XGEMMDIRECTBATCHEDTT XGEMMDIRECTBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDTT XGEMMDIRECTBATCHEDTT##PARAMS
#define XGEMMDIRECTBATCHEDTN XGEMMDIRECTBATCHEDTN##PARAMS
#define XGEMMDIRECTBATCHEDNT XGEMMDIRECTBATCHEDNT##PARAMS
#define XGEMMDIRECTBATCHEDNN XGEMMDIRECTBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDNN2 XGEMMDIRECTBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDNT2 XGEMMDIRECTBATCHEDNT##PARAMS
#define XGEMMDIRECTBATCHEDTN2 XGEMMDIRECTBATCHEDTN##PARAMS
#define XGEMMDIRECTBATCHEDTT2 XGEMMDIRECTBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN2 XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT2 XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN2 XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT2 XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
#define XGEMMDIRECTBATCHEDNN2 CONCATENATE(XGEMMDIRECTBATCHEDNN,PARAMS)
#define XGEMMDIRECTBATCHEDNT2 CONCATENATE(XGEMMDIRECTBATCHEDNT,PARAMS)
#define XGEMMDIRECTBATCHEDTN2 CONCATENATE(XGEMMDIRECTBATCHEDTN,PARAMS)
#define XGEMMDIRECTBATCHEDTT2 CONCATENATE(XGEMMDIRECTBATCHEDTT,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDNN2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDNN,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDNT2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDNT,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDTN2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDTN,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDTT2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDTT,PARAMS)
#define XGEMMDIRECTBATCHEDNN2 CONCATENATE(XGEMMDIRECTBATCHEDNN,PARAMS)
#define XGEMMDIRECTBATCHEDNT2 CONCATENATE(XGEMMDIRECTBATCHEDNT,PARAMS)
#define XGEMMDIRECTBATCHEDTN2 CONCATENATE(XGEMMDIRECTBATCHEDTN,PARAMS)
#define XGEMMDIRECTBATCHEDTT2 CONCATENATE(XGEMMDIRECTBATCHEDTT,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDNN2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDNN,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDNT2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDNT,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDTN2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDTN,PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDTT2 CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDTT,PARAMS)
// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)
// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTBATCHEDNN XGEMMDIRECTBATCHEDNN##PARAMS
#define XGEMMDIRECTBATCHEDNN XGEMMDIRECTBATCHEDNN##PARAMS
XGEMMDIRECTBATCHEDNN2(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTBATCHEDNT XGEMMDIRECTBATCHEDNT##PARAMS
#define XGEMMDIRECTBATCHEDNT XGEMMDIRECTBATCHEDNT##PARAMS
XGEMMDIRECTBATCHEDNT2(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the batched GEMM kernel with [A, B] = [transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTBATCHEDTN XGEMMDIRECTBATCHEDTN##PARAMS
#define XGEMMDIRECTBATCHEDTN XGEMMDIRECTBATCHEDTN##PARAMS
XGEMMDIRECTBATCHEDTN2(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the batched GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTBATCHEDTT XGEMMDIRECTBATCHEDTT##PARAMS
#define XGEMMDIRECTBATCHEDTT XGEMMDIRECTBATCHEDTT##PARAMS
XGEMMDIRECTBATCHEDTT2(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}
#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)
// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTSTRIDEDBATCHEDNN XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN XGEMMDIRECTSTRIDEDBATCHEDNN##PARAMS
XGEMMDIRECTSTRIDEDBATCHEDNN2(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTSTRIDEDBATCHEDNT XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT XGEMMDIRECTSTRIDEDBATCHEDNT##PARAMS
XGEMMDIRECTSTRIDEDBATCHEDNT2(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTSTRIDEDBATCHEDTN XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN XGEMMDIRECTSTRIDEDBATCHEDTN##PARAMS
XGEMMDIRECTSTRIDEDBATCHEDTN2(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTSTRIDEDBATCHEDTT XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT XGEMMDIRECTSTRIDEDBATCHEDTT##PARAMS
XGEMMDIRECTSTRIDEDBATCHEDTT2(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = GET_GROUP_ID2(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}
#endif
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================
