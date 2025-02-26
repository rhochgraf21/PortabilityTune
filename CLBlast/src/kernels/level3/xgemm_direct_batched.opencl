
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
R"(#define XGEMMDIRECTBATCHEDTN2 XgemmDirectBatchedTN
#define XGEMMDIRECTBATCHEDTN CONCATENATE(XGEMMDIRECTBATCHEDTN2, PARAMS)
#define XGEMMDIRECTBATCHEDTT2 XgemmDirectBatchedTT
#define XGEMMDIRECTBATCHEDTT CONCATENATE(XGEMMDIRECTBATCHEDTT2, PARAMS)
#define XGEMMDIRECTBATCHEDNN2 XgemmDirectBatchedNN
#define XGEMMDIRECTBATCHEDNN CONCATENATE(XGEMMDIRECTBATCHEDNN2, PARAMS)
#define XGEMMDIRECTBATCHEDNT2 XgemmDirectBatchedNT
#define XGEMMDIRECTBATCHEDNT CONCATENATE(XGEMMDIRECTBATCHEDNT2, PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDTN2 XgemmDirectStridedBatchedTN
#define XGEMMDIRECTSTRIDEDBATCHEDTN CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDTN2, PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDNT2 XgemmDirectStridedBatchedNT
#define XGEMMDIRECTSTRIDEDBATCHEDNT CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDNT2, PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDTT2 XgemmDirectStridedBatchedTT
#define XGEMMDIRECTSTRIDEDBATCHEDTT CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDTT2, PARAMS)
#define XGEMMDIRECTSTRIDEDBATCHEDNN2 XgemmDirectStridedBatchedNN
#define XGEMMDIRECTSTRIDEDBATCHEDNN CONCATENATE(XGEMMDIRECTSTRIDEDBATCHEDNN2, PARAMS)


// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XGEMMDIRECTBATCHEDNN(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XGEMMDIRECTBATCHEDNT(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the batched GEMM kernel with [A, B] = [transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XGEMMDIRECTBATCHEDTN(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the batched GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XGEMMDIRECTBATCHEDTT(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __constant real_arg* arg_alphas, const __constant real_arg* arg_betas,
                          const __global realMD* restrict agm, const __constant int* a_offsets, const int a_ld,
                          const __global realND* restrict bgm, const __constant int* b_offsets, const int b_ld,
                          __global real* cgm, const __constant int* c_offsets, const int c_ld,
                          const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = arg_alphas[batch];
  const real_arg arg_beta = arg_betas[batch];
  const int a_offset = a_offsets[batch];
  const int b_offset = b_offsets[batch];
  const int c_offset = c_offsets[batch];
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
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
void XGEMMDIRECTSTRIDEDBATCHEDNN(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XGEMMDIRECTSTRIDEDBATCHEDNT(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XGEMMDIRECTSTRIDEDBATCHEDTN(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XGEMMDIRECTSTRIDEDBATCHEDTT(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const real_arg arg_alpha, const real_arg arg_beta,
                                 const __global realMD* restrict agm, const int a_offset, const int a_ld, const int a_stride,
                                 const __global realND* restrict bgm, const int b_offset, const int b_ld, const int b_stride,
                                 __global real* cgm, const int c_offset, const int c_ld, const int c_stride,
                                 const int c_transpose, const int a_conjugate, const int b_conjugate) {
  const int batch = get_group_id(2);
  const int a_offset_batch = a_offset + a_stride * batch;
  const int b_offset_batch = b_offset + b_stride * batch;
  const int c_offset_batch = c_offset + c_stride * batch;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
