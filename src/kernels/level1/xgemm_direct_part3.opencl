// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define XGEMMDIRECT2 XGEMMDIRECT##PARAMS
#define XGEMMDIRECTNN2 XGEMMDIRECTNN##PARAMS
#define XGEMMDIRECTNT2 XGEMMDIRECTNT##PARAMS
#define XGEMMDIRECTTN2 XGEMMDIRECTTN##PARAMS
#define XGEMMDIRECTTT2 XGEMMDIRECTTT##PARAMS
#define XGEMMDIRECT22 XGEMMDIRECT2##PARAMS
#define XGEMMDIRECTNN22 XGEMMDIRECTNN2##PARAMS
#define XGEMMDIRECTNT22 XGEMMDIRECTNT2##PARAMS
#define XGEMMDIRECTTN22 XGEMMDIRECTTN2##PARAMS
#define XGEMMDIRECTTT22 XGEMMDIRECTTT2##PARAMS
#define GLOBALTOLOCALCHECKEDB2 GLOBALTOLOCALCHECKEDB##PARAMS
#define GLOBALTOLOCALCHECKEDA2 GLOBALTOLOCALCHECKEDA##PARAMS
#define GLOBALTOLOCALSCALARB2 GLOBALTOLOCALSCALARB##PARAMS
#define GLOBALTOLOCALSCALARA2 GLOBALTOLOCALSCALARA##PARAMS
#define GLOBALTOLOCALDIRECTB2 GLOBALTOLOCALDIRECTB##PARAMS
#define GLOBALTOLOCALDIRECTA2 GLOBALTOLOCALDIRECTA##PARAMS
#define XGEMMDIRECTTT2 XGEMMDIRECTTT##PARAMS
#define XGEMMDIRECTTN2 XGEMMDIRECTTN##PARAMS
#define XGEMMDIRECTNT2 XGEMMDIRECTNT##PARAMS
#define XGEMMDIRECTNN2 XGEMMDIRECTNN##PARAMS
#define XGEMMDIRECT2 XGEMMDIRECT##PARAMS
#define XGEMMDIRECT XGEMMDIRECT##PARAMS
#define XGEMMDIRECTNN XGEMMDIRECTNN##PARAMS
#define XGEMMDIRECTNT XGEMMDIRECTNT##PARAMS
#define XGEMMDIRECTTN XGEMMDIRECTTN##PARAMS
#define XGEMMDIRECTTT XGEMMDIRECTTT##PARAMS
#define XGEMMDIRECT XGEMMDIRECT##PARAMS
#define XGEMMDIRECTNN XGEMMDIRECTNN##PARAMS
#define XGEMMDIRECTNT XGEMMDIRECTNT##PARAMS
#define XGEMMDIRECTTN XGEMMDIRECTTN##PARAMS
#define XGEMMDIRECTTT XGEMMDIRECTTT##PARAMS
#define XGEMMDIRECTTT XGEMMDIRECTTT##PARAMS
#define XGEMMDIRECTTN XGEMMDIRECTTN##PARAMS
#define XGEMMDIRECTNT XGEMMDIRECTNT##PARAMS
#define XGEMMDIRECTNN XGEMMDIRECTNN##PARAMS
#define XGEMMDIRECT XGEMMDIRECT##PARAMS
#define XGEMMDIRECT2 XGEMMDIRECT##PARAMS
#define XGEMMDIRECTNN2 XGEMMDIRECTNN##PARAMS
#define XGEMMDIRECTNT2 XGEMMDIRECTNT##PARAMS
#define XGEMMDIRECTTN2 XGEMMDIRECTTN##PARAMS
#define XGEMMDIRECTTT2 XGEMMDIRECTTT##PARAMS
#define XGEMMDIRECT2 CONCATENATE(XGEMMDIRECT,PARAMS)
#define XGEMMDIRECTNN2 CONCATENATE(XGEMMDIRECTNN,PARAMS)
#define XGEMMDIRECTNT2 CONCATENATE(XGEMMDIRECTNT,PARAMS)
#define XGEMMDIRECTTN2 CONCATENATE(XGEMMDIRECTTN,PARAMS)
#define XGEMMDIRECTTT2 CONCATENATE(XGEMMDIRECTTT,PARAMS)
#define XGEMMDIRECT2 CONCATENATE(XGEMMDIRECT,PARAMS)
#define XGEMMDIRECTNN2 CONCATENATE(XGEMMDIRECTNN,PARAMS)
#define XGEMMDIRECTNT2 CONCATENATE(XGEMMDIRECTNT,PARAMS)
#define XGEMMDIRECTTN2 CONCATENATE(XGEMMDIRECTTN,PARAMS)
#define XGEMMDIRECTTT2 CONCATENATE(XGEMMDIRECTTT,PARAMS)
// =================================================================================================
// Main body of the kernel. This is the direct version without pre/post processing and restrictions.
INLINE_FUNC void #define XGEMMDIRECT XGEMMDIRECT##PARAMS
#define XGEMMDIRECT XGEMMDIRECT##PARAMS
XGEMMDIRECT2(const int kSizeM, const int kSizeN, const int kSizeK,
                             const real_arg arg_alpha,
                             const real_arg arg_beta,
                             const __global realMD* restrict agm, const int a_offset, const int a_ld,
                             const __global realND* restrict bgm, const int b_offset, const int b_ld,
                             __global real* cgm, const int c_offset, const int c_ld,
                             LOCAL_PTR real* alm, LOCAL_PTR real* blm,
                             const int a_transpose, const int b_transpose, const int c_transpose,
                             const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);
  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;
  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  real apd[MWID];
  #pragma promote_to_registers
  real bpd[NWID];
  #pragma promote_to_registers
  real cpd[NWID * MWID];
  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWID; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      SetToZero(cpd[_ni * MWID + _mi]);
    }
  }
  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = GET_LOCAL_ID2(0) * MWID + GETGROUPID02() * WGD;
  const int idn = GET_LOCAL_ID2(1) * NWID + GETGROUPID12() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {
    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {
      // Loads data: off-chip --> local (matrix A and B)
      if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
        GLOBALTOLOCALDIRECTA2(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      else {
        GLOBALTOLOCALSCALARA2(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      if (b_ld % VWND == 0 && b_offset % VWND == 0) {
        GLOBALTOLOCALDIRECTB2(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      else {
        GLOBALTOLOCALSCALARB2(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;
          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LOCALTOPRIVATEDIRECTA2(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LOCALTOPRIVATEDIRECTB2(blm, _ni, kg, b_transpose);
          }
          // Performs the accumulation (Cpmd += Apmd * Bpmd)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {
      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GLOBALTOPRIVATEDIRECTA2(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GLOBALTOPRIVATEDIRECTB2(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
      }
      // Performs the accumulation (Cpmd += Apmd * Bpmd)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }
    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        STORERESULTSDIRECT2(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
                           alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }
  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {
    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {
      // Loads data: off-chip --> local (matrix A and B)
      GLOBALTOLOCALCHECKEDA2(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
      GLOBALTOLOCALCHECKEDB2(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
      barrier(CLK_LOCAL_MEM_FENCE);
      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;
          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LOCALTOPRIVATEDIRECTA2(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LOCALTOPRIVATEDIRECTB2(blm, _ni, kg, b_transpose);
          }
          // Performs the accumulation (C += A * B)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {
      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GLOBALTOPRIVATECHECKEDA2(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GLOBALTOPRIVATECHECKEDB2(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
      }
      // Performs the accumulation (C += A * B)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }
    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        STORERESULTSCHECKED2(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,
                            alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }
}
// =================================================================================================
// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTNN XGEMMDIRECTNN##PARAMS
#define XGEMMDIRECTNN XGEMMDIRECTNN##PARAMS
XGEMMDIRECTNN2(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTNT XGEMMDIRECTNT##PARAMS
#define XGEMMDIRECTNT XGEMMDIRECTNT##PARAMS
XGEMMDIRECTNT2(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTTN XGEMMDIRECTTN##PARAMS
#define XGEMMDIRECTTN XGEMMDIRECTTN##PARAMS
XGEMMDIRECTTN2(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}
// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void #define XGEMMDIRECTTT XGEMMDIRECTTT##PARAMS
#define XGEMMDIRECTTT XGEMMDIRECTTT##PARAMS
XGEMMDIRECTTT2(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XGEMMDIRECT2(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================
