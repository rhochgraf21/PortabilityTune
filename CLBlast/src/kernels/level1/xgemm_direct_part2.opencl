// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define GLOBALTOLOCALDIRECTA2 GLOBALTOLOCALDIRECTA##PARAMS
#define GLOBALTOLOCALDIRECTB2 GLOBALTOLOCALDIRECTB##PARAMS
#define GLOBALTOLOCALSCALARA2 GLOBALTOLOCALSCALARA##PARAMS
#define GLOBALTOLOCALSCALARB2 GLOBALTOLOCALSCALARB##PARAMS
#define GLOBALTOLOCALCHECKEDA2 GLOBALTOLOCALCHECKEDA##PARAMS
#define GLOBALTOLOCALCHECKEDB2 GLOBALTOLOCALCHECKEDB##PARAMS
#define GLOBALTOLOCALDIRECTA22 GLOBALTOLOCALDIRECTA2##PARAMS
#define GLOBALTOLOCALDIRECTB22 GLOBALTOLOCALDIRECTB2##PARAMS
#define GLOBALTOLOCALSCALARA22 GLOBALTOLOCALSCALARA2##PARAMS
#define GLOBALTOLOCALSCALARB22 GLOBALTOLOCALSCALARB2##PARAMS
#define GLOBALTOLOCALCHECKEDA22 GLOBALTOLOCALCHECKEDA2##PARAMS
#define GLOBALTOLOCALCHECKEDB22 GLOBALTOLOCALCHECKEDB2##PARAMS
#define GLOBALTOLOCALDIRECTA GLOBALTOLOCALDIRECTA##PARAMS
#define GLOBALTOLOCALDIRECTB GLOBALTOLOCALDIRECTB##PARAMS
#define GLOBALTOLOCALSCALARA GLOBALTOLOCALSCALARA##PARAMS
#define GLOBALTOLOCALSCALARB GLOBALTOLOCALSCALARB##PARAMS
#define GLOBALTOLOCALCHECKEDA GLOBALTOLOCALCHECKEDA##PARAMS
#define GLOBALTOLOCALCHECKEDB GLOBALTOLOCALCHECKEDB##PARAMS
#define GLOBALTOLOCALDIRECTA GLOBALTOLOCALDIRECTA##PARAMS
#define GLOBALTOLOCALDIRECTB GLOBALTOLOCALDIRECTB##PARAMS
#define GLOBALTOLOCALSCALARA GLOBALTOLOCALSCALARA##PARAMS
#define GLOBALTOLOCALSCALARB GLOBALTOLOCALSCALARB##PARAMS
#define GLOBALTOLOCALCHECKEDA GLOBALTOLOCALCHECKEDA##PARAMS
#define GLOBALTOLOCALCHECKEDB GLOBALTOLOCALCHECKEDB##PARAMS
#define GLOBALTOLOCALCHECKEDB GLOBALTOLOCALCHECKEDB##PARAMS
#define GLOBALTOLOCALCHECKEDA GLOBALTOLOCALCHECKEDA##PARAMS
#define GLOBALTOLOCALSCALARB GLOBALTOLOCALSCALARB##PARAMS
#define GLOBALTOLOCALSCALARA GLOBALTOLOCALSCALARA##PARAMS
#define GLOBALTOLOCALDIRECTB GLOBALTOLOCALDIRECTB##PARAMS
#define GLOBALTOLOCALDIRECTA GLOBALTOLOCALDIRECTA##PARAMS
#define GLOBALTOLOCALDIRECTA2 GLOBALTOLOCALDIRECTA##PARAMS
#define GLOBALTOLOCALDIRECTB2 GLOBALTOLOCALDIRECTB##PARAMS
#define GLOBALTOLOCALSCALARA2 GLOBALTOLOCALSCALARA##PARAMS
#define GLOBALTOLOCALSCALARB2 GLOBALTOLOCALSCALARB##PARAMS
#define GLOBALTOLOCALCHECKEDA2 GLOBALTOLOCALCHECKEDA##PARAMS
#define GLOBALTOLOCALCHECKEDB2 GLOBALTOLOCALCHECKEDB##PARAMS
#define GLOBALTOLOCALDIRECTA2 CONCATENATE(GLOBALTOLOCALDIRECTA,PARAMS)
#define GLOBALTOLOCALDIRECTB2 CONCATENATE(GLOBALTOLOCALDIRECTB,PARAMS)
#define GLOBALTOLOCALSCALARA2 CONCATENATE(GLOBALTOLOCALSCALARA,PARAMS)
#define GLOBALTOLOCALSCALARB2 CONCATENATE(GLOBALTOLOCALSCALARB,PARAMS)
#define GLOBALTOLOCALCHECKEDA2 CONCATENATE(GLOBALTOLOCALCHECKEDA,PARAMS)
#define GLOBALTOLOCALCHECKEDB2 CONCATENATE(GLOBALTOLOCALCHECKEDB,PARAMS)
#define GLOBALTOLOCALDIRECTA2 CONCATENATE(GLOBALTOLOCALDIRECTA,PARAMS)
#define GLOBALTOLOCALDIRECTB2 CONCATENATE(GLOBALTOLOCALDIRECTB,PARAMS)
#define GLOBALTOLOCALSCALARA2 CONCATENATE(GLOBALTOLOCALSCALARA,PARAMS)
#define GLOBALTOLOCALSCALARB2 CONCATENATE(GLOBALTOLOCALSCALARB,PARAMS)
#define GLOBALTOLOCALCHECKEDA2 CONCATENATE(GLOBALTOLOCALCHECKEDA,PARAMS)
#define GLOBALTOLOCALCHECKEDB2 CONCATENATE(GLOBALTOLOCALCHECKEDB,PARAMS)
// =================================================================================================
// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
INLINE_FUNC void #define GLOBALTOLOCALDIRECTA GLOBALTOLOCALDIRECTA##PARAMS
#define GLOBALTOLOCALDIRECTA GLOBALTOLOCALDIRECTA##PARAMS
GLOBALTOLOCALDIRECTA2(const __global realMD* restrict agm, LOCAL_PTR real* alm,
                                      const int a_ld, const int a_offset, const int kwg,
                                      const int a_transpose, const int a_conjugate) {
  #if MDIMCD == MDIMAD
    const int la0 = GET_LOCAL_ID2(0);
    const int la1 = GET_LOCAL_ID2(1);
  #else
    const int tid = GET_LOCAL_ID2(0) + MDIMCD*GET_LOCAL_ID2(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int _mia = 0; _mia < MWAD/VWMD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {
      // Computes the indices for the global memory
      int mg = _mia + la0*(MWAD/VWMD);
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg/VWMD : mg + GETGROUPID02()*(WGD/VWMD);
      int idk = (a_transpose) ? kg + GETGROUPID02()*WGD : kg + kwg;
      // Loads the data from global memory into the local memory
      const realMD avec = agm[idk*(a_ld/VWMD) + idm + (a_offset/VWMD)];
      #if VWMD == 1
         alm[kg*(WGD + PADA) + mg] = avec;
      #elif VWMD == 2
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;
      #elif VWMD == 4
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.z;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.w;
      #elif VWMD == 8
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;
         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;
         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;
         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;
         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;
      #elif VWMD == 16
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;
         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;
         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;
         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;
         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;
         alm[kg*(WGD + PADA) + mg*VWMD + 8] = avec.s8;
         alm[kg*(WGD + PADA) + mg*VWMD + 9] = avec.s9;
         alm[kg*(WGD + PADA) + mg*VWMD + 10] = avec.sA;
         alm[kg*(WGD + PADA) + mg*VWMD + 11] = avec.sB;
         alm[kg*(WGD + PADA) + mg*VWMD + 12] = avec.sC;
         alm[kg*(WGD + PADA) + mg*VWMD + 13] = avec.sD;
         alm[kg*(WGD + PADA) + mg*VWMD + 14] = avec.sE;
         alm[kg*(WGD + PADA) + mg*VWMD + 15] = avec.sF;
      #endif
      if (a_conjugate) {
        for (int vm=0; vm<VWMD; ++vm) {
          COMPLEX_CONJUGATE(alm[kg*(WGD + PADA) + mg*VWMD + vm]);
        }
      }
    }
  }
}
// Same as above, but now for the B input matrix
INLINE_FUNC void #define GLOBALTOLOCALDIRECTB GLOBALTOLOCALDIRECTB##PARAMS
#define GLOBALTOLOCALDIRECTB GLOBALTOLOCALDIRECTB##PARAMS
GLOBALTOLOCALDIRECTB2(const __global realND* restrict bgm, LOCAL_PTR real* blm,
                                      const int b_ld, const int b_offset, const int kwg,
                                      const int b_transpose, const int b_conjugate) {
  #if MDIMCD == NDIMBD
    const int lb0 = GET_LOCAL_ID2(0);
    const int lb1 = GET_LOCAL_ID2(1);
  #else
    const int tid = GET_LOCAL_ID2(0) + MDIMCD*GET_LOCAL_ID2(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD/VWND; _nib += 1) {
      // Computes the indices for the global memory
      int ng = _nib + lb0*(NWBD/VWND);
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg/VWND : ng + GETGROUPID12()*(WGD/VWND);
      int idk = (b_transpose) ? kg + GETGROUPID12()*WGD : kg + kwg;
      // Loads the data from global memory into the local memory
      const realND bvec = bgm[idk*(b_ld/VWND) + idn + (b_offset/VWND)];
      #if VWND == 1
         blm[kg*(WGD + PADB) + ng] = bvec;
      #elif VWND == 2
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;
      #elif VWND == 4
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.z;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.w;
      #elif VWND == 8
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;
         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;
         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;
         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;
         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;
      #elif VWND == 16
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;
         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;
         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;
         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;
         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;
         blm[kg*(WGD + PADB) + ng*VWND + 8] = bvec.s8;
         blm[kg*(WGD + PADB) + ng*VWND + 9] = bvec.s9;
         blm[kg*(WGD + PADB) + ng*VWND + 10] = bvec.sA;
         blm[kg*(WGD + PADB) + ng*VWND + 11] = bvec.sB;
         blm[kg*(WGD + PADB) + ng*VWND + 12] = bvec.sC;
         blm[kg*(WGD + PADB) + ng*VWND + 13] = bvec.sD;
         blm[kg*(WGD + PADB) + ng*VWND + 14] = bvec.sE;
         blm[kg*(WGD + PADB) + ng*VWND + 15] = bvec.sF;
      #endif
      if (b_conjugate) {
        #pragma unroll
        for (int _vn = 0; _vn < VWND; _vn += 1) {
          COMPLEX_CONJUGATE(blm[kg*(WGD + PADB) + ng*VWND + _vn]);
        }
      }
    }
  }
}
// =================================================================================================
// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs doesn't
// use the vector data-types.
INLINE_FUNC void #define GLOBALTOLOCALSCALARA GLOBALTOLOCALSCALARA##PARAMS
#define GLOBALTOLOCALSCALARA GLOBALTOLOCALSCALARA##PARAMS
GLOBALTOLOCALSCALARA2(const __global real* restrict agms, LOCAL_PTR real* alm,
                                      const int a_ld, const int a_offset, const int kwg,
                                      const int a_transpose, const int a_conjugate) {
  #if MDIMCD == MDIMAD
    const int la0 = GET_LOCAL_ID2(0);
    const int la1 = GET_LOCAL_ID2(1);
  #else
    const int tid = GET_LOCAL_ID2(0) + MDIMCD*GET_LOCAL_ID2(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int _mia = 0; _mia < MWAD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {
      // Computes the indices for the global memory
      int mg = _mia + la0*MWAD;
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg : mg + GETGROUPID02()*WGD;
      int idk = (a_transpose) ? kg + GETGROUPID02()*WGD : kg + kwg;
      // Loads the data from global memory into the local memory
      real result = agms[idk*a_ld + idm + a_offset];
      if (a_conjugate) { COMPLEX_CONJUGATE(result); }
      alm[kg*(WGD + PADA) + mg] = result;
    }
  }
}
// Same as above, but now for the B input matrix
INLINE_FUNC void #define GLOBALTOLOCALSCALARB GLOBALTOLOCALSCALARB##PARAMS
#define GLOBALTOLOCALSCALARB GLOBALTOLOCALSCALARB##PARAMS
GLOBALTOLOCALSCALARB2(const __global real* restrict bgms, LOCAL_PTR real* blm,
                                      const int b_ld, const int b_offset, const int kwg,
                                      const int b_transpose, const int b_conjugate) {
  #if MDIMCD == NDIMBD
    const int lb0 = GET_LOCAL_ID2(0);
    const int lb1 = GET_LOCAL_ID2(1);
  #else
    const int tid = GET_LOCAL_ID2(0) + MDIMCD*GET_LOCAL_ID2(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD; _nib += 1) {
      // Computes the indices for the global memory
      int ng = _nib + lb0*NWBD;
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg : ng + GETGROUPID12()*WGD;
      int idk = (b_transpose) ? kg + GETGROUPID12()*WGD : kg + kwg;
      // Loads the data from global memory into the local memory
      real result = bgms[idk*b_ld + idn + b_offset];
      if (b_conjugate) { COMPLEX_CONJUGATE(result); }
      blm[kg*(WGD + PADB) + ng] = result;
    }
  }
}
// =================================================================================================
// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs bounds
// checks and doesn't use the vector data-types.
INLINE_FUNC void #define GLOBALTOLOCALCHECKEDA GLOBALTOLOCALCHECKEDA##PARAMS
#define GLOBALTOLOCALCHECKEDA GLOBALTOLOCALCHECKEDA##PARAMS
GLOBALTOLOCALCHECKEDA2(const __global real* restrict agms, LOCAL_PTR real* alm,
                                       const int a_ld, const int a_offset, const int kwg,
                                       const int a_transpose, const int a_conjugate,
                                       const int kSizeM, const int kSizeK) {
  #if MDIMCD == MDIMAD
    const int la0 = GET_LOCAL_ID2(0);
    const int la1 = GET_LOCAL_ID2(1);
  #else
    const int tid = GET_LOCAL_ID2(0) + MDIMCD*GET_LOCAL_ID2(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int _mia = 0; _mia < MWAD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {
      // Computes the indices for the global memory
      int mg = _mia + la0*MWAD;
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg : mg + GETGROUPID02()*WGD;
      int idk = (a_transpose) ? kg + GETGROUPID02()*WGD : kg + kwg;
      // Loads the data from global memory into the local memory
      int condition = (a_transpose) ? (idm < kSizeK) && (idk < kSizeM) :
                                      (idm < kSizeM) && (idk < kSizeK);
      if (condition) {
        real result = agms[idk*a_ld + idm + a_offset];
        if (a_conjugate) { COMPLEX_CONJUGATE(result); }
        alm[kg*(WGD + PADA) + mg] = result;
      }
      else {
        SetToZero(alm[kg*(WGD + PADA) + mg]);
      }
    }
  }
}
// Same as above, but now for the B input matrix
INLINE_FUNC void #define GLOBALTOLOCALCHECKEDB GLOBALTOLOCALCHECKEDB##PARAMS
#define GLOBALTOLOCALCHECKEDB GLOBALTOLOCALCHECKEDB##PARAMS
GLOBALTOLOCALCHECKEDB2(const __global real* restrict bgms, LOCAL_PTR real* blm,
                                       const int b_ld, const int b_offset, const int kwg,
                                       const int b_transpose, const int b_conjugate,
                                       const int kSizeN, const int kSizeK) {
  #if MDIMCD == NDIMBD
    const int lb0 = GET_LOCAL_ID2(0);
    const int lb1 = GET_LOCAL_ID2(1);
  #else
    const int tid = GET_LOCAL_ID2(0) + MDIMCD*GET_LOCAL_ID2(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD; _nib += 1) {
      // Computes the indices for the global memory
      int ng = _nib + lb0*NWBD;
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg : ng + GETGROUPID12()*WGD;
      int idk = (b_transpose) ? kg + GETGROUPID12()*WGD : kg + kwg;
      // Loads the data from global memory into the local memory
      int condition = (b_transpose) ? (idn < kSizeK) && (idk < kSizeN) :
                                      (idn < kSizeN) && (idk < kSizeK);
      if (condition) {
        real result = bgms[idk*b_ld + idn + b_offset];
        if (b_conjugate) { COMPLEX_CONJUGATE(result); }
        blm[kg*(WGD + PADB) + ng] = result;
      }
      else {
        SetToZero(blm[kg*(WGD + PADB) + ng]);
      }
    }
  }
}
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================
