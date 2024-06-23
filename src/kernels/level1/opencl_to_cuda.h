// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an (incomplete) header to interpret OpenCL kernels as CUDA kernels.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define GET_LOCAL_ID2 GET_LOCAL_ID##PARAMS
#define GET_GROUP_ID2 GET_GROUP_ID##PARAMS
#define GET_LOCAL_SIZE2 GET_LOCAL_SIZE##PARAMS
#define GET_NUM_GROUPS2 GET_NUM_GROUPS##PARAMS
#define GET_GLOBAL_SIZE2 GET_GLOBAL_SIZE##PARAMS
#define GET_GLOBAL_ID2 GET_GLOBAL_ID##PARAMS
#define GET_LOCAL_ID22 GET_LOCAL_ID2##PARAMS
#define GET_GROUP_ID22 GET_GROUP_ID2##PARAMS
#define GET_LOCAL_SIZE22 GET_LOCAL_SIZE2##PARAMS
#define GET_NUM_GROUPS22 GET_NUM_GROUPS2##PARAMS
#define GET_GLOBAL_SIZE22 GET_GLOBAL_SIZE2##PARAMS
#define GET_GLOBAL_ID22 GET_GLOBAL_ID2##PARAMS
#define GET_LOCAL_SIZE2 GET_LOCAL_SIZE##PARAMS
#define GET_LOCAL_ID GET_LOCAL_ID##PARAMS
#define GET_GROUP_ID GET_GROUP_ID##PARAMS
#define GET_LOCAL_SIZE GET_LOCAL_SIZE##PARAMS
#define GET_NUM_GROUPS GET_NUM_GROUPS##PARAMS
#define GET_GLOBAL_SIZE GET_GLOBAL_SIZE##PARAMS
#define GET_GLOBAL_ID GET_GLOBAL_ID##PARAMS
#define GET_LOCAL_ID GET_LOCAL_ID##PARAMS
#define GET_GROUP_ID GET_GROUP_ID##PARAMS
#define GET_LOCAL_SIZE GET_LOCAL_SIZE##PARAMS
#define GET_NUM_GROUPS GET_NUM_GROUPS##PARAMS
#define GET_GLOBAL_SIZE GET_GLOBAL_SIZE##PARAMS
#define GET_GLOBAL_ID GET_GLOBAL_ID##PARAMS
#define GET_GLOBAL_ID GET_GLOBAL_ID##PARAMS
#define GET_GLOBAL_SIZE GET_GLOBAL_SIZE##PARAMS
#define GET_NUM_GROUPS GET_NUM_GROUPS##PARAMS
#define GET_LOCAL_SIZE GET_LOCAL_SIZE##PARAMS
#define GET_GROUP_ID GET_GROUP_ID##PARAMS
#define GET_LOCAL_ID GET_LOCAL_ID##PARAMS
#define GET_LOCAL_ID get_local_id##PARAMS
#define GET_GROUP_ID get_group_id##PARAMS
#define GET_LOCAL_SIZE get_local_size##PARAMS
#define GET_NUM_GROUPS get_num_groups##PARAMS
#define GET_GLOBAL_SIZE get_global_size##PARAMS
#define GET_GLOBAL_ID get_global_id##PARAMS
#define GETGROUPIDFLAT GetGroupIDFlat##PARAMS
#define GETGROUPID1 GetGroupID1##PARAMS
#define GETGROUPID0 GetGroupID0##PARAMS
#define GETGROUPID1 GetGroupID1##PARAMS
#define GETGROUPID0 GetGroupID0##PARAMS
#define XCONVGEMMFLIP XconvgemmFlip##PARAMS
#define XCONVGEMMNORMAL XconvgemmNormal##PARAMS
#define GRID_CEIL grid_ceil##PARAMS
#define XCOL2IM Xcol2im##PARAMS
#define XCOL2IMKERNELFLIP Xcol2imKernelFlip##PARAMS
#define XCOL2IMKERNELNORMAL Xcol2imKernelNormal##PARAMS
#define XIM2COL Xim2col##PARAMS
#define XIM2COLKERNELFLIP Xim2colKernelFlip##PARAMS
#define XIM2COLKERNELNORMAL Xim2colKernelNormal##PARAMS
#define GLOBALTOPRIVATECHECKEDIMAGE GlobalToPrivateCheckedImage##PARAMS
#define GLOBALTOLOCALCHECKEDIMAGE GlobalToLocalCheckedImage##PARAMS
#define XGEMMUPPER XgemmUpper##PARAMS
#define XGEMMLOWER XgemmLower##PARAMS
#define XGEMM Xgemm##PARAMS
#define GLOBALTOPRIVATEDIRECTA GlobalToPrivateDirectA##PARAMS
#define GLOBALTOPRIVATEDIRECTB GlobalToPrivateDirectB##PARAMS
#define GLOBALTOPRIVATECHECKEDA GlobalToPrivateCheckedA##PARAMS
#define GLOBALTOPRIVATECHECKEDB GlobalToPrivateCheckedB##PARAMS
#define LOCALTOPRIVATEDIRECTA LocalToPrivateDirectA##PARAMS
#define LOCALTOPRIVATEDIRECTB LocalToPrivateDirectB##PARAMS
#define STORERESULTSDIRECT StoreResultsDirect##PARAMS
#define STORERESULTSCHECKED StoreResultsChecked##PARAMS
#define SYMMLOWERTOSQUARED SymmLowerToSquared##PARAMS
#define SYMMUPPERTOSQUARED SymmUpperToSquared##PARAMS
#define XGEMMDIRECT XgemmDirect##PARAMS
#define XGEMMDIRECTNN XgemmDirectNN##PARAMS
#define XGEMMDIRECTNT XgemmDirectNT##PARAMS
#define XGEMMDIRECTTN XgemmDirectTN##PARAMS
#define XGEMMDIRECTTT XgemmDirectTT##PARAMS
#define TRANSPOSEMATRIXFAST TransposeMatrixFast##PARAMS
#define MULTIPLYADDVECTOR MultiplyAddVector##PARAMS
#define STORERESULTS StoreResults##PARAMS
#define FILLMATRIX FillMatrix##PARAMS
#define COPYMATRIXFAST CopyMatrixFast##PARAMS
#define HERMLOWERTOSQUARED HermLowerToSquared##PARAMS
#define HERMUPPERTOSQUARED HermUpperToSquared##PARAMS
#define INVERTDIAGONALBLOCK InvertDiagonalBlock##PARAMS
#define TRIPLEMATMUL TripleMatMul##PARAMS
#define TRIPLEMATMULPART1 TripleMatMulPart1##PARAMS
#define TRIPLEMATMULPART2 TripleMatMulPart2##PARAMS
#define _TRANSPOSEPADMATRIX _TransposePadMatrix##PARAMS
#define TRANSPOSEPADMATRIX TransposePadMatrix##PARAMS
#define _TRANSPOSEMATRIX _TransposeMatrix##PARAMS
#define TRANSPOSEMATRIX TransposeMatrix##PARAMS
#define TRANSPOSEPADMATRIXBATCHED TransposePadMatrixBatched##PARAMS
#define TRANSPOSEMATRIXBATCHED TransposeMatrixBatched##PARAMS
#define TRANSPOSEPADMATRIXSTRIDEDBATCHED TransposePadMatrixStridedBatched##PARAMS
#define TRANSPOSEMATRIXSTRIDEDBATCHED TransposeMatrixStridedBatched##PARAMS
#define XGEMMDIRECTBATCHEDNN XgemmDirectBatchedNN##PARAMS
#define XGEMMDIRECTBATCHEDNT XgemmDirectBatchedNT##PARAMS
#define XGEMMDIRECTBATCHEDTN XgemmDirectBatchedTN##PARAMS
#define XGEMMDIRECTBATCHEDTT XgemmDirectBatchedTT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNN XgemmDirectStridedBatchedNN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDNT XgemmDirectStridedBatchedNT##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTN XgemmDirectStridedBatchedTN##PARAMS
#define XGEMMDIRECTSTRIDEDBATCHEDTT XgemmDirectStridedBatchedTT##PARAMS
#define GLOBALTOLOCALDIRECTA GlobalToLocalDirectA##PARAMS
#define GLOBALTOLOCALDIRECTB GlobalToLocalDirectB##PARAMS
#define GLOBALTOLOCALSCALARA GlobalToLocalScalarA##PARAMS
#define GLOBALTOLOCALSCALARB GlobalToLocalScalarB##PARAMS
#define GLOBALTOLOCALCHECKEDA GlobalToLocalCheckedA##PARAMS
#define GLOBALTOLOCALCHECKEDB GlobalToLocalCheckedB##PARAMS
#define TRIALOWERTOSQUARED TriaLowerToSquared##PARAMS
#define TRIAUPPERTOSQUARED TriaUpperToSquared##PARAMS
#define CLBLAST_GET_SUB_GROUP_LOCAL_ID clblast_get_sub_group_local_id##PARAMS
#define CLBLAST_SUB_GROUP_SHUFFLE clblast_sub_group_shuffle##PARAMS
#define XGEMMBODY XgemmBody##PARAMS
#define TRIPLEMATMUL16PART1LOWER TripleMatMul16Part1Lower##PARAMS
#define TRIPLEMATMUL16PART2LOWER TripleMatMul16Part2Lower##PARAMS
#define TRIPLEMATMUL32PART1LOWER TripleMatMul32Part1Lower##PARAMS
#define TRIPLEMATMUL32PART2LOWER TripleMatMul32Part2Lower##PARAMS
#define TRIPLEMATMUL64PART1LOWER TripleMatMul64Part1Lower##PARAMS
#define TRIPLEMATMUL64PART2LOWER TripleMatMul64Part2Lower##PARAMS
#define TRIPLEMATMUL16PART1UPPER TripleMatMul16Part1Upper##PARAMS
#define TRIPLEMATMUL16PART2UPPER TripleMatMul16Part2Upper##PARAMS
#define TRIPLEMATMUL32PART1UPPER TripleMatMul32Part1Upper##PARAMS
#define TRIPLEMATMUL32PART2UPPER TripleMatMul32Part2Upper##PARAMS
#define TRIPLEMATMUL64PART1UPPER TripleMatMul64Part1Upper##PARAMS
#define TRIPLEMATMUL64PART2UPPER TripleMatMul64Part2Upper##PARAMS
#define _COPYPADMATRIX _CopyPadMatrix##PARAMS
#define COPYPADMATRIX CopyPadMatrix##PARAMS
#define _COPYMATRIX _CopyMatrix##PARAMS
#define COPYMATRIX CopyMatrix##PARAMS
#define COPYPADMATRIXBATCHED CopyPadMatrixBatched##PARAMS
#define COPYMATRIXBATCHED CopyMatrixBatched##PARAMS
#define COPYPADMATRIXSTRIDEDBATCHED CopyPadMatrixStridedBatched##PARAMS
#define COPYMATRIXSTRIDEDBATCHED CopyMatrixStridedBatched##PARAMS
#define XGEMMBATCHED XgemmBatched##PARAMS
#define XGEMMSTRIDEDBATCHED XgemmStridedBatched##PARAMS
#define INITACCREGISTERS InitAccRegisters##PARAMS
#define GLOBALTOLOCALA GlobalToLocalA##PARAMS
#define GLOBALTOLOCALB GlobalToLocalB##PARAMS
#define GLOBALTOPRIVATEA GlobalToPrivateA##PARAMS
#define GLOBALTOPRIVATEB GlobalToPrivateB##PARAMS
#define GLOBALTOPRIVATEA2D GlobalToPrivateA2D##PARAMS
#define GLOBALTOPRIVATEB2D GlobalToPrivateB2D##PARAMS
#define LOCALTOPRIVATEA LocalToPrivateA##PARAMS
#define LOCALTOPRIVATEB LocalToPrivateB##PARAMS
#define XHER Xher##PARAMS
#define LOADMATRIXA LoadMatrixA##PARAMS
#define XGEMV Xgemv##PARAMS
#define LOADMATRIXAVF LoadMatrixAVF##PARAMS
#define XGEMVFAST XgemvFast##PARAMS
#define XGEMVFASTROT XgemvFastRot##PARAMS
#define FILLVECTOR FillVector##PARAMS
#define TRSV_FORWARD trsv_forward##PARAMS
#define TRSV_BACKWARD trsv_backward##PARAMS
#define XHER2 Xher2##PARAMS
#define LOADVECTOR LoadVector##PARAMS
#define MATRIXUPDATE MatrixUpdate##PARAMS
#define MATRIXUPDATE2 MatrixUpdate2##PARAMS
#define XGER Xger##PARAMS
#define XAMAX Xamax##PARAMS
#define XAMAXEPILOGUE XamaxEpilogue##PARAMS
#define XNRM2 Xnrm2##PARAMS
#define XNRM2EPILOGUE Xnrm2Epilogue##PARAMS
#define XAXPYFASTER XaxpyFaster##PARAMS
#define XAXPYFASTEST XaxpyFastest##PARAMS
#define XAXPYBATCHED XaxpyBatched##PARAMS
#define MULTIPLYVECTOR MultiplyVector##PARAMS
#define MULTIPLYADDVECTOR MultiplyAddVector##PARAMS
#define XDOT Xdot##PARAMS
#define XDOTEPILOGUE XdotEpilogue##PARAMS
#define XCOPY Xcopy##PARAMS
#define XCOPYFAST XcopyFast##PARAMS
#define XASUM Xasum##PARAMS
#define XASUMEPILOGUE XasumEpilogue##PARAMS
#define XSCAL Xscal##PARAMS
#define XSCALFAST XscalFast##PARAMS
#define MULTIPLYVECTORVECTOR MultiplyVectorVector##PARAMS
#define XHAD Xhad##PARAMS
#define XHADFASTER XhadFaster##PARAMS
#define XHADFASTEST XhadFastest##PARAMS
#define XSWAP Xswap##PARAMS
#define XSWAPFAST XswapFast##PARAMS
#define GET_LOCAL_ID2 GET_LOCAL_ID##PARAMS
#define GET_GROUP_ID2 GET_GROUP_ID##PARAMS
#define GET_LOCAL_SIZE2 GET_LOCAL_SIZE##PARAMS
#define GET_NUM_GROUPS2 GET_NUM_GROUPS##PARAMS
#define GET_GLOBAL_SIZE2 GET_GLOBAL_SIZE##PARAMS
#define GET_GLOBAL_ID2 GET_GLOBAL_ID##PARAMS
#define GET_LOCAL_ID2 CONCATENATE(GET_LOCAL_ID,PARAMS)
#define GET_GROUP_ID2 CONCATENATE(GET_GROUP_ID,PARAMS)
#define GET_LOCAL_SIZE2 CONCATENATE(GET_LOCAL_SIZE,PARAMS)
#define GET_NUM_GROUPS2 CONCATENATE(GET_NUM_GROUPS,PARAMS)
#define GET_GLOBAL_SIZE2 CONCATENATE(GET_GLOBAL_SIZE,PARAMS)
#define GET_GLOBAL_ID2 CONCATENATE(GET_GLOBAL_ID,PARAMS)
#define GET_LOCAL_ID2 CONCATENATE(GET_LOCAL_ID,PARAMS)
#define GET_GROUP_ID2 CONCATENATE(GET_GROUP_ID,PARAMS)
#define GET_LOCAL_SIZE2 CONCATENATE(GET_LOCAL_SIZE,PARAMS)
#define GET_NUM_GROUPS2 CONCATENATE(GET_NUM_GROUPS,PARAMS)
#define GET_GLOBAL_SIZE2 CONCATENATE(GET_GLOBAL_SIZE,PARAMS)
#define GET_GLOBAL_ID2 CONCATENATE(GET_GLOBAL_ID,PARAMS)
// =================================================================================================
// CLBlast specific additions
#define CUDA 1
#define LOCAL_PTR  // pointers to local memory don't have to be annotated in CUDA
// Replaces the OpenCL get_xxx_ID with CUDA equivalents
__device__ int #define GET_LOCAL_ID GET_LOCAL_ID##PARAMS
#define GET_LOCAL_ID GET_LOCAL_ID##PARAMS
GET_LOCAL_ID2(const int x) {
  if (x == 0) { return threadIdx.x; }
  if (x == 1) { return threadIdx.y; }
  return threadIdx.z;
}
__device__ int #define GET_GROUP_ID GET_GROUP_ID##PARAMS
#define GET_GROUP_ID GET_GROUP_ID##PARAMS
GET_GROUP_ID2(const int x) {
  if (x == 0) { return blockIdx.x; }
  if (x == 1) { return blockIdx.y; }
  return blockIdx.z;
}
__device__ int #define GET_LOCAL_SIZE GET_LOCAL_SIZE##PARAMS
#define GET_LOCAL_SIZE GET_LOCAL_SIZE##PARAMS
GET_LOCAL_SIZE2(const int x) {
  if (x == 0) { return blockDim.x; }
  if (x == 1) { return blockDim.y; }
  return blockDim.z;
}
__device__ int #define GET_NUM_GROUPS GET_NUM_GROUPS##PARAMS
#define GET_NUM_GROUPS GET_NUM_GROUPS##PARAMS
GET_NUM_GROUPS2(const int x) {
  if (x == 0) { return gridDim.x; }
  if (x == 1) { return gridDim.y; }
  return gridDim.z;
}
__device__ int #define GET_GLOBAL_SIZE GET_GLOBAL_SIZE##PARAMS
#define GET_GLOBAL_SIZE GET_GLOBAL_SIZE##PARAMS
GET_GLOBAL_SIZE2(const int x) {
  if (x == 0) { return gridDim.x * blockDim.x; }
  if (x == 1) { return gridDim.y * blockDim.y; }
  return gridDim.z * blockDim.z;
}
__device__ int #define GET_GLOBAL_ID GET_GLOBAL_ID##PARAMS
#define GET_GLOBAL_ID GET_GLOBAL_ID##PARAMS
GET_GLOBAL_ID2(const int x) {
  if (x == 0) { return blockIdx.x*blockDim.x + threadIdx.x; }
  if (x == 1) { return blockIdx.y*blockDim.y + threadIdx.y; }
  return blockIdx.z*blockDim.z + threadIdx.z;
}
// Adds the data-types which are not available natively under CUDA
typedef struct { float s0; float s1; float s2; float s3;
                 float s4; float s5; float s6; float s7; } float8;
typedef struct { float s0; float s1; float s2; float s3;
                 float s4; float s5; float s6; float s7;
                 float s8; float s9; float s10; float s11;
                 float s12; float s13; float s14; float s15; } float16;
typedef struct { double s0; double s1; double s2; double s3;
                 double s4; double s5; double s6; double s7; } double8;
typedef struct { double s0; double s1; double s2; double s3;
                 double s4; double s5; double s6; double s7;
                 double s8; double s9; double s10; double s11;
                 double s12; double s13; double s14; double s15; } double16;
// Replaces the OpenCL keywords with CUDA equivalent
#define __kernel __placeholder__
#define __global
#define __placeholder__ extern "C" __global__
#define __local __shared__
#define restrict __restrict__
#define __constant const
#define inline __device__ // assumes all device functions are annotated with inline in OpenCL
// Kernel attributes (don't replace currently)
#define reqd_work_group_size(x, y, z)
// Replaces OpenCL synchronisation with CUDA synchronisation
#define barrier(x) __syncthreads()
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================