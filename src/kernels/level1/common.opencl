// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common defines and type-defs for the CLBlast OpenCL kernels.
//
// =================================================================================================
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define GETGROUPIDFLAT2 GETGROUPIDFLAT##PARAMS
#define GETGROUPID12 GETGROUPID1##PARAMS
#define GETGROUPID02 GETGROUPID0##PARAMS
#define GETGROUPIDFLAT22 GETGROUPIDFLAT2##PARAMS
#define GETGROUPID122 GETGROUPID12##PARAMS
#define GETGROUPID022 GETGROUPID02##PARAMS
#define GETGROUPIDFLAT2 GETGROUPIDFLAT##PARAMS
#define GETGROUPIDFLAT GETGROUPIDFLAT##PARAMS
#define GETGROUPID1 GETGROUPID1##PARAMS
#define GETGROUPID0 GETGROUPID0##PARAMS
#define GETGROUPIDFLAT GETGROUPIDFLAT##PARAMS
#define GETGROUPID1 GETGROUPID1##PARAMS
#define GETGROUPID0 GETGROUPID0##PARAMS
#define GETGROUPID0 GETGROUPID0##PARAMS
#define GETGROUPID1 GETGROUPID1##PARAMS
#define GETGROUPIDFLAT GETGROUPIDFLAT##PARAMS
#define GETGROUPIDFLAT2 GETGROUPIDFLAT##PARAMS
#define GETGROUPID12 GETGROUPID1##PARAMS
#define GETGROUPID02 GETGROUPID0##PARAMS
#define GETGROUPIDFLAT2 CONCATENATE(GETGROUPIDFLAT,PARAMS)
#define GETGROUPID12 CONCATENATE(GETGROUPID1,PARAMS)
#define GETGROUPID02 CONCATENATE(GETGROUPID0,PARAMS)
#define GETGROUPIDFLAT2 CONCATENATE(GETGROUPIDFLAT,PARAMS)
#define GETGROUPID12 CONCATENATE(GETGROUPID1,PARAMS)
#define GETGROUPID02 CONCATENATE(GETGROUPID0,PARAMS)
// =================================================================================================
// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this file is used outside of the CLBlast library.
#ifndef PRECISION
  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular
#endif
// =================================================================================================
#ifndef CUDA
  // Enable support for half-precision
  #if PRECISION == 16
    #pragma OPENCL EXTENSION cl_khr_fp16: enable
  #endif
  // Enable support for double-precision
  #if PRECISION == 64 || PRECISION == 6464
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #endif
#endif
// Half-precision
#if PRECISION == 16
	#define real half
	#define real2 half2
	#define real4 half4
	#define real8 half8
	#define real16 half16
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14
// Single-precision
#elif PRECISION == 32
	#define real float
	#define real2 float2
	#define real4 float4
	#define real8 float8
	#define real16 float16
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f
// Double-precision 
#elif PRECISION == 64
	#define real double
	#define real2 double2
	#define real4 double4
	#define real8 double8
	#define real16 double16
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37
// Complex single-precision
#elif PRECISION == 3232
	#define real float2
	#define cfloat2 struct
	#define cfloat4 struct
	#define cfloat8 struct
                          real s4; real s5; real s6; real s7;} real8;
	#define cfloat16 struct
                           real s4; real s5; real s6; real s7;
                           real s8; real s9; real sA; real sB;
                           real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f
// Complex double-precision
#elif PRECISION == 6464
	#define real double2
	#define cdouble2 struct
	#define cdouble4 struct
	#define cdouble8 struct
                           real s4; real s5; real s6; real s7;} real8;
	#define cdouble16 struct
                            real s4; real s5; real s6; real s7;
                            real s8; real s9; real sA; real sB;
                            real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37
#endif
// Single-element version of a complex number
#if PRECISION == 3232
	#define singlereal float
#elif PRECISION == 6464
	#define singlereal double
#else
	#define singlereal real
#endif
// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
	#define real_arg float
  #define GetRealArg(x) (half)x
#else
	#define real_arg real
  #define GetRealArg(x) x
#endif
// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
  #define LOCAL_PTR __local
#endif
// =================================================================================================
// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cpp).
#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif
// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size 
// requirement results in worse performance and is disabled (src/utilities/compile.cpp)
#ifndef RELAX_WORKGROUP_SIZE
  #define RELAX_WORKGROUP_SIZE 0
#endif
// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
  #define SetToZero(a) a = ZERO
#endif
// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
  #define ImagToZero(a) a.y = ZERO
#else
  #define ImagToZero(a) 
#endif
// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToOne(a) a.x = ONE; a.y = ZERO
#else
  #define SetToOne(a) a = ONE
#endif
// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
  #define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))
#else
  #define IsZero(a) (a == ZERO)
#endif
// The absolute value (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define AbsoluteValue(value) value.x = fabs(value.x); value.y = fabs(value.y)
#else
  #define AbsoluteValue(value) value = fabs(value)
#endif
// Negation (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define Negate(value) value.x = -(value.x); value.y = -(value.y)
#else
  #define Negate(value) value = -(value)
#endif
// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#else
  #define Add(c,a,b) c = a + b
#endif
// Subtracts two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Subtract(c,a,b) c.x = a.x - b.x; c.y = a.y - b.y
#else
  #define Subtract(c,a,b) c = a - b
#endif
// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
  #define MulReal(a,b) a.x*b.x - a.y*b.y
  #define MulImag(a,b) a.x*b.y + a.y*b.x
#endif
// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
  #define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
  #define Multiply(c,a,b) c = a * b
#endif
// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplyAdd(c,a,b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
  #if USE_CL_MAD == 1
    #define MultiplyAdd(c,a,b) c = mad(a, b, c)
  #else
    #define MultiplyAdd(c,a,b) c += a * b
  #endif
#endif
// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplySubtract(c,a,b) c.x -= MulReal(a,b); c.y -= MulImag(a,b)
#else
  #define MultiplySubtract(c,a,b) c -= a * b
#endif
// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
  #define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c.x = num_x / denom; c.y = num_y / denom
#else
  #define DivideFull(c,a,b) c = a / b
#endif
// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
  #define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
  #define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif
// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
  #define COMPLEX_CONJUGATE(value) 
#endif
// =================================================================================================
// Force inlining functions or not: some compilers don't support the inline keyword
#ifdef USE_INLINE_KEYWORD
  #define INLINE_FUNC inline
#else
  #define INLINE_FUNC
#endif
// =================================================================================================
// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
  #define USE_STAGGERED_INDICES 0
#endif
// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1 && GEMMK == 0
  INLINE_FUNC int #define GETGROUPIDFLAT GETGROUPIDFLAT##PARAMS
#define GETGROUPIDFLAT GETGROUPIDFLAT##PARAMS
GETGROUPIDFLAT2() {
    return GET_GROUP_ID2(0) + GET_NUM_GROUPS2(0) * GET_GROUP_ID2(1);
  }
  INLINE_FUNC int #define GETGROUPID1 GETGROUPID1##PARAMS
#define GETGROUPID1 GETGROUPID1##PARAMS
GETGROUPID12() {
    return (GETGROUPIDFLAT2()) % GET_NUM_GROUPS2(1);
  }
  INLINE_FUNC int #define GETGROUPID0 GETGROUPID0##PARAMS
#define GETGROUPID0 GETGROUPID0##PARAMS
GETGROUPID02() {
    return ((GETGROUPIDFLAT2() / GET_NUM_GROUPS2(1)) + GETGROUPID12()) % GET_NUM_GROUPS2(0);
  }
#else
  INLINE_FUNC int GETGROUPID12() { return GET_GROUP_ID2(1); }
  INLINE_FUNC int GETGROUPID02() { return GET_GROUP_ID2(0); }
#endif
// =================================================================================================
// End of the C++11 raw string literal
)"
// =================================================================================================
