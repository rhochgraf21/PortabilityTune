
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 2, see part 1 of the invert kernel for a description
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(#define TRIPLEMATMUL16PART1UPPER2 TripleMatMul16Part1Upper
#define TRIPLEMATMUL16PART1UPPER CONCATENATE(TRIPLEMATMUL16PART1UPPER2, PARAMS)
#define TRIPLEMATMUL32PART2UPPER2 TripleMatMul32Part2Upper
#define TRIPLEMATMUL32PART2UPPER CONCATENATE(TRIPLEMATMUL32PART2UPPER2, PARAMS)
#define TRIPLEMATMUL16PART2LOWER2 TripleMatMul16Part2Lower
#define TRIPLEMATMUL16PART2LOWER CONCATENATE(TRIPLEMATMUL16PART2LOWER2, PARAMS)
#define TRIPLEMATMUL64PART2LOWER2 TripleMatMul64Part2Lower
#define TRIPLEMATMUL64PART2LOWER CONCATENATE(TRIPLEMATMUL64PART2LOWER2, PARAMS)
#define TRIPLEMATMUL32PART2LOWER2 TripleMatMul32Part2Lower
#define TRIPLEMATMUL32PART2LOWER CONCATENATE(TRIPLEMATMUL32PART2LOWER2, PARAMS)
#define TRIPLEMATMUL32PART1LOWER2 TripleMatMul32Part1Lower
#define TRIPLEMATMUL32PART1LOWER CONCATENATE(TRIPLEMATMUL32PART1LOWER2, PARAMS)
#define TRIPLEMATMUL16PART1LOWER2 TripleMatMul16Part1Lower
#define TRIPLEMATMUL16PART1LOWER CONCATENATE(TRIPLEMATMUL16PART1LOWER2, PARAMS)
#define TRIPLEMATMUL32PART1UPPER2 TripleMatMul32Part1Upper
#define TRIPLEMATMUL32PART1UPPER CONCATENATE(TRIPLEMATMUL32PART1UPPER2, PARAMS)
#define TRIPLEMATMUL64PART1UPPER2 TripleMatMul64Part1Upper
#define TRIPLEMATMUL64PART1UPPER CONCATENATE(TRIPLEMATMUL64PART1UPPER2, PARAMS)
#define TRIPLEMATMUL64PART2UPPER2 TripleMatMul64Part2Upper
#define TRIPLEMATMUL64PART2UPPER CONCATENATE(TRIPLEMATMUL64PART2UPPER2, PARAMS)
#define TRIPLEMATMUL64PART1LOWER2 TripleMatMul64Part1Lower
#define TRIPLEMATMUL64PART1LOWER CONCATENATE(TRIPLEMATMUL64PART1LOWER2, PARAMS)
#define TRIPLEMATMUL16PART2UPPER2 TripleMatMul16Part2Upper
#define TRIPLEMATMUL16PART2UPPER CONCATENATE(TRIPLEMATMUL16PART2UPPER2, PARAMS)


// =================================================================================================
#if defined(ROUTINE_INVERT)

// B21 = A21 * B11
__kernel
void TRIPLEMATMUL16PART1LOWER(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART1(16, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel
void TRIPLEMATMUL16PART2LOWER(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART2(16, false, lm, n, dest, current_size, num_pages, block_size);
}

// B21 = A21 * B11
__kernel
void TRIPLEMATMUL32PART1LOWER(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART1(32, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel
void TRIPLEMATMUL32PART2LOWER(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART2(32, false, lm, n, dest, current_size, num_pages, block_size);
}

// B21 = A21 * B11
__kernel
void TRIPLEMATMUL64PART1LOWER(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART1(64, false, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B21 = -B22 * B21
__kernel
void TRIPLEMATMUL64PART2LOWER(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART2(64, false, lm, n, dest, current_size, num_pages, block_size);
}

// =================================================================================================

// B12 =  A12 * B22
__kernel
void TRIPLEMATMUL16PART1UPPER(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART1(16, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel
void TRIPLEMATMUL16PART2UPPER(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART2(16, true, lm, n, dest, current_size, num_pages, block_size);
}

// B12 =  A12 * B22
__kernel
void TRIPLEMATMUL32PART1UPPER(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART1(32, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel
void TRIPLEMATMUL32PART2UPPER(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART2(32, true, lm, n, dest, current_size, num_pages, block_size);
}

// B12 =  A12 * B22
__kernel
void TRIPLEMATMUL64PART1UPPER(int n, __global const real* restrict src, const int a_offset, const int lda,
                              __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART1(64, true, lm, n, src, a_offset, lda, dest, current_size, num_pages, block_size);
}

// B12 = -B11 * B12
__kernel
void TRIPLEMATMUL64PART2UPPER(int n, __global real* restrict dest, int current_size, int num_pages, const int block_size)
{
  __local real lm[LOCALY * LOCALX];
  TRIPLEMATMULPART2(64, true, lm, n, dest, current_size, num_pages, block_size);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
