
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the interfaces to common kernels, such as copying, padding, and
// transposing a matrix. These functions are templated and thus header-only. This file also contains
// other common functions to routines, such as a function to launch a kernel.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_COMMON_H_
#define CLBLAST_ROUTINES_COMMON_H_

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "utilities/compile.hpp"
#include "database/database.hpp"

namespace clblast {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(Kernel &kernel, Queue &queue, const Device &device,
               std::vector<size_t> global, const std::vector<size_t> &local,
               EventPointer event, const std::vector<Event> &waitForEvents = {});

// =================================================================================================

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(Queue &queue, const Device &device,
                const std::shared_ptr<Program> program,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t m, const size_t n, const size_t ld, const size_t offset,
                const Buffer<T> &dest, const T constant_value, const size_t local_size);

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(Queue &queue, const Device &device,
                const std::shared_ptr<Program> program,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t n, const size_t inc, const size_t offset,
                const Buffer<T> &dest, const T constant_value, const size_t local_size);

// =================================================================================================

// Copies or transposes a matrix and optionally pads/unpads it with zeros. This method is also able
// to write to symmetric and triangular matrices through optional arguments.
template <typename T>
void PadCopyTransposeMatrix(Queue &queue, const Device &device,
                            const Databases &db_,
                            EventPointer event, const std::vector<Event> &waitForEvents,
                            const size_t src_one, const size_t src_two,
                            const size_t src_ld, const size_t src_offset,
                            const Buffer<T> &src,
                            const size_t dest_one, const size_t dest_two,
                            const size_t dest_ld, const size_t dest_offset,
                            const Buffer<T> &dest,
                            const T alpha,
                            const std::shared_ptr<Program> program, const bool do_pad,
                            const bool do_transpose, const bool do_conjugate,
                            const bool upper = false, const bool lower = false,
                            const bool diagonal_imag_zero = false) {

  // retrieve the input
  auto input = database::Input{src_one, src_two, 0};

  // retrieve the databases with parameters
  Database db_copy = db_("Copy");
  Database db_pad = db_("Pad");
  Database db_transpose = db_("Transpose");
  Database db_padtranspose = db_("Padtranspose");

  std::string db_values;

  // Determines whether or not the fast-version could potentially be used
  auto use_fast_kernel = (src_offset == 0) && (dest_offset == 0) && (do_conjugate == false) &&
                         (src_one == dest_one) && (src_two == dest_two) && (src_ld == dest_ld) &&
                         (upper == false) && (lower == false) && (diagonal_imag_zero == false);

  // Determines the right kernel
  auto kernel_name = std::string{};
  auto pad_kernel = false;
  if (do_transpose) {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db_transpose.GetParameterValue(input, "TRA_WPT") &&
        IsMultiple(src_one, db_transpose.GetParameterValue(input, "TRA_WPT")*db_transpose.GetParameterValue(input, "TRA_DIM")) &&
        IsMultiple(src_two, db_transpose.GetParameterValue(input, "TRA_WPT")*db_transpose.GetParameterValue(input, "TRA_DIM")))) {
      kernel_name = "TransposeMatrixFast";
      db_values = db_transpose.GetValuesStringForInput(input);
    }
    else {
      use_fast_kernel = false;
      pad_kernel = (do_pad || do_conjugate);
      kernel_name = (pad_kernel) ? "TransposePadMatrix" : "TransposeMatrix";
      db_values = (pad_kernel) ? db_padtranspose.GetValuesStringForInput(input): db_padtranspose.GetValuesStringForInput(input);
    }
  }
  else {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db_copy.GetParameterValue(input, "COPY_VW")) &&
        IsMultiple(src_one, db_copy.GetParameterValue(input, "COPY_VW")*db_copy.GetParameterValue(input, "COPY_DIMX")) &&
        IsMultiple(src_two, db_copy.GetParameterValue(input, "COPY_WPT")*db_copy.GetParameterValue(input, "COPY_DIMY"))) {
      kernel_name = "CopyMatrixFast";
      db_values = db_copy.GetValuesStringForInput(input);
    }
    else {
      use_fast_kernel = false;
      pad_kernel = do_pad;
      kernel_name = (pad_kernel) ? "CopyPadMatrix" : "CopyMatrix";
      db_values = db_pad.GetValuesStringForInput(input);
    }
  }
  
  // Retrieves the kernel from the compiled binary
  log_debug("Running kernel " + kernel_name + db_values);
  auto kernel = Kernel(program, kernel_name + db_values);

  // Sets the kernel arguments
  if (use_fast_kernel) {
    kernel.SetArgument(0, static_cast<int>(src_ld));
    kernel.SetArgument(1, src());
    kernel.SetArgument(2, dest());
    kernel.SetArgument(3, GetRealArg(alpha));
  }
  else {
    kernel.SetArgument(0, static_cast<int>(src_one));
    kernel.SetArgument(1, static_cast<int>(src_two));
    kernel.SetArgument(2, static_cast<int>(src_ld));
    kernel.SetArgument(3, static_cast<int>(src_offset));
    kernel.SetArgument(4, src());
    kernel.SetArgument(5, static_cast<int>(dest_one));
    kernel.SetArgument(6, static_cast<int>(dest_two));
    kernel.SetArgument(7, static_cast<int>(dest_ld));
    kernel.SetArgument(8, static_cast<int>(dest_offset));
    kernel.SetArgument(9, dest());
    kernel.SetArgument(10, GetRealArg(alpha));
    if (pad_kernel) {
      kernel.SetArgument(11, static_cast<int>(do_conjugate));
    }
    else {
      kernel.SetArgument(11, static_cast<int>(upper));
      kernel.SetArgument(12, static_cast<int>(lower));
      kernel.SetArgument(13, static_cast<int>(diagonal_imag_zero));
    }
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    if (use_fast_kernel) {
      const auto global = std::vector<size_t>{
        dest_one / db_transpose.GetParameterValue(input, "TRA_WPT"),
        dest_two / db_transpose.GetParameterValue(input, "TRA_WPT")
      };
      const auto local = std::vector<size_t>{db_transpose.GetParameterValue(input, "TRA_DIM"), db_transpose.GetParameterValue(input, "TRA_DIM")};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
    else {
      const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db_padtranspose.GetParameterValue(input, "PADTRA_WPT")), db_padtranspose.GetParameterValue(input, "PADTRA_TILE")),
        Ceil(CeilDiv(dest_two, db_padtranspose.GetParameterValue(input, "PADTRA_WPT")), db_padtranspose.GetParameterValue(input, "PADTRA_TILE"))
      };
      const auto local = std::vector<size_t>{db_padtranspose.GetParameterValue(input, "PADTRA_TILE"), db_padtranspose.GetParameterValue(input, "PADTRA_TILE")};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
  }
  else {
    if (use_fast_kernel) {
      const auto global = std::vector<size_t>{
        dest_one / db_copy.GetParameterValue(input, "COPY_VW"),
        dest_two / db_copy.GetParameterValue(input, "COPY_WPT")
      };
      const auto local = std::vector<size_t>{db_copy.GetParameterValue(input, "COPY_DIMX"), db_copy.GetParameterValue(input, "COPY_DIMY")};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
    else {
      const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db_pad.GetParameterValue(input, "PAD_WPTX")), db_pad.GetParameterValue(input, "PAD_DIMX")),
        Ceil(CeilDiv(dest_two, db_pad.GetParameterValue(input, "PAD_WPTY")), db_pad.GetParameterValue(input, "PAD_DIMY"))
      };
      const auto local = std::vector<size_t>{db_pad.GetParameterValue(input, "PAD_DIMX"), db_pad.GetParameterValue(input, "PAD_DIMY")};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
  }
}

// Batched version of the above
template <typename T>
void PadCopyTransposeMatrixBatched(Queue &queue, const Device &device,
                                   const Databases &db_,
                                   EventPointer event, const std::vector<Event> &waitForEvents,
                                   const size_t src_one, const size_t src_two,
                                   const size_t src_ld, const Buffer<int> &src_offsets,
                                   const Buffer<T> &src,
                                   const size_t dest_one, const size_t dest_two,
                                   const size_t dest_ld, const Buffer<int> &dest_offsets,
                                   const Buffer<T> &dest,
                                   const std::shared_ptr<Program> program, const bool do_pad,
                                   const bool do_transpose, const bool do_conjugate,
                                   const size_t batch_count) {

  auto input = database::Input{src_one, src_two, 0};

  // retrieve the db
  Database db_copy = db_("Copy");
  Database db_pad = db_("Pad");
  Database db_transpose = db_("Transpose");
  Database db_padtranspose = db_("Padtranspose");

  Database db;

  // Determines the right kernel
  auto kernel_name = std::string{};
  if (do_transpose) {
    kernel_name = (do_pad) ? "TransposePadMatrixBatched" : "TransposeMatrixBatched";
    db = (do_pad) ? db_padtranspose : db_transpose;
  }
  else {
    kernel_name = (do_pad) ? "CopyPadMatrixBatched" : "CopyMatrixBatched";
    db = (do_pad) ? db_pad : db_copy;
  }

  // Retrieves the kernel from the compiled binary
  auto values = db.GetValuesStringForInput(input);
  log_debug("Running kernel " + kernel_name + values);
  auto kernel = Kernel(program, kernel_name + values);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(src_one));
  kernel.SetArgument(1, static_cast<int>(src_two));
  kernel.SetArgument(2, static_cast<int>(src_ld));
  kernel.SetArgument(3, src_offsets());
  kernel.SetArgument(4, src());
  kernel.SetArgument(5, static_cast<int>(dest_one));
  kernel.SetArgument(6, static_cast<int>(dest_two));
  kernel.SetArgument(7, static_cast<int>(dest_ld));
  kernel.SetArgument(8, dest_offsets());
  kernel.SetArgument(9, dest());
  if (do_pad) {
    kernel.SetArgument(10, static_cast<int>(do_conjugate));
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    const auto global = std::vector<size_t>{
      Ceil(CeilDiv(dest_one, db.GetParameterValue(input, "PADTRA_WPT")), db.GetParameterValue(input, "PADTRA_TILE")),
      Ceil(CeilDiv(dest_two, db.GetParameterValue(input, "PADTRA_WPT")), db.GetParameterValue(input, "PADTRA_TILE")),
      batch_count
    };
    const auto local = std::vector<size_t>{db.GetParameterValue(input, "PADTRA_TILE"), db.GetParameterValue(input, "PADTRA_TILE"), 1};
    RunKernel(kernel, queue, device, global, local, event, waitForEvents);
  }
  else {
    const auto global = std::vector<size_t>{
      Ceil(CeilDiv(dest_one, db.GetParameterValue(input, "PAD_WPTX")), db.GetParameterValue(input, "PAD_DIMX")),
      Ceil(CeilDiv(dest_two, db.GetParameterValue(input, "PAD_WPTY")), db.GetParameterValue(input, "PAD_DIMY")),
      batch_count
    };
    const auto local = std::vector<size_t>{db.GetParameterValue(input, "PAD_DIMX"), db.GetParameterValue(input, "PAD_DIMY"), 1};
    RunKernel(kernel, queue, device, global, local, event, waitForEvents);
  }
}

// Batched version of the above
template <typename T>
void PadCopyTransposeMatrixStridedBatched(Queue &queue, const Device &device,
                                          const Databases &db_,
                                          EventPointer event, const std::vector<Event> &waitForEvents,
                                          const size_t src_one, const size_t src_two,
                                          const size_t src_ld, const size_t src_offset,
                                          const size_t src_stride, const Buffer<T> &src,
                                          const size_t dest_one, const size_t dest_two,
                                          const size_t dest_ld, const size_t dest_offset,
                                          const size_t dest_stride, const Buffer<T> &dest,
                                          const std::shared_ptr<Program> program, const bool do_pad,
                                          const bool do_transpose, const bool do_conjugate,
                                          const size_t batch_count) {


  // retrieve the databases with parameters
  Database db_copy = db_("Copy");
  Database db_pad = db_("Pad");
  Database db_transpose = db_("Transpose");
  Database db_padtranspose = db_("Padtranspose");

  Database db;

  // retrieve the input
  auto input = database::Input{src_one, src_two, 0};

  // Determines the right kernel
  auto kernel_name = std::string{};
  if (do_transpose) {
    kernel_name = (do_pad) ? "TransposePadMatrixStridedBatched" : "TransposeMatrixStridedBatched";
    db = (do_pad) ? db_padtranspose : db_transpose;
  }
  else {
    kernel_name = (do_pad) ? "CopyPadMatrixStridedBatched" : "CopyMatrixStridedBatched";
    db = (do_pad) ? db_pad : db_copy;
  }

  auto values = db.GetValuesStringForInput(input);

  // Retrieves the kernel from the compiled binary
  auto kernel = Kernel(program, kernel_name + values);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(src_one));
  kernel.SetArgument(1, static_cast<int>(src_two));
  kernel.SetArgument(2, static_cast<int>(src_ld));
  kernel.SetArgument(3, static_cast<int>(src_offset));
  kernel.SetArgument(4, static_cast<int>(src_stride));
  kernel.SetArgument(5, src());
  kernel.SetArgument(6, static_cast<int>(dest_one));
  kernel.SetArgument(7, static_cast<int>(dest_two));
  kernel.SetArgument(8, static_cast<int>(dest_ld));
  kernel.SetArgument(9, static_cast<int>(dest_offset));
  kernel.SetArgument(10, static_cast<int>(dest_stride));
  kernel.SetArgument(11, dest());
  if (do_pad) {
    kernel.SetArgument(12, static_cast<int>(do_conjugate));
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db.GetParameterValue(input, "PADTRA_WPT")), db.GetParameterValue(input, "PADTRA_TILE")),
        Ceil(CeilDiv(dest_two, db.GetParameterValue(input, "PADTRA_WPT")), db.GetParameterValue(input, "PADTRA_TILE")),
        batch_count
    };
    const auto local = std::vector<size_t>{db.GetParameterValue(input, "PADTRA_TILE"), db.GetParameterValue(input, "PADTRA_TILE"), 1};
    RunKernel(kernel, queue, device, global, local, event, waitForEvents);
  }
  else {
    const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db.GetParameterValue(input, "PAD_WPTX")), db.GetParameterValue(input, "PAD_DIMX")),
        Ceil(CeilDiv(dest_two, db.GetParameterValue(input, "PAD_WPTY")), db.GetParameterValue(input, "PAD_DIMY")),
        batch_count
    };
    const auto local = std::vector<size_t>{db.GetParameterValue(input, "PAD_DIMX"), db.GetParameterValue(input, "PAD_DIMY"), 1};
    RunKernel(kernel, queue, device, global, local, event, waitForEvents);
  }
}

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_COMMON_H_
#endif
