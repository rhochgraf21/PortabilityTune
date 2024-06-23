
# CLBlast v2

- This version of CLBlast has been modified to store kernels prior to compilation.
- A relevant location of code is `src/routine.cpp`, especially function `Routine::InitProgram`.
  - An example of the generated OpenCL code is in `build/build.txt` and shows how multiple kernels are compiled at the same time. The basic idea is that a separate function is made in the binary for each version of the parameters (ie each variant), using the preprocessor.
  - Each variant can be accessed through a unique entrypoint that is made of a kernel name and the compiled parameters.
- This current version passes the tests (`cmake -DTESTS=ON ..`, `make clblast_test_xgemm`) for SGEMM only. However, *all* kernel definitions in `src/kernels/` have been preprocessed.
- The current version may only build the `xgemm` tests and binaries since the main APIs for `OverrideParams` and similar are broken.
- I've done my best to remove any code specific to the device this was initially developed on, but there might be some lingering around (you have been warned).

See also: CLBlast readme