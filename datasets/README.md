# Datasets

- `clblast_database_new_pruned.json(.zip)`: Sourced from [the CLBlast database](https://github.com/CNugteren/CLBlast-database/), pruned to only contain newer entries with full results (ie, minimal missing fields relative to current). CLBlast by default collects the minimum runtime from its tuning trials, so that's what these results report. Contains data for ~75 devices, but only a single input per device.

- `database_means.json`: Sourced from a modified version of CLBlast to use the mean runtime instead of the minimum. Data collected from our 5 devices, the mean runtime is reported. Contains up to 64 inputs for each device, with omissions whenever the parameters led to mathematically incorrect results (or other failures). The Intel devices have the most failures for GEMM.

## The Other Dataset

- Data from John Lawson can be found in `tuning_kernels/data/`, but these results are not currently used in the ICS submission revision.