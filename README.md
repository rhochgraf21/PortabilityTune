# Tuning Kernels

[Original repo.](https://github.com/jwlawson/tuning_kernels)

Files:
- `data/`: csv formatted data (not our data, can be helpful for debugging -- too big to include here, please see the repo above).
- `matmul_sizes/`: the sizes of matrices in the data collection.
- `compute_tree.py`: generates a decision tree to predict the optimal variant given the input characteristics and device label.
- `dataset.py`: loads data files from csv/pickles.
- `kernel_tree_classifier.py`: compares classifier performance for varying #s of variants.
- `utils.py`: utility functions (PCA calculation, decision tree to human-readable text, geometric mean)

Modifications:
- data loading: loads our gemm kernel data
- decision tree features: uses device label as a decision tree feature