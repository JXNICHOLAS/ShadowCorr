"""
ShadowCorr model components.

Sub-modules
-----------
encoder   – SegmentIDEncoder (word-embedding style segment representations)
network   – RockInstanceNetSparse, MultiHeadLocalAttention (sparse voxel CNN)
features  – heatmap_to_sparse_tensor_with_geometry (voxel feature builder)
data      – RockVoxelDatasetPrecomputed, load_data_from_folder (NPZ loaders)

Import directly from the sub-modules above as needed.
"""
