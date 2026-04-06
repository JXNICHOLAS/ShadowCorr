"""
ShadowCorr training and evaluation pipeline.

Sub-modules
-----------
early_stopping – keyboard-triggered graceful stop (backtick key)
losses         – discriminative_loss, graph_based_loss, prototypical_clustering_loss
postprocessing – merge_small_clusters, merge_tiny_clusters
metrics        – compute_clustering_ari, evaluate_segment_clustering
io             – save_cumulative_results
train_one      – train_model_focused, evaluate_model_on_scene_focused (one combination)
sweep          – run_focused_grid_search (sweep over all combinations in train.yaml)
evaluator      – load_model_once, process_single_file, process_batch
"""
