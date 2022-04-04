#!/bin/bash

# Plot Figure 1 - Reproduction Lockwood
Rscript plot_lockwood_reproduction.r data/lockwood_reproduction/ lockwood_reproduction pdf

# Plot Figure 2 - Replication Skolik
Rscript plot_skolik_replication.r data/skolik_replication/ skolik_replication pdf

# Plot Figure 3 - Comparison Encoding - Extraction
Rscript plot_encoding_extraction.r data/encoding_extraction/ encoding_extraction pdf

# Plot Figure 4 - Cross-Validation - Baseline - best-performing
Rscript plot_xval_best.r data/xval_baseline/best/ xval_baseline_best pdf

# Cross-Validation - Baseline - all
Rscript plot_xval_all.r data/xval_baseline/lockwood_gs_c/ xval_baseline_lockwood_gs_c pdf
Rscript plot_xval_all.r data/xval_baseline/lockwood_gs_sc/ xval_baseline_lockwood_gs_sc pdf
Rscript plot_xval_all.r data/xval_baseline/lockwood_gsp_c/ xval_baseline_lockwood_gsp_c pdf
Rscript plot_xval_all.r data/xval_baseline/lockwood_gsp_sc/ xval_baseline_lockwood_gsp_sc pdf
Rscript plot_xval_all.r data/xval_baseline/skolik_gs_c/ xval_baseline_skolik_gs_c pdf
Rscript plot_xval_all.r data/xval_baseline/skolik_gs_sc/ xval_baseline_skolik_gs_sc pdf
Rscript plot_xval_all.r data/xval_baseline/skolik_gsp_c/ xval_baseline_skolik_gsp_c pdf
Rscript plot_xval_all.r data/xval_baseline/skolik_gsp_sc/ xval_baseline_skolik_gsp_sc pdf

# Plot Figure 5 - Cross-Validation - Baseline with Data Re-uploading - best-performing
Rscript plot_xval_best.r data/xval_baseline_data_reup/best/ xval_baseline_data_reup_best pdf

# Cross-Validation - Baseline with Data Re-uploading - all
Rscript plot_xval_all.r data/xval_baseline_data_reup/lockwood_gs_c/ xval_baseline_data_reup_lockwood_gs_c pdf
Rscript plot_xval_all.r data/xval_baseline_data_reup/lockwood_gs_sc/ xval_baseline_data_reup_lockwood_gs_sc pdf
Rscript plot_xval_all.r data/xval_baseline_data_reup/lockwood_gsp_c/ xval_baseline_data_reup_lockwood_gsp_c pdf
Rscript plot_xval_all.r data/xval_baseline_data_reup/lockwood_gsp_sc/ xval_baseline_data_reup_lockwood_gsp_sc pdf
Rscript plot_xval_all.r data/xval_baseline_data_reup/skolik_gs_c/ xval_baseline_data_reup_skolik_gs_c pdf
Rscript plot_xval_all.r data/xval_baseline_data_reup/skolik_gs_sc/ xval_baseline_data_reup_skolik_gs_sc pdf
Rscript plot_xval_all.r data/xval_baseline_data_reup/skolik_gsp_c/ xval_baseline_data_reup_skolik_gsp_c pdf
Rscript plot_xval_all.r data/xval_baseline_data_reup/skolik_gsp_sc/ xval_baseline_data_reup_skolik_gsp_sc pdf

# Plot Figure 6 - IBMQ Validation
Rscript plot_ibmq_validation.r data/ibmq_validation.csv ibmq_validation pdf

# Plot Figure 7 - Comarison NN - VQ-DQN
Rscript plot_nn_comparison.r data/nn_comparison/vqdqn/ data/nn_comparison/nn/ nn_comparison pdf

# Plot convergence without early stopping criterion 
Rscript plot_convergence.r data/convergence/ convergence pdf
