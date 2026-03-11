#!/bin/bash

# PROJECT-SPECIFIC RUN for PREDICTIVE MODELING with an ELASTIC NET
# This file can (and should) be copied to your project directory (along with config.yaml template)
# Edit these files where indicated

# Set project dirs, paths, and server parameters
config_path="/absolute/path/to/project/config.yaml"
slurm_log_path="/absolute/path/for/slurm/outputs"
mem_per_node_GB=120
cpus_per_task=30 # should match "n_cores" in config.yaml file
n_jobs=100 # should be able to divide this number by stats_params.n_permutations in config file cleaningly

# ---------------------------
# DO NOT EDIT BELOW HERE
# ----------------------------
sh /home/tjk33/project/fmri-elastic-net/run_fmri-elastic-net.sh \
    ${config_path} ${slurm_log_path} ${mem_per_node_GB} ${cpus_per_task} ${n_jobs}