#!/bin/bash

# =================================================================
# MASTER SLURM ORCHESTRATOR FOR ELASTIC NET PIPELINE
# Assumptions:
# 1. ALWAYS uses psych_week unless PARTITION is changed below
# 2. Read access to /home/tjk33/project directory
# 3. Access to fmri-elastic-net conda environment (create with environment.yaml)

# Last updated: 01/30/2026
# =================================================================

# ARGUMENT PARSING
if [ "$#" -ne 5 ]; then
    echo "Usage: sh run_fmri-elastic-net.sh <CONFIG_PATH> <LOG_DIR> <MEM_GB> <CPUS_PER_TASK> <N_JOBS>"
    exit 1
fi

CONFIG_FILE="$1"
LOG_DIR="$2"
MEM_INT="$3"
CPUS_PER_TASK="$4"
N_JOBS="$5"

# SETUP VARIABLES
PYTHON_SCRIPT="/home/tjk33/project/fmri-elastic-net/fmri-elastic-net.py"
MEM_PER_NODE="${MEM_INT}G"
PARTITION="psych_week"
TIME_LIMIT="07-00:00:00"

# ENVIRONMENT SETUP
ENV_SETUP="module load StdEnv && module load miniconda && conda activate fmri-elastic-net"
mkdir -p "$LOG_DIR"

echo "-----------------------------------------------------------"
echo "PIPELINE CONFIGURATION"
echo "Config File:    $CONFIG_FILE"
echo "Log Directory:  $LOG_DIR"
echo "Partition:      $PARTITION"
echo "Memory:         $MEM_PER_NODE"
echo "Job Array Size: $N_JOBS"
echo "Environment:    fmri-elastic-net"
echo "-----------------------------------------------------------"

# =================================================================
# STEP 1: MAIN ANALYSIS
# (Bootstrap, Block Perms, Clustering - Skips Local Permutation)
# =================================================================

JOB_ID_MAIN=$(sbatch --parsable \
    --job-name="EN_Main" \
    --partition=$PARTITION \
    --output="${LOG_DIR}/main_%j.log" \
    --error="${LOG_DIR}/main_%j.err" \
    --cpus-per-task=$CPUS_PER_TASK \
    --mem=$MEM_PER_NODE \
    --time=$TIME_LIMIT \
    --wrap="${ENV_SETUP}; python3 $PYTHON_SCRIPT --config $CONFIG_FILE --mode main --skip_main_perm")

echo "Submitted MAIN Job: $JOB_ID_MAIN"

# =================================================================
# STEP 2: PERMUTATION WORKERS (Job Array)
# =================================================================

JOB_ID_WORKERS=$(sbatch --parsable \
    --array=0-$((N_JOBS-1)) \
    --job-name="EN_Worker" \
    --partition=$PARTITION \
    --output="${LOG_DIR}/worker_%A_%a.log" \
    --error="${LOG_DIR}/worker_%A_%a.err" \
    --cpus-per-task=$CPUS_PER_TASK \
    --mem=$MEM_PER_NODE \
    --time=$TIME_LIMIT \
    --wrap="${ENV_SETUP}; python3 $PYTHON_SCRIPT --config $CONFIG_FILE --mode perm_worker --n_jobs $N_JOBS --job_id \$SLURM_ARRAY_TASK_ID")

echo "Submitted WORKER Array: $JOB_ID_WORKERS (Indices 0-$((N_JOBS-1)))"

# =================================================================
# STEP 3: AGGREGATION
# (Waits for Main and Workers to finish)
# =================================================================

JOB_ID_AGG=$(sbatch --parsable \
    --dependency=afterok:${JOB_ID_MAIN}:${JOB_ID_WORKERS} \
    --job-name="EN_Agg" \
    --partition=$PARTITION \
    --output="${LOG_DIR}/agg_%j.log" \
    --error="${LOG_DIR}/agg_%j.err" \
    --cpus-per-task=$CPUS_PER_TASK \
    --mem=$MEM_PER_NODE \
    --time="01:00:00" \
    --wrap="${ENV_SETUP}; python3 $PYTHON_SCRIPT --config $CONFIG_FILE --mode aggregate")

echo "Submitted AGGREGATE Job: $JOB_ID_AGG"
echo "-----------------------------------------------------------"
echo "Status command: squeue -u $USER"