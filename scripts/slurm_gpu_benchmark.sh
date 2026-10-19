#!/bin/bash
#SBATCH --job-name=spanormpy_gpu_bench
#SBATCH --account=a_frazer
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/%u/logs/%x_%A.out
#SBATCH --error=/scratch/user/%u/logs/%x_%A.err
# Submit from the repo root: sbatch scripts/slurm_gpu_benchmark.sh [benchmark_gpu.py args]
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
cd "${SLURM_SUBMIT_DIR}"
PY=${PY:-$HOME/miniconda3/envs/spanormpy/bin/python}
nvidia-smi -L
$PY scripts/benchmark_gpu.py --out "/scratch/user/${USER}/logs/bench_${SLURM_JOB_ID}.json" "$@"
