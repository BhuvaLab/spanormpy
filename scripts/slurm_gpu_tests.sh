#!/bin/bash
#SBATCH --job-name=spanormpy_gpu_tests
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
# Run the full test suite (including [cuda]-parametrised tests) on one GPU.
# Submit from the repo root (first run: mkdir -p /scratch/user/$USER/logs): sbatch scripts/slurm_gpu_tests.sh
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
cd "${SLURM_SUBMIT_DIR}"
PY=${PY:-/scratch/user/uqdbhuva/envs/spanormpy/bin/python}
echo "job ${SLURM_JOB_ID} on $(hostname) | ${SLURM_CPUS_PER_TASK} cpus | $(date -Is)"
nvidia-smi -L
$PY -c "import torch;print(torch.__version__, torch.cuda.is_available())"
$PY -m pytest -q -m "gpu or not gpu" -p no:cacheprovider "$@"
