#!/bin/bash --login
#SBATCH --job-name=jax-gpu-jet
#SBATCH --account=pawsey1384-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Runs jax-simple on an actual GPU, via Setonix's unsupported ROCm 7.2.4
# module (see jax-test/README.md for why -- the supported module tree
# only goes up to rocm/6.4.1, too old for current jax-rocm wheels).
#
# NB: Setonix auto-assigns 8 CPUs + ~29GB RAM per --gres=gpu:1 pack;
# do NOT add --cpus-per-task or --mem here, it errors (cli_filter).

set -euo pipefail

REPO=/scratch/pawsey1384/lockstoch/periscope-jax
MESHES=/scratch/pawsey1384/lockstoch/jax-test/meshes

# CHECK against `ml avail` under the unsupported tree -- this drifts
ml use /software/setonix/unsupported/
ml rocm/7.2.4

source /scratch/pawsey1384/lockstoch/jax-test/jax-gpu-venv/bin/activate

# sanity check: confirm JAX actually sees the GPU before doing
# anything else -- a silent CPU fallback here would still "work",
# just uselessly slowly, and is a real failure mode we've hit before
python3 -c "import jax; print('JAX devices:', jax.devices())"

export BINDIR="$REPO"
export MSHDIR="$MESHES"
export SOLNFILE="$REPO/out_jet_cvt_7_gpu.nc"
export NUMCPU=8   # matches the 8 CPUs auto-assigned per --gres=gpu:1
export PYTHON=python3

cd "$REPO"
[ -f "$MESHES/jet_cvt_7.nc" ] || bash run/jet/ics_jet_60km.sh
bash run/jet/run_jet_60km_30d.sh

LOGFILE="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
WALLTIME=$(command grep "wall-time" "$LOGFILE" | command grep -oE '[0-9.]+$')
FILEIO=$(command grep "file-i/o" "$LOGFILE" | command grep -oE '[0-9.]+$')

TIMING_CSV="$SLURM_SUBMIT_DIR/timing.csv"
[ -f "$TIMING_CSV" ] || echo "timestamp,run,job_id,wall_time_sec,file_io_sec" > "$TIMING_CSV"
echo "$(date -Iseconds),${SLURM_JOB_NAME},${SLURM_JOB_ID},${WALLTIME},${FILEIO}" >> "$TIMING_CSV"
