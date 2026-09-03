#!/bin/bash --login
#SBATCH --job-name=simple-jet
#SBATCH --account=pawsey1384
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO=/scratch/pawsey1384/lockstoch/periscope-simple
MESHES=/scratch/pawsey1384/lockstoch/jax-test/meshes

module load python/3.11.6
source /scratch/pawsey1384/lockstoch/jax-test/simple-venv/bin/activate

export BINDIR="$REPO"
export MSHDIR="$MESHES"
export SOLNFILE="$REPO/out_jet_cvt_7.nc"
export NUMCPU=$SLURM_CPUS_PER_TASK
export PYTHON=python3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPO"
[ -f "$MESHES/jet_cvt_7.nc" ] || bash run/jet/ics_jet_60km.sh
bash run/jet/run_jet_60km_30d.sh

# record wall-time/file-i/o from swe.py's own printed summary, so CPU
# and GPU runs end up in one shared, comparable log
LOGFILE="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
WALLTIME=$(command grep "wall-time" "$LOGFILE" | command grep -oE '[0-9.]+$')
FILEIO=$(command grep "file-i/o" "$LOGFILE" | command grep -oE '[0-9.]+$')

TIMING_CSV="$SLURM_SUBMIT_DIR/timing.csv"
[ -f "$TIMING_CSV" ] || echo "timestamp,run,job_id,wall_time_sec,file_io_sec" > "$TIMING_CSV"
echo "$(date -Iseconds),${SLURM_JOB_NAME},${SLURM_JOB_ID},${WALLTIME},${FILEIO}" >> "$TIMING_CSV"
