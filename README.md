# jax-test

Three experiments, all running the Galewsky jet (60km mesh, 30 days,
1-day save/stat cadence) and all comparable against each other since
they're the same physics, same formulas, only the execution engine
and hardware differ:

- **`simple`** -- NumPy, CPU. Pre-JAX-port reference.
- **`jax-simple`, CPU** -- the JAX port, running on ordinary processors.
- **`jax-simple`, GPU** -- the same JAX port, running on an actual GPU
  via Setonix's *unsupported* ROCm 7.2.4 module (see below for why).

If `simple` and `jax-simple` (either engine) disagree beyond
float32-rounding levels, that's a bug in the port. If CPU and GPU
`jax-simple` disagree at all, that's a bug too, it's the exact same
compiled JAX code either way, just dispatched to different hardware.

## Why GPU needs the unsupported module tree

Setonix's *supported* module tree only has ROCm up to `6.4.1`, but the
actively-maintained ROCm build of JAX needs ROCm 7.x, current releases
don't target 6.x at all. `rocm/7.2.4` exists, but only via
`ml use /software/setonix/unsupported/`, a second, undocumented module
tree Pawsey doesn't officially support or commit to maintaining. We've
asked Pawsey to promote this to a supported path (see the support
ticket drafted separately); until/unless that happens, this is what
we've got, and it does work.

## One-time setup

```bash
cd /scratch/pawsey1384/lockstoch

git clone -b simple     https://github.com/LockStoch/periscope-jax.git periscope-simple
git clone -b jax-simple https://github.com/LockStoch/periscope-jax.git periscope-jax

mkdir -p jax-test/meshes
cd jax-test/meshes
wget https://github.com/dengwirda/periscope/releases/download/data-v1/mesh_w_elev_cvt_7.zip
unzip mesh_w_elev_cvt_7.zip
cd ../..

module load python/3.11.6

# simple: CPU, NumPy
python3 -m venv jax-test/simple-venv
source jax-test/simple-venv/bin/activate
pip install -r periscope-simple/requirements.txt
deactivate

# jax-simple: CPU, ordinary pip jax
python3 -m venv jax-test/jax-venv
source jax-test/jax-venv/bin/activate
pip install -r periscope-jax/requirements.txt
deactivate

# jax-simple: GPU, needs the unsupported ROCm tree first
ml use /software/setonix/unsupported/
ml rocm/7.2.4
python3 -m venv jax-test/jax-gpu-venv
source jax-test/jax-gpu-venv/bin/activate
pip install -r periscope-jax/requirements.txt
pip install -U "jax[rocm7-local]"
deactivate
```

## Run

```bash
cd /scratch/pawsey1384/lockstoch/jax-test
sbatch run_simple.slurm
sbatch run_jax.slurm
sbatch run_jax_gpu.slurm
```

Each writes to its own output file, so all three can run at once
without colliding:

| Run | Output |
|---|---|
| `simple` | `periscope-simple/out_jet_cvt_7.nc` |
| `jax-simple`, CPU | `periscope-jax/out_jet_cvt_7_cpu.nc` |
| `jax-simple`, GPU | `periscope-jax/out_jet_cvt_7_gpu.nc` |

(Note the `_cpu`/`_gpu` suffixes -- before this third experiment,
the CPU jax-simple output was just `out_jet_cvt_7.nc`. If you have
an old file at that name from before, it's the CPU run's output,
just not renamed.)

`run_jax_gpu.slurm` prints `JAX devices: [...]` right at the start,
before running anything else -- check that log first if the job
finishes suspiciously fast or the numbers look wrong. If it says
`CpuDevice` instead of a ROCm/GPU device, the run silently fell back
to CPU rather than actually using the GPU.

## Compare correctness

```bash
source jax-test/simple-venv/bin/activate   # any venv works, all have netCDF4 + matplotlib
python3 jax-test/compare.py \
    periscope-simple/out_jet_cvt_7.nc \
    periscope-jax/out_jet_cvt_7_cpu.nc \
    --outdir jax-test/plots-cpu

python3 jax-test/compare.py \
    periscope-simple/out_jet_cvt_7.nc \
    periscope-jax/out_jet_cvt_7_gpu.nc \
    --outdir jax-test/plots-gpu
```

Prints a table of max/RMS relative difference in `hh_cell`/`uu_edge`
at each saved day, plus volume drift, and writes four plots per
comparison (see `compare.py`'s docstring for what each one shows).

## Timing

Each job script pulls `swe.py`'s own printed `*wall-time`/`*file-i/o.`
lines out of its SLURM log and appends a row to `jax-test/timing.csv`
(created automatically on first run). One shared file across all three
experiments, so CPU-vs-GPU speed is directly comparable:

```
timestamp,run,job_id,wall_time_sec,file_io_sec
2026-09-03T...,simple-jet,12345,1264.2,3.1
2026-09-03T...,jax-cpu-jet,12346,268.4,2.8
2026-09-03T...,jax-gpu-jet,12347,...,...
```

`wall_time_sec` is just the integration loop (matches `slv.py`'s own
`*wall-time` print), not mesh loading/setup, which is roughly fixed
cost either way and not what a GPU speeds up. Worth keeping this file
around and appending to it over repeated runs rather than resetting it,
so it builds up a real history rather than a single snapshot.
