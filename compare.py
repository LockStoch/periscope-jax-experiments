#!/usr/bin/env python3
"""Compare simple vs jax-simple output: diagnostics, tables and plots.

Usage:
    python3 compare.py <simple_output.nc> <jax_output.nc> [--outdir DIR]

Produces, in --outdir (default: current directory):
    diff_timeseries.png   hh_cell / uu_edge relative difference over time
    volume_timeseries.png total water volume for each run, and its drift
    hh_cell_maps.png      spatial maps of hh_cell: simple, jax-simple, diff
    uu_edge_maps.png      same, for uu_edge

and prints a text summary table to the console.
"""

import os
import sys
import argparse

import numpy as np
import netCDF4 as nc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(path):
    d = nc.Dataset(path, "r")
    data = {
        "hh_cell":  np.asarray(d.variables["hh_cell"][:, :, 0]),  # (time, ncell)
        "uu_edge":  np.asarray(d.variables["uu_edge"][:, :, 0]),  # (time, nedge)
        "lonCell":  np.asarray(d.variables["lonCell"][:]),
        "latCell":  np.asarray(d.variables["latCell"][:]),
        "areaCell": np.asarray(d.variables["areaCell"][:]),
        "lonEdge":  np.asarray(d.variables["lonEdge"][:]),
        "latEdge":  np.asarray(d.variables["latEdge"][:]),
    }
    d.close()
    return data


def rel_diff_series(a, b):
    """Per saved-step max and RMS relative difference between two
    (time, n) arrays, relative to B's own magnitude at that step."""
    diff = np.abs(a - b)
    scale = np.abs(b).max(axis=1, keepdims=True) + np.finfo(np.float32).eps
    rel = diff / scale
    return rel.max(axis=1), np.sqrt((rel ** 2).mean(axis=1))


def volume_series(hh_cell, area_cell):
    """Total water volume at each saved step: sum(h * cell area)."""
    return hh_cell @ area_cell


def plot_diff_timeseries(days, hh_max, hh_rms, uu_max, uu_rms, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].semilogy(days, hh_max, "o-", label="max", color="#b5651d")
    axes[0].semilogy(days, hh_rms, "o-", label="rms", color="#1b5e63")
    axes[0].set_title("hh_cell relative difference")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("relative difference")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy(days, uu_max, "o-", label="max", color="#b5651d")
    axes[1].semilogy(days, uu_rms, "o-", label="rms", color="#1b5e63")
    axes[1].set_title("uu_edge relative difference")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("relative difference")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_volume(days, vol_simple, vol_jax, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(days, vol_simple, "o-", label="simple", color="#1b5e63")
    axes[0].plot(days, vol_jax, "o-", label="jax-simple", color="#b5651d")
    axes[0].set_title("total volume")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("sum(hh_cell * area)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    rel_drift_simple = (vol_simple - vol_simple[0]) / vol_simple[0]
    rel_drift_jax = (vol_jax - vol_jax[0]) / vol_jax[0]
    axes[1].plot(days, rel_drift_simple, "o-", label="simple", color="#1b5e63")
    axes[1].plot(days, rel_drift_jax, "o-", label="jax-simple", color="#b5651d")
    axes[1].set_title("volume drift, relative to day 0")
    axes[1].set_xlabel("day")
    axes[1].set_ylabel("relative drift")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_field_maps(lon, lat, a0, b0, aN, bN, day_n, varname, outpath):
    """2 rows (day 0, day N) x 3 cols (simple, jax-simple, difference).

    Colour scale is shared between simple/jax-simple within a row, and
    the diff panel is scaled to its own row: each day gets whatever
    range actually shows its structure, rather than day 0 being washed
    out by day N's (likely much larger) range, or vice versa.
    """
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5),
                              subplot_kw={"aspect": "equal"})

    rows = [(a0, b0, "day 0"), (aN, bN, f"day {day_n}")]

    for r, (a, b, label) in enumerate(rows):
        vmin, vmax = min(a.min(), b.min()), max(a.max(), b.max())
        diff = b - a
        dmax = np.abs(diff).max()
        dmax = dmax if dmax > 0 else 1.0

        panels = [
            (a,    "viridis", vmin,  vmax, f"simple, {label}"),
            (b,    "viridis", vmin,  vmax, f"jax-simple, {label}"),
            (diff, "RdBu_r", -dmax,  dmax, f"difference, {label}"),
        ]
        for c, (field, cmap, lo, hi, title) in enumerate(panels):
            ax = axes[r, c]
            sc = ax.scatter(lon_deg, lat_deg, c=field, s=1.2,
                             cmap=cmap, vmin=lo, vmax=hi,
                             linewidths=0, rasterized=True)
            ax.set_title(title)
            if r == 1:
                ax.set_xlabel("longitude")
            if c == 0:
                ax.set_ylabel("latitude")
            fig.colorbar(sc, ax=ax, shrink=0.75)

    fig.suptitle(varname)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("simple_nc")
    ap.add_argument("jax_nc")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    a = load_run(args.simple_nc)
    b = load_run(args.jax_nc)

    na, nb = a["hh_cell"].shape[0], b["hh_cell"].shape[0]
    if na != nb:
        print(f"warning: simple has {na} saved steps, jax-simple has {nb}; "
              f"comparing the first {min(na, nb)} only")
    nframe = min(na, nb)
    days = np.arange(nframe)

    hh_max, hh_rms = rel_diff_series(a["hh_cell"][:nframe], b["hh_cell"][:nframe])
    uu_max, uu_rms = rel_diff_series(a["uu_edge"][:nframe], b["uu_edge"][:nframe])

    vol_simple = volume_series(a["hh_cell"][:nframe], a["areaCell"])
    vol_jax = volume_series(b["hh_cell"][:nframe], b["areaCell"])

    print(f"{'day':>4}  {'hh max':>10}  {'hh rms':>10}  {'uu max':>10}  {'uu rms':>10}")
    for t in range(nframe):
        print(f"{t:>4}  {hh_max[t]:>10.3e}  {hh_rms[t]:>10.3e}  "
              f"{uu_max[t]:>10.3e}  {uu_rms[t]:>10.3e}")

    print()
    print("volume drift over run, relative to day 0:")
    print(f"  simple:     {(vol_simple[-1] - vol_simple[0]) / vol_simple[0]:+.3e}")
    print(f"  jax-simple: {(vol_jax[-1] - vol_jax[0]) / vol_jax[0]:+.3e}")

    plot_diff_timeseries(
        days, hh_max, hh_rms, uu_max, uu_rms,
        os.path.join(args.outdir, "diff_timeseries.png"))

    plot_volume(
        days, vol_simple, vol_jax,
        os.path.join(args.outdir, "volume_timeseries.png"))

    plot_field_maps(
        a["lonCell"], a["latCell"],
        a["hh_cell"][0], b["hh_cell"][0],
        a["hh_cell"][nframe - 1], b["hh_cell"][nframe - 1],
        nframe - 1, "hh_cell",
        os.path.join(args.outdir, "hh_cell_maps.png"))

    plot_field_maps(
        a["lonEdge"], a["latEdge"],
        a["uu_edge"][0], b["uu_edge"][0],
        a["uu_edge"][nframe - 1], b["uu_edge"][nframe - 1],
        nframe - 1, "uu_edge",
        os.path.join(args.outdir, "uu_edge_maps.png"))

    print()
    print(f"plots saved to {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
