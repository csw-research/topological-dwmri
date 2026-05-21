# topological-dwmri

Topological persistent homology of diffusion-weighted MRI signals.

**Companion paper**: Warioba CS. *Topological persistent homology of
diffusion-weighted MRI: a model-free non-Gaussianity biomarker linked to
Levy-stable displacement statistics*. In review at *Magnetic Resonance
in Medicine* (2026).

**Parent paper**: Warioba CS. *Topological phase transition in the
sample paths of Levy processes*. In review at *Physical Review E*
(2026). [https://github.com/csw-research/topological-levy-phase-transition](https://github.com/csw-research/topological-levy-phase-transition)

## Overview

The parent paper proves that the sublevel-set persistent homology of an
alpha-stable Levy sample path has a power-law lifetime tail with
exponent equal to the stability index alpha. This repository carries
that result over to diffusion-weighted MRI:

* Section 2 of the manuscript derives an explicit cumulative-sum
  construction that maps per-direction HARDI signals to a Levy bridge
  whose persistence-tail exponent estimates the underlying displacement
  alpha.
* Section 3-4 implements forward models for free Gaussian, restricted
  Gaussian, alpha-stable, and mixed compartments, simulates HCP-style
  multi-shell acquisitions, and benchmarks the persistence-tail
  estimator against DKI kurtosis K and the stretched-exponential alpha.
* Section 5 ships a SLURM-ready pipeline to compute voxelwise maps of
  alpha_persistence, K, D, and alpha_se from HCP / MGH Adult Diffusion
  datasets on the Sherlock cluster.

## Layout

```
topological-dwmri/
  src/                    Python package
    generators/           DW-MRI signal and displacement samplers
    estimators/           Persistence-tail, DKI, stretched-exponential fits
    simulation/           Region simulator and experiment scripts
    realdata/             NIfTI I/O and voxelwise map computation
    utils/parent_bridge.py
                          Imports the parent paper's persistence and
                          stable-process modules without modification
  scripts/                CLI entrypoints
    run_all_experiments.py
    process_subject.py
  configs/                YAML configs (placeholder for batch sweeps)
  sherlock/               SLURM batch scripts, requirements, setup
  analysis/make_figures.py
                          Publication-quality figure regeneration
  manuscript/             LaTeX source for the MRM submission
  results/                JSON experiment outputs (regenerable)
  figures/                Compiled PDFs of all manuscript figures
  tests/                  (reserved for unit tests)
```

## Quick start

```bash
git clone https://github.com/csw-research/topological-dwmri.git
git clone https://github.com/csw-research/topological-levy-phase-transition.git \
    ../topological-levy-phase-transition  # or set TOPO_DWMRI_PARENT_PATH

python -m venv .venv && source .venv/bin/activate
pip install -r sherlock/requirements.txt

# Run the simulation study
python scripts/run_all_experiments.py --out-dir results

# Regenerate the manuscript figures
python analysis/make_figures.py
```

The parent paper's repository must be cloned to a sibling directory
(default: `../topological-levy-phase-transition`) or its path must be
exposed through the `TOPO_DWMRI_PARENT_PATH` environment variable.

## Sherlock workflow

```bash
ssh sherlock
cd $SCRATCH
git clone https://github.com/csw-research/topological-dwmri.git
git clone https://github.com/csw-research/topological-levy-phase-transition.git non_gaussian_stochastic
bash topological-dwmri/sherlock/setup_env.sh
sbatch topological-dwmri/sherlock/run_simulation.sbatch
# Then edit topological-dwmri/sherlock/subjects.txt and submit
sbatch topological-dwmri/sherlock/run_hcp_subject.sbatch
```

## Data availability

The simulation study uses no human-subjects data; every figure in the
manuscript is reproducible from the code alone. The in-vivo pipeline
targets the publicly available HCP S1200 and MGH Adult Diffusion
datasets (see manuscript Data and Code Availability).

## Citation

```
@article{warioba2026topodwmri,
  title     = {Topological persistent homology of diffusion-weighted MRI:
               a model-free non-Gaussianity biomarker linked to
               Levy-stable displacement statistics},
  author    = {Warioba, Chisondi S.},
  journal   = {(in review)},
  year      = {2026},
}
```

## Licence

MIT. See `LICENSE`.
