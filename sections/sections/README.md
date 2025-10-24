# Microtubule_Simulation

A simulator for quantum coherence in microtubular structures with optional open-quantum-system extensions. This implementation focuses on Option B (SSE — stochastic Schrödinger equation) to model Tegmark-style dephasing as spatially varying, cytokine‑modulated noise.

Highlights
- Geometry: cylindrical grids (r, z) with Fibonacci and regular variants.
- Environments: HIV phase–dependent cytokine fields drive dephasing.
- Option B (SSE) dephasing:
  - SSE_local (uncorrelated noise)
  - SSE_correlated (Gaussian‑kernel spatially correlated noise)
- Data and figures: Γ_map and kernels saved; coherence metrics and CSV time series exported; overlay plots produced.
- Stability safeguards: RNG seeding, dt·Γ guard/clipping, convergence checklist and small‑grid demo.

Environment setup (virtual environment)
- Create and use a local virtual environment (.venv) to isolate dependencies:
  - make venv        # creates .venv
  - make install     # installs requirements into .venv
  - source .venv/bin/activate  # macOS/Linux; on Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
- Core deps are in requirements.txt (numpy, scipy, matplotlib). You can export exact versions with:
  - make pip-freeze   # writes requirements.lock.txt from the active venv

Quickstart
1) Minimal run (local SSE):
   
   from quantum.microtubule_quantum_coherence_full_simulation import MicrotubuleQuantumSimulator
   sim = MicrotubuleQuantumSimulator(config={
       'dephasing_model': 'SSE_local',
       'hiv_phase': 'acute',
       'rng_seed': 42,
       'N_r': 40, 'N_z': 40, 'dt': 0.01, 'time_steps': 80, 'frames_to_save': 8,
   })
   sim.run_simulation()
   sim.save_data()

2) Correlated SSE (set correlation length ξ in grid units):
   
   sim = MicrotubuleQuantumSimulator(config={
       'dephasing_model': 'SSE_correlated',
       'corr_length_xi': 0.8,
   })

Outputs
- Where are outputs saved?
  - Project directories:
    - datafiles/: JSON summaries, CSV time series, NPZ arrays (|ψ|^2, cytokine_field, Γ_map_sse, and sse_kernel for correlated runs)
    - figures/: PNG overlays and other figures
    - results/: Markdown and JSON reports from interpretation tools (Extra/sse_interpret.py, geometry_compare.py)
  - Desktop mirror (convenience copies):
    - ~/Desktop/microtubule_simulation/datafiles
    - ~/Desktop/microtubule_simulation/figures
- What files are written per run?
  - Summary JSON: {run}_summary.json (includes config, timestamps, SSE metrics, flags: Gamma_map_sse_final_present, SSE_kernel_present, dt_gamma_guard_triggered)
  - CSV: {run}_sse_coherence.csv with columns [time, sse_coherence_reg, sse_coherence_fib]
  - NPZ arrays: {run}_arrays.npz (final |ψ|^2, cytokine_final, event_horizon_final)
  - NPZ Γ-map: {run}_gamma_map_sse.npz (when SSE enabled)
  - NPZ kernel: {run}_sse_kernel.npz (SSE_correlated mode only)
- Figures:
  - Overlay PNG: {run}_overlay.png (final |ψ|^2 with Γ_map overlay)

Configuration (key parameters)
- hiv_phase: one of {"none","acute","art_controlled","chronic"} — scales Γ via cytokine field
- Gamma_0, alpha_c: baseline and cytokine coupling used in Γ_map
- dephasing_model: "none" | "SSE_local" | "SSE_correlated"
- corr_length_xi: correlation length for SSE_correlated (grid units)
- rng_seed: seed for reproducible stochastic steps
- dt_gamma_guard_max: threshold; if max(Γ)*dt exceeds this, Γ is clipped and a warning recorded
- temperature_K, ionic_strength_M, dielectric_rel, gamma_scale_alpha: environmental context and global scaling used in Γ_map construction and analyses

Validation helpers (Extra/)
- sse_validation.py — short runs for SSE_local and SSE_correlated; prints norms and coherence
  - python Extra/sse_validation.py
- tegmark_compare.py — order‑of‑magnitude analytical Γ_Tegmark vs simulated decay fit
  - python Extra/tegmark_compare.py path/to/{run}_summary.json --delta_x 1e-9
- sse_phase_sweep.py — multi‑trajectory phase sweep with variance bands and half‑life stats
  - python Extra/sse_phase_sweep.py --trajs 16 --steps 200 --phase acute
- sse_demo_config.py — prints recommended small‑grid settings; with --run executes a demo
  - python Extra/sse_demo_config.py --run
- sse_smoke_test.py — lightweight test ensuring norms ~1 and finite metrics
  - python Extra/sse_smoke_test.py

Developer references
- quantum/open_systems.py — SSE implementation (Γ_map builder, local and correlated SSE steps, metrics wrapper)
- quantum/shims.py — stable imports for cylindrical‑aware metrics from legacy directories
- quantum/microtubule_quantum_coherence_full_simulation.py — main simulator with SSE wiring and outputs
- project docs/Option_B_SSE_Quickstart.md — usage details
- project docs/SSE_Convergence_Checklist.md — convergence and stability guidance
- Project Roadmap — status of the SSE roadmap and future work

Notes
- Γ_map units are currently relative to simulator units; Extra/tegmark_compare.py provides order‑of‑magnitude comparisons to analytical estimates.
- If the dt·Γ guard frequently triggers, consider reducing dt, Gamma_0/alpha_c, or gamma_scale_alpha.

Citations
- M. Tegmark, Phys. Rev. E 61, 4194 (2000)
- H. P. Breuer, F. Petruccione, The Theory of Open Quantum Systems (2002)
- Haken–Strobl–Reineker model (1972) for dephasing in excitonic systems
- Additional context and citations appear in project docs and inline docstrings.



---

Terminal quickstart
- Run a local SSE simulation:
  - python -m quantum.cli --mode SSE_local --hiv_phase acute --N_r 36 --N_z 36 --dt 0.01 --time_steps 120 --frames_to_save 12 --rng_seed 1234
- Run a correlated SSE simulation (xi=0.8 grid units):
  - python -m quantum.cli --mode SSE_correlated --xi 0.8 --hiv_phase acute --N_r 36 --N_z 36 --dt 0.01 --time_steps 120 --frames_to_save 12 --rng_seed 1234
- Use Makefile shortcuts (if make is available):
  - make help
  - make run-local
  - make run-corr
  - make validate
  - make smoke
  - make tegmark-compare SUMMARY=path/to/*_summary.json DELTA_X=1e-9
  - make phase-sweep K=8 phases='none art_controlled chronic acute' MODE=SSE_local XI=0.8
  - make demo-config
  - make demo-run

Notes
- CLI maps directly to the simulator configuration; see project docs/Option_B_SSE_Quickstart.md for parameter details.
- Outputs are saved under figures/ and datafiles/ and mirrored to ~/Desktop/microtubule_simulation/.



### Visualization
- A lightweight helper is available to visualize SSE outputs:
  - Coherence time series from CSV
  - Final |ψ|² heatmaps and Γ_map overlays (from saved NPZ files)
  - Phase‑sweep mean and 10–90% variance bands
  - SSE kernel previews for correlated mode

Usage (helper script)
- Coherence time series from CSV (replace with your actual file path; do not type angle brackets):
  - python Extra/sse_visualize.py coherence --csv datafiles/microtubule_simulation_acute_20251017_180739_sse_coherence.csv
- Summary overlays (replace with your actual summary path):
  - python Extra/sse_visualize.py summary --summary datafiles/microtubule_simulation_acute_20251017_180719_summary.json
- Phase‑sweep variance bands (default path):
  - python Extra/sse_visualize.py phase --summary-json datafiles/sse_phase_sweep_summary.json
- Kernel preview (replace with your actual kernel path when present):
  - python Extra/sse_visualize.py kernel --kernel datafiles/microtubule_simulation_acute_20251017_180719_sse_kernel.npz

Makefile shortcuts
- make viz-coherence CSV=datafiles/microtubule_simulation_acute_20251017_180739_sse_coherence.csv
- make viz-summary SUMMARY=datafiles/microtubule_simulation_acute_20251017_180719_summary.json
- make viz-latest-coherence   # Auto-detect latest *_sse_coherence.csv
- make viz-latest-summary     # Auto-detect latest *_summary.json
- make viz-phase-sweep
- make viz-kernel KERNEL=path/to/*_sse_kernel.npz
- make viz-latest-kernel      # Auto-detect latest *_sse_kernel.npz (SSE_correlated runs only)

Note: Do not include angle brackets <> in commands. They indicate placeholders in documentation; in zsh, < is treated as redirection and will error. Use your actual file paths or the viz-latest-* targets. Kernels are only saved for SSE_correlated runs; use make run-corr or set dephasing_model="SSE_correlated" to generate them.

Monte Carlo analytics
- Run an ensemble and aggregate results (override variables as needed):
  - make mc-run MC_N=16 MC_MODE=SSE_local MC_PHASES='none art_controlled chronic acute' MC_G0_MIN=0.03 MC_G0_MAX=0.07 MC_A_MIN=0.08 MC_A_MAX=0.12
- Visualize the aggregated distributions:
  - make mc-viz  # expects datafiles/sse_mc_summary.json by default
- Run a tiny smoke test:
  - make mc-smoke
- Direct CLI usage:
  - PYTHONPATH=. python Extra/sse_mc_analytics.py run --N 16 --mode SSE_local --N_r 36 --N_z 36 --dt 0.01 --time_steps 120 --frames_to_save 12 --Gamma0_min 0.03 --Gamma0_max 0.07 --alpha_min 0.08 --alpha_max 0.12 --phases none art_controlled chronic acute
  - PYTHONPATH=. python Extra/sse_mc_analytics.py viz --summary datafiles/sse_mc_summary.json

Outputs
- Figures are written to figures/ and mirrored to ~/Desktop/microtubule_simulation/figures when possible.
- The helper looks for NPZ files created by the simulator (e.g., <base>_arrays.npz, <base>_gamma_map_sse.npz); if unavailable, it falls back to already-saved overlay PNGs when present.



### Data interpretation
- Generate concise reports from saved outputs without re-running simulations.

Usage
- Single run interpretation (Markdown + JSON under results/):
  - make interpret-run SUMMARY=datafiles/your_run_summary.json
  - Or: PYTHONPATH=. python Extra/sse_interpret.py run --summary datafiles/your_run_summary.json --delta_x 1e-9
- Phase-sweep interpretation (expects datafiles/sse_phase_sweep_summary.json):
  - make interpret-phase
  - Or: PYTHONPATH=. python Extra/sse_interpret.py phase --summary-json datafiles/sse_phase_sweep_summary.json
- Monte Carlo interpretation (expects datafiles/sse_mc_summary.json):
  - make interpret-mc
  - Or: PYTHONPATH=. python Extra/sse_interpret.py mc --summary datafiles/sse_mc_summary.json

Notes
- Reports are written to results/ and mirrored to ~/Desktop/microtubule_simulation/results when possible.
- Per-run reports include γ_fit, t1/2, comparison to a Tegmark-style OOM estimate (configurable Δx), and stability flags (dt·Γ guard).



### Geometry comparison (Fibonacci vs. uniform)
- Compare coherence decay and half-life between geometries for a single run:
  - make compare-geometry SUMMARY=datafiles/your_run_summary.json
  - This writes Markdown and JSON under results/ with γ_fit and t1/2 per geometry and a bootstrap estimate of the mean coherence gap Δ(t)=fib−reg.
- Plot reg vs fib coherence from the CSV (with optional Δ panel):
  - make viz-geometry CSV=datafiles/your_run_sse_coherence.csv
- Direct CLI (advanced):
  - PYTHONPATH=. python Extra/geometry_compare.py run --summary datafiles/your_run_summary.json
  - PYTHONPATH=. python Extra/geometry_compare.py viz --csv datafiles/your_run_sse_coherence.csv
- Notes:
  - The CSV is produced when SSE is enabled; if absent, the comparator falls back to time series in the summary JSON.
  - Figures and reports are mirrored to ~/Desktop/microtubule_simulation/ when possible.
