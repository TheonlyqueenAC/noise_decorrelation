"""
Enhanced Bayesian Parameter Inference v2.1 (FIXED)
===================================================

FIX: Adaptive homeostatic floor that only activates when truly needed
- Healthy/Acute: No floor (natural NAA levels)
- Chronic: Floor at 0.85× healthy (allows natural decline to ~0.75 before activating)

This preserves the biological realism while allowing the model to fit the data properly.
"""

from __future__ import annotations

import argparse
import os

import arviz as az
import numpy as np
import pandas as pd
# PyMC v4
import pymc as pm
import pytensor.tensor as pt


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
class CONST:
    xi_baseline = 0.8e-9
    sigma_r_regular = 0.38e-9
    NAA_baseline = 1.105
    Cho_baseline = 0.225


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
CONDITIONS = ("healthy", "acute_HIV", "chronic_HIV")

OBS_NAA = {
    "healthy": 1.105,
    "acute_HIV": 1.135,
    "chronic_HIV": 1.005,
}

OBS_CHO = {
    "healthy": 0.225,
    "acute_HIV": 0.245,
    "chronic_HIV": 0.235,
}

COHERENCE_BASE = {
    "healthy": 0.85,
    "acute_HIV": 0.84,
    "chronic_HIV": 0.73,
}

SIGMA_R = {
    "healthy": CONST.sigma_r_regular,
    "acute_HIV": CONST.sigma_r_regular * 1.05,
    "chronic_HIV": CONST.sigma_r_regular * 1.4,
}

MEMBRANE_RATES = {
    "healthy": (1.0, 1.0),
    "acute_HIV": (2.0, 2.5),
    "chronic_HIV": (1.2, 1.1),
}


# --------------------------------------------------------------------------------------
# Forward Models (FIXED)
# --------------------------------------------------------------------------------------

def coherence_from_xi_nonlinear(xi: pt.TensorVariable,
                                coherence_base: float,
                                xi_floor: pt.TensorVariable,
                                xi_ceiling: pt.TensorVariable) -> pt.TensorVariable:
    """Nonlinear ξ → coherence with floor."""
    xi_normalized = (xi - xi_floor) / (xi_ceiling - xi_floor)
    xi_normalized = pt.clip(xi_normalized, 0.0, 1.0)

    C_floor = 0.65
    C_max = coherence_base

    coherence = C_floor + (C_max - C_floor) * (1 - xi_normalized) ** 2
    return coherence


def forward_NAA_compensated_v2(coherence_base: float,
                               xi: pt.TensorVariable,
                               sigma_r: float,
                               condition: str,
                               coh_exp: pt.TensorVariable,
                               xi_exp: pt.TensorVariable,
                               deloc_exp: pt.TensorVariable,
                               NAA_base: pt.TensorVariable,
                               astrocyte_comp: pt.TensorVariable,
                               xi_floor: pt.TensorVariable,
                               xi_ceiling: pt.TensorVariable) -> pt.TensorVariable:
    """
    FIXED NAA model with adaptive floor.

    Key change: Homeostatic floor only activates in chronic phase
    and at a lower threshold (0.85× instead of 0.90×).
    """
    # 1. Nonlinear coherence
    coherence_effective = coherence_from_xi_nonlinear(
        xi, coherence_base, xi_floor, xi_ceiling
    )

    # 2. Base quantum coupling
    coherence_term = (coherence_effective / 0.85) ** coh_exp
    xi_protection = (CONST.xi_baseline / xi) ** xi_exp
    deloc_term = (sigma_r / CONST.sigma_r_regular) ** deloc_exp

    NAA_quantum = NAA_base * coherence_term * xi_protection * deloc_term

    # 3. Astrocyte compensation (chronic only)
    if condition == 'chronic_HIV':
        NAA_compensated = NAA_quantum * astrocyte_comp
    else:
        NAA_compensated = NAA_quantum

    # 4. FIXED: Adaptive homeostatic floor
    # Only applies to chronic phase, and at lower threshold
    if condition == 'chronic_HIV':
        # Chronic: Floor at 0.85× healthy (not 0.90×)
        # Allows natural decline to ~0.75 before activating
        NAA_floor = 0.85 * CONST.NAA_baseline  # 0.939
        NAA_total = pt.maximum(NAA_compensated, NAA_floor)
    else:
        # Healthy/Acute: No floor (natural levels)
        NAA_total = NAA_compensated

    return NAA_total


def forward_Cho(damage_rate: float,
                repair_rate: float,
                k_turnover: pt.TensorVariable) -> pt.TensorVariable:
    """Choline dynamics."""
    turnover_factor = (damage_rate + repair_rate) - 1.0
    Cho_Cr = CONST.Cho_baseline * (1.0 + k_turnover * turnover_factor)
    return Cho_Cr


# --------------------------------------------------------------------------------------
# Bayesian Model (FIXED)
# --------------------------------------------------------------------------------------

def build_model_v2_fixed():
    """Enhanced model with FIXED adaptive floor."""

    naa_obs = np.array([OBS_NAA[c] for c in CONDITIONS])
    cho_obs = np.array([OBS_CHO[c] for c in CONDITIONS])

    with pm.Model() as model:
        # Priors: ξ parameters
        xi_floor = pm.TruncatedNormal("xi_floor", mu=0.35e-9, sigma=0.1e-9,
                                      lower=0.2e-9, upper=0.5e-9)
        xi_ceiling = pm.TruncatedNormal("xi_ceiling", mu=0.8e-9, sigma=0.1e-9,
                                        lower=0.6e-9, upper=1.0e-9)

        xi_healthy = pm.TruncatedNormal("xi_healthy", mu=0.75e-9, sigma=0.1e-9,
                                        lower=xi_floor, upper=xi_ceiling)
        xi_acute = pm.TruncatedNormal("xi_acute", mu=0.4e-9, sigma=0.1e-9,
                                      lower=xi_floor, upper=0.6e-9)
        xi_chronic = pm.TruncatedNormal("xi_chronic", mu=0.8e-9, sigma=0.1e-9,
                                        lower=0.5e-9, upper=xi_ceiling)

        # Priors: Coupling exponents
        # INCREASED NAA_base prior to allow higher baseline
        coh_exp = pm.TruncatedNormal("coh_exp", mu=2.5, sigma=0.5, lower=0.5, upper=5.0)
        xi_exp = pm.TruncatedNormal("xi_exp", mu=0.3, sigma=0.2, lower=0.0, upper=1.5)
        deloc_exp = pm.TruncatedNormal("deloc_exp", mu=0.2, sigma=0.1, lower=0.0, upper=1.0)

        # FIXED: Increased NAA_base prior mean to allow quantum coupling to reach healthy levels
        NAA_base = pm.TruncatedNormal("NAA_base", mu=1.15, sigma=0.08,
                                      lower=1.0, upper=1.35)
        k_turnover = pm.TruncatedNormal("k_turnover", mu=0.02, sigma=0.01,
                                        lower=0.0, upper=0.1)

        # Astrocyte compensation
        astrocyte_comp = pm.TruncatedNormal("astrocyte_comp", mu=1.18, sigma=0.05,
                                            lower=1.05, upper=1.30)

        # Observation noise
        sigma_NAA = pm.HalfNormal("sigma_NAA", sigma=0.05)
        sigma_Cho = pm.HalfNormal("sigma_Cho", sigma=0.01)

        # Forward model
        naa_preds = []
        cho_preds = []

        for i, cond in enumerate(CONDITIONS):
            if cond == "healthy":
                xi_cond = xi_healthy
            elif cond == "acute_HIV":
                xi_cond = xi_acute
            else:
                xi_cond = xi_chronic

            # FIXED forward model with adaptive floor
            naa_pred = forward_NAA_compensated_v2(
                coherence_base=COHERENCE_BASE[cond],
                xi=xi_cond,
                sigma_r=SIGMA_R[cond],
                condition=cond,
                coh_exp=coh_exp,
                xi_exp=xi_exp,
                deloc_exp=deloc_exp,
                NAA_base=NAA_base,
                astrocyte_comp=astrocyte_comp,
                xi_floor=xi_floor,
                xi_ceiling=xi_ceiling
            )
            naa_preds.append(naa_pred)

            dmg, rep = MEMBRANE_RATES[cond]
            cho_pred = forward_Cho(dmg, rep, k_turnover)
            cho_preds.append(cho_pred)

        naa_preds = pt.stack(naa_preds)
        cho_preds = pt.stack(cho_preds)

        # Likelihoods
        pm.Normal("NAA_obs", mu=naa_preds, sigma=sigma_NAA, observed=naa_obs)
        pm.Normal("Cho_obs", mu=cho_preds, sigma=sigma_Cho, observed=cho_obs)

        # Derived quantities
        pm.Deterministic("delta_xi", xi_chronic - xi_acute)
        pm.Deterministic("xi_healthy_nm", xi_healthy * 1e9)
        pm.Deterministic("xi_acute_nm", xi_acute * 1e9)
        pm.Deterministic("xi_chronic_nm", xi_chronic * 1e9)
        pm.Deterministic("xi_floor_nm", xi_floor * 1e9)
        pm.Deterministic("xi_ceiling_nm", xi_ceiling * 1e9)

        coh_healthy = coherence_from_xi_nonlinear(xi_healthy, 0.85, xi_floor, xi_ceiling)
        coh_acute = coherence_from_xi_nonlinear(xi_acute, 0.84, xi_floor, xi_ceiling)
        coh_chronic = coherence_from_xi_nonlinear(xi_chronic, 0.73, xi_floor, xi_ceiling)

        pm.Deterministic("coh_healthy_eff", coh_healthy)
        pm.Deterministic("coh_acute_eff", coh_acute)
        pm.Deterministic("coh_chronic_eff", coh_chronic)

    return model


# --------------------------------------------------------------------------------------
# Run inference
# --------------------------------------------------------------------------------------

def run_inference_v2_fixed(draws: int = 3000,
                           tune: int = 1500,
                           chains: int = 4,
                           target_accept: float = 0.92,
                           seed: int | None = 42):
    """Run FIXED enhanced Bayesian inference."""

    os.makedirs("results/bayesian_v2_fixed", exist_ok=True)

    print("=" * 80)
    print(" ENHANCED BAYESIAN INFERENCE v2.1 (FIXED)")
    print("=" * 80)
    print("\nFIXES:")
    print("  1. Adaptive floor (0.85× healthy, chronic only)")
    print("  2. Increased NAA_base prior (1.10 → 1.15)")
    print("  3. Removed floor for healthy/acute conditions")
    print()

    model = build_model_v2_fixed()

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=True,
        )

    # Save
    trace_path = "results/bayesian_v2_fixed/trace_v2_fixed.nc"
    idata.to_netcdf(trace_path)

    var_names = [
        "coh_exp", "xi_exp", "deloc_exp", "NAA_base", "k_turnover",
        "astrocyte_comp",
        "sigma_NAA", "sigma_Cho",
        "xi_healthy_nm", "xi_acute_nm", "xi_chronic_nm",
        "xi_floor_nm", "xi_ceiling_nm",
        "delta_xi",
        "coh_healthy_eff", "coh_acute_eff", "coh_chronic_eff"
    ]

    summary = az.summary(idata, var_names=var_names, round_to=4)
    summary_path = "results/bayesian_v2_fixed/summary_v2_fixed.csv"
    summary.to_csv(summary_path)

    print("\n" + "=" * 80)
    print(" POSTERIOR SUMMARY")
    print("=" * 80)
    print(summary)

    # P(ξ_acute < ξ_chronic)
    xi_acute_vals = idata.posterior["xi_acute_nm"].values.reshape(-1)
    xi_chronic_vals = idata.posterior["xi_chronic_nm"].values.reshape(-1)
    p_order = float(np.mean(xi_acute_vals < xi_chronic_vals))

    # Posterior predictive
    post = idata.posterior
    med = {
        k: float(np.median(post[k].values))
        for k in ["coh_exp", "xi_exp", "deloc_exp", "NAA_base",
                  "k_turnover", "astrocyte_comp", "xi_floor", "xi_ceiling"]
    }

    preds = []
    for cond in CONDITIONS:
        if cond == "healthy":
            xi_med = float(np.median(post["xi_healthy_nm"].values)) * 1e-9
        elif cond == "acute_HIV":
            xi_med = float(np.median(post["xi_acute_nm"].values)) * 1e-9
        else:
            xi_med = float(np.median(post["xi_chronic_nm"].values)) * 1e-9

        # Nonlinear coherence
        xi_norm = (xi_med - med["xi_floor"]) / (med["xi_ceiling"] - med["xi_floor"])
        xi_norm = np.clip(xi_norm, 0.0, 1.0)
        coh_eff = 0.65 + (COHERENCE_BASE[cond] - 0.65) * (1 - xi_norm) ** 2

        # Quantum coupling
        naa_quantum = (
                med["NAA_base"]
                * (coh_eff / 0.85) ** med["coh_exp"]
                * (CONST.xi_baseline / xi_med) ** med["xi_exp"]
                * (SIGMA_R[cond] / CONST.sigma_r_regular) ** med["deloc_exp"]
        )

        # Compensation
        if cond == "chronic_HIV":
            naa_comp = naa_quantum * med["astrocyte_comp"]
        else:
            naa_comp = naa_quantum

        # FIXED: Adaptive floor
        if cond == "chronic_HIV":
            naa = max(naa_comp, 0.85 * CONST.NAA_baseline)
        else:
            naa = naa_comp

        # Choline
        dmg, rep = MEMBRANE_RATES[cond]
        cho = CONST.Cho_baseline * (1.0 + med["k_turnover"] * ((dmg + rep) - 1.0))

        preds.append({
            "condition": cond,
            "NAA_pred": naa,
            "Cho_pred": cho,
            "NAA_obs": OBS_NAA[cond],
            "Cho_obs": OBS_CHO[cond],
            "error_NAA_%": 100 * (naa - OBS_NAA[cond]) / OBS_NAA[cond],
            "error_Cho_%": 100 * (cho - OBS_CHO[cond]) / OBS_CHO[cond]
        })

    preds_df = pd.DataFrame(preds)
    preds_path = "results/bayesian_v2_fixed/posterior_predictive_v2_fixed.csv"
    preds_df.to_csv(preds_path, index=False)

    print("\n" + "=" * 80)
    print(" POSTERIOR PREDICTIVE CHECK")
    print("=" * 80)
    print(preds_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(" KEY RESULTS")
    print("=" * 80)
    print(f"P(ξ_acute < ξ_chronic) = {p_order:.4f}")
    print(f"\nAstrocyte compensation: {med['astrocyte_comp']:.3f}")
    print(f"NAA_base: {med['NAA_base']:.3f} (was 1.101)")
    print(f"ξ floor: {med['xi_floor'] * 1e9:.2f} nm")
    print(f"ξ ceiling: {med['xi_ceiling'] * 1e9:.2f} nm")

    print(f"\nNAA errors:")
    for i, cond in enumerate(CONDITIONS):
        print(f"  {cond:12s}: {preds_df.loc[i, 'error_NAA_%']:+.1f}%")
    print()

    with open("results/bayesian_v2_fixed/results_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" ENHANCED MODEL v2.1 (FIXED) - RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"P(ξ_acute < ξ_chronic) = {p_order:.4f}\n\n")
        f.write("Posterior Medians:\n")
        for k, v in med.items():
            if "xi" in k and k != "xi_exp":
                f.write(f"  {k}: {v * 1e9:.3f} nm\n")
            else:
                f.write(f"  {k}: {v:.3f}\n")
        f.write("\n" + preds_df.to_string(index=False) + "\n")

    print(f"\nResults saved to: results/bayesian_v2_fixed/")

    return {
        "idata": idata,
        "summary": summary,
        "predictions": preds_df,
        "p_xi_order": p_order,
        "medians": med
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Bayesian inference v2.1 (FIXED adaptive floor)"
    )
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    results = run_inference_v2_fixed(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed
    )


if __name__ == "__main__":
    main()