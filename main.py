"""
Kavach pipeline demo (SYNTHETIC DATA ONLY)
Team Ace, SIH 2026, problem statement SIH26186

What this script does, end to end:
  1. Generates made-up duty rosters, leave/transfer history and optional self-check scores.
  2. Builds the features named in the deck: rest gaps, long shifts, night duty, leave gap, etc.
  3. Applies fixed rules (the safety net) and an XGBoost model.
  4. Explains each flag in plain language using SHAP.
  5. Audits results across rank, region and language groups.
  6. Builds a commander view with small groups (< 10) hidden.

IMPORTANT: the labels are produced by our own generator, so any score printed here only shows
that the pipeline runs. It is NOT evidence of real-world accuracy. Do not quote it.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SEED = 42
WEEKS = 8
DAYS = WEEKS * 7
MIN_GROUP = 10            # privacy filter: hide groups smaller than this
MODEL_THRESHOLD = 0.50    # probability above which the model raises a flag
SALT = os.environ.get("KAVACH_SALT", "demo-salt-change-in-production")
OUT = "outputs"
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------------------------
# 1. Synthetic data
# ----------------------------------------------------------------------------------------------
def simulate_person(intensity):
    """One person's 8-week roster. `intensity` (0..1) is how overloaded they are."""
    worked = rng.random(DAYS) < (0.70 + 0.25 * intensity)
    night = rng.random(DAYS) < (0.05 + 0.45 * intensity)
    start = np.where(night, rng.integers(19, 23, DAYS), rng.integers(5, 9, DAYS))
    dur = np.clip(rng.normal(8 + 5 * intensity, 1.5, DAYS), 6, 18)
    days = np.where(worked)[0]
    gaps = []
    for a, b in zip(days[:-1], days[1:]):
        end_prev = a * 24 + start[a] + dur[a]
        gaps.append(max(b * 24 + start[b] - end_prev, 0))
    gaps = np.array(gaps) if gaps else np.array([24.0])
    return {
        "min_rest_gap_hrs": gaps.min(),
        "short_rest_share": float((gaps < 8).mean()),          # share of rest gaps under 8 hours
        "long_shifts_per_week": float(((dur > 12) & worked).sum() / WEEKS),
        "night_shifts_per_week": float((night & worked).sum() / WEEKS),
    }


def make_dataset():
    sizes = [4, 7, 9] + [int(x) for x in rng.integers(25, 110, 27)]   # a few tiny units on purpose
    rows = []
    pid = 0
    for unit_id, size in enumerate(sizes):
        unit_load = rng.beta(2, 4)                                   # some units are busier
        for _ in range(size):
            intensity = float(np.clip(rng.normal(unit_load, 0.18), 0, 1))
            f = simulate_person(intensity)
            f["person_id"] = pid
            f["unit"] = f"Unit-{unit_id:02d}"
            f["rank_group"] = rng.choice(["Junior", "Mid", "Senior"], p=[0.5, 0.35, 0.15])
            f["region"] = rng.choice(["North", "East", "West", "South"])
            f["language_group"] = rng.choice(["Lang-1", "Lang-2", "Lang-3"], p=[0.5, 0.3, 0.2])
            f["days_since_leave"] = float(np.clip(rng.normal(60 + 450 * intensity, 90), 0, 720))
            f["transfers_24m"] = int(np.clip(rng.poisson(0.4 + 1.6 * intensity), 0, 5))
            f["sleep_checkin_hrs"] = float(np.clip(7.3 - 1.8 * intensity + rng.normal(0, 0.6), 3, 9))
            # Hidden factors the model never sees (family, health, etc.)
            hidden = rng.normal(0, 6)
            latent = 25 * intensity + 3 * (f["transfers_24m"] > 2) + hidden + rng.normal(0, 3)
            f["_latent_stress"] = latent
            # Optional self-check (PSS-10-like, 0..40). ~60% opt in; the rest stay missing.
            # DELIBERATE DEMO EFFECT: Lang-3 under-reports by 25% so the fairness audit has something to find.
            report = 0.75 if f["language_group"] == "Lang-3" else 1.0
            opted_in = rng.random() < 0.60
            f["self_check_score"] = (float(np.clip(latent * report + rng.normal(0, 4), 0, 40))
                                     if opted_in else np.nan)
            rows.append(f)
            pid += 1
    df = pd.DataFrame(rows)
    cut = np.quantile(df["_latent_stress"], 0.80)                     # top 20% = "high stress" proxy label
    df["high_stress_proxy"] = (df["_latent_stress"] >= cut).astype(int)
    return df.drop(columns=["_latent_stress"])


FEATURES = ["min_rest_gap_hrs", "short_rest_share", "long_shifts_per_week", "night_shifts_per_week",
            "days_since_leave", "transfers_24m", "sleep_checkin_hrs", "self_check_score"]
# Group columns (rank, region, language) are NOT model features. They are only used for the audit.


# ----------------------------------------------------------------------------------------------
# 2. Fixed rules (safety net). Thresholds are illustrative and need expert review.
# ----------------------------------------------------------------------------------------------
def rule_flags(df):
    r1 = (df.short_rest_share >= 0.25) & (df.long_shifts_per_week >= 2)
    r2 = (df.days_since_leave >= 365) & (df.night_shifts_per_week >= 2.5)
    r3 = df.self_check_score.fillna(0) >= 30
    return (r1 | r2 | r3).astype(int)


# ----------------------------------------------------------------------------------------------
# 3. Model: out-of-fold predictions (units never appear in both train and test)
# ----------------------------------------------------------------------------------------------
def new_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)


def out_of_fold_scores(df):
    scores = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(df, df.high_stress_proxy, df.unit):
        m = new_model().fit(df.iloc[tr][FEATURES], df.iloc[tr].high_stress_proxy)
        scores[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
    return scores


# ----------------------------------------------------------------------------------------------
# 4. Plain-language SHAP reasons
# ----------------------------------------------------------------------------------------------
TEXT = {
    "min_rest_gap_hrs": "Shortest rest between shifts: {:.1f} h",
    "short_rest_share": "Rest gaps under 8 h: {:.0%} of gaps",
    "long_shifts_per_week": "Shifts over 12 h: {:.1f} per week",
    "night_shifts_per_week": "Night shifts: {:.1f} per week",
    "days_since_leave": "Days since last leave: {:.0f}",
    "transfers_24m": "Transfers in 24 months: {:.0f}",
    "sleep_checkin_hrs": "Sleep check-in: {:.1f} h",
    "self_check_score": "Self-check score: {:.0f} / 40",
}


def pseudonym(person_id):
    return "P-" + hashlib.sha256(f"{SALT}{person_id}".encode()).hexdigest()[:8]


def explain_flags(df, model, flagged_idx, top_k=3):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(df.loc[flagged_idx, FEATURES])
    out = []
    for row, (i, s) in enumerate(zip(flagged_idx, sv)):
        order = np.argsort(-s)[:top_k]
        reasons = []
        for j in order:
            if s[j] <= 0:
                continue
            val = df.loc[i, FEATURES[j]]
            if pd.isna(val):
                continue
            reasons.append(TEXT[FEATURES[j]].format(val))
        out.append(" | ".join(reasons) or "No single dominant factor")
    return out


# ----------------------------------------------------------------------------------------------
# 5. Fairness audit and 6. Privacy filter
# ----------------------------------------------------------------------------------------------
def fairness_audit(df):
    rows = []
    for col in ["rank_group", "region", "language_group"]:
        for g, part in df.groupby(col):
            pos = part[part.high_stress_proxy == 1]
            flagged = part[part.final_flag == 1]
            rows.append({
                "attribute": col, "group": g, "n": len(part),
                "flag_rate": round(part.final_flag.mean(), 3),
                "recall (true high-stress caught)": round(pos.final_flag.mean(), 3) if len(pos) else np.nan,
                "precision (flags that were right)": round(flagged.high_stress_proxy.mean(), 3) if len(flagged) else np.nan,
            })
    return pd.DataFrame(rows)


def commander_view(df):
    rows = []
    for unit, part in df.groupby("unit"):
        if len(part) < MIN_GROUP:
            rows.append({"unit": unit, "personnel": "suppressed (<%d)" % MIN_GROUP, "flagged": "suppressed", "flag_rate": "suppressed"})
        else:
            rows.append({"unit": unit, "personnel": len(part), "flagged": int(part.final_flag.sum()),
                         "flag_rate": f"{part.final_flag.mean():.0%}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Vercel looks specifically for this top-level 'app' variable
app = FastAPI(title="Project Kavach Pipeline API")

# Enable CORS so your HTML/Vercel frontend can talk to this endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def run_pipeline():
    """Runs the Kavach synthetic pipeline and returns demo flags and commander views."""
    df = make_dataset()
    df["rule_flag"] = rule_flags(df)
    df["model_score"] = out_of_fold_scores(df)
    df["model_flag"] = (df.model_score >= MODEL_THRESHOLD).astype(int)
    df["final_flag"] = ((df.rule_flag == 1) | (df.model_flag == 1)).astype(int)

    # Train model to generate SHAP reasons
    final_model = new_model().fit(df[FEATURES], df.high_stress_proxy)
    top = df[df.final_flag == 1].sort_values("model_score", ascending=False).head(5).index
    reasons = explain_flags(df, final_model, top)

    # Prepare response payload
    example_flags = []
    for i, reason in zip(top, reasons):
        example_flags.append({
            "pseudonym": pseudonym(df.loc[i, "person_id"]),
            "unit": str(df.loc[i, "unit"]),
            "model_score": float(round(df.loc[i, "model_score"], 2)),
            "rule_flag": int(df.loc[i, "rule_flag"]),
            "reasons_shap": reason
        })

    view = commander_view(df)

    return {
        "status": "success",
        "total_personnel": len(df),
        "high_stress_flagged": int(df.final_flag.sum()),
        "example_flags": example_flags,
        "commander_view": view.to_dict(orient="records")
    }
