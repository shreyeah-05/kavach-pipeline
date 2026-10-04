"""
Kavach pipeline demo (SYNTHETIC DATA ONLY)
Team Ace, SIH 2026, problem statement SIH26186
"""
import hashlib
import json
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
        "short_rest_share": float((gaps < 8).mean()),
        "long_shifts_per_week": float(((dur > 12) & worked).sum() / WEEKS),
        "night_shifts_per_week": float((night & worked).sum() / WEEKS),
    }


def make_dataset():
    sizes = [4, 7, 9] + [int(x) for x in rng.integers(25, 110, 27)]
    rows = []
    pid = 0
    for unit_id, size in enumerate(sizes):
        unit_load = rng.beta(2, 4)
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
            
            hidden = rng.normal(0, 6)
            latent = 25 * intensity + 3 * (f["transfers_24m"] > 2) + hidden + rng.normal(0, 3)
            f["_latent_stress"] = latent
            
            report = 0.75 if f["language_group"] == "Lang-3" else 1.0
            opted_in = rng.random() < 0.60
            f["self_check_score"] = (float(np.clip(latent * report + rng.normal(0, 4), 0, 40))
                                     if opted_in else np.nan)
            rows.append(f)
            pid += 1
    df = pd.DataFrame(rows)
    cut = np.quantile(df["_latent_stress"], 0.80)
    df["high_stress_proxy"] = (df["_latent_stress"] >= cut).astype(int)
    return df.drop(columns=["_latent_stress"])


FEATURES = ["min_rest_gap_hrs", "short_rest_share", "long_shifts_per_week", "night_shifts_per_week",
            "days_since_leave", "transfers_24m", "sleep_checkin_hrs", "self_check_score"]


# ----------------------------------------------------------------------------------------------
# 2. Fixed rules & Model
# ----------------------------------------------------------------------------------------------
def rule_flags(df):
    r1 = (df.short_rest_share >= 0.25) & (df.long_shifts_per_week >= 2)
    r2 = (df.days_since_leave >= 365) & (df.night_shifts_per_week >= 2.5)
    r3 = df.self_check_score.fillna(0) >= 30
    return (r1 | r2 | r3).astype(int)


def new_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)


def out_of_fold_scores(df):
    scores = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(df, df.high_stress_proxy, df.unit):
        m = new_model().fit(df.iloc[tr][FEATURES], df.iloc[tr].high_stress_proxy)
        scores[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
    return scores


def pseudonym(person_id):
    return "P-" + hashlib.sha256(f"{SALT}{person_id}".encode()).hexdigest()[:8]


# ----------------------------------------------------------------------------------------------
# 3. Export to data.json
# ----------------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    df = make_dataset()
    df["rule_flag"] = rule_flags(df)
    df["model_score"] = out_of_fold_scores(df)
    df["model_flag"] = (df.model_score >= MODEL_THRESHOLD).astype(int)
    df["final_flag"] = ((df.rule_flag == 1) | (df.model_flag == 1)).astype(int)

    # 1. Personnel List for Welfare Officer View
    personnel_list = []
    for _, row in df.head(10).iterrows():
        risk = "High" if row["final_flag"] == 1 else ("Medium" if row["model_score"] > 0.35 else "Low")
        personnel_list.append({
            "id": pseudonym(row["person_id"]),
            "risk": risk,
            "days": int(row["days_since_leave"]),
            "night": round(float(row["night_shifts_per_week"]), 1),
            "tr": int(row["transfers_24m"]),
            "trend": f"Score: {row['model_score']:.2f}",
            "tv": float(row["model_score"])
        })

    # 2. Aggregated Unit Totals for Commander View
    units_list = []
    for unit_name, part in df.groupby("unit"):
        size = len(part)
        if size < MIN_GROUP:
            units_list.append({"name": unit_name, "total": size, "hidden": True})
        else:
            high = int((part.final_flag == 1).sum())
            med = int(((part.final_flag == 0) & (part.model_score > 0.35)).sum())
            low = size - high - med
            units_list.append({
                "name": unit_name,
                "total": size,
                "low": low,
                "medium": med,
                "high": high,
                "hidden": False
            })

    output_data = {
        "personnel": personnel_list,
        "units": units_list[:5]
    }

    with open('data.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Successfully generated data.json for GitHub Pages!")

if __name__ == "__main__":
    main()