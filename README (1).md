# Kavach pipeline (synthetic data demo)

Team Ace, Smart India Hackathon 2026, problem statement SIH26186
(AI-Based Predictive Personnel Stress and Welfare Monitoring System for Uniformed Forces).

This repo is a small, runnable version of the pipeline described in our idea deck. It uses **made-up data only**.
No real personnel data is used anywhere.

## What it does

| Step | What the script does | Deck reference |
|---|---|---|
| 1. Collect | Generates synthetic 8-week duty rosters, leave and transfer history, sleep check-ins and an optional self-check score (about 60% opt in) | Slide 3, steps 1 and 2 |
| 2. Analyse | Builds features: shortest rest gap, share of rest gaps under 8 h, shifts over 12 h per week, night shifts per week, days since leave, transfers in 24 months | Slide 3, step 3 |
| 3. Predict | Fixed rules (safety net) plus an XGBoost model. A person is flagged if either one fires | Slide 3, step 4 |
| 4. Explain | SHAP gives the top reasons behind each flag in plain language, with pseudonymized IDs | Slide 3, step 5 |
| 5. Audit | Compares flag rate, recall and precision across rank, region and language groups | Slide 4, "Bias" row |
| 6. Privacy filter | Commander view hides any unit under 10 people | Slides 3 and 4 |

## Run it

```bash
pip install -r requirements.txt
python pipeline.py
```

It takes under a minute and writes three files to `outputs/`: `example_flags.csv`, `fairness_report.csv`, `commander_unit_view.csv`.
The run is deterministic (fixed random seed). `sample_output.txt` shows what a run prints.

## What this proves, and what it does not

- It shows the pipeline works end to end: data in, features, rules and model, plain-language reasons, audit, privacy filter.
- **It does not show real-world accuracy.** The labels come from our own generator, so a good score only means the model learned our generator. The script prints an AUC as a sanity check and says not to quote it. Real accuracy will be claimed only after a clinician-checked pilot on real data, as stated in the deck.
- The model never receives rank, region or language as a feature. Those columns are used only for the audit.

## Things built in on purpose

- **Under-reporting demo:** in the synthetic data, the group `Lang-3` under-reports on the self-check by 25%. This is there so the fairness audit has something to find. Across seven random seeds we tried, recall for that group was lower every time (by about 0.05 to 0.23), and the script printed a warning when the gap passed 0.10, which it did in six of the seven. It illustrates the "hiding stress in self-reports" and "bias" risks on slide 4.
- **Rules as a safety net:** a few people are flagged by the rules but missed by the model. The thresholds are illustrative and would need review by welfare and clinical experts.
- **Optional self-check:** about 40% of people have no self-check score. XGBoost handles the missing values, so the roster signals still work.

## Limits

- Synthetic data cannot capture real behaviour, culture or reporting patterns.
- Pseudonymization here is a salted hash for illustration. Production needs a managed key and stronger controls (see slide 3, security layers).
- SHAP reasons explain the model, not causes. They are prompts for a private conversation by a welfare officer, never a diagnosis.

## Next steps (pilot)

1. Agree features, thresholds and consent rules with a partner force, welfare officers and clinicians.
2. Collect real data under ethics approval and the DPDP Act, 2023, with clinician-confirmed labels.
3. Re-run the fairness audit on real data before any use.
