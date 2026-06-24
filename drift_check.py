import pandas as pd
import json
import sys
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

# ── Config ─────────────────────────────────────────────────────────────────────
DRIFT_THRESHOLD  = 0.5
REFERENCE_PATH   = "data/reference_sample.csv"
PRODUCTION_PATH  = "data/production_sample.csv"
CONFIG_PATH      = "model_config.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)

FEATURE_COLS = config["feature_cols"]

# ── Load data ──────────────────────────────────────────────────────────────────
reference  = pd.read_csv(REFERENCE_PATH)
production = pd.read_csv(PRODUCTION_PATH)

ref_sample  = reference.sample(n=5000, random_state=42).reset_index(drop=True)
prod_sample = production.sample(n=5000, random_state=42).reset_index(drop=True)

# ── Run report ─────────────────────────────────────────────────────────────────
data_definition = DataDefinition(numerical_columns=FEATURE_COLS)
ref_dataset     = Dataset.from_pandas(ref_sample,  data_definition=data_definition)
prod_dataset    = Dataset.from_pandas(prod_sample, data_definition=data_definition)

report  = Report(metrics=[DataDriftPreset()])
my_eval = report.run(reference_data=ref_dataset, current_data=prod_dataset)
my_eval.save_html("evidently_drift_report.html")

# ── Parse drift share ──────────────────────────────────────────────────────────
result      = my_eval.dict()
first_metric = result["metrics"][0]
drift_share  = first_metric["value"]["share"]
drift_count  = int(first_metric["value"]["count"])
total        = len(FEATURE_COLS)

print(f"Drift report complete.")
print(f"Drifted features: {drift_count}/{total} ({drift_share*100:.1f}%)")
print(f"Threshold: {DRIFT_THRESHOLD*100:.0f}%")

if drift_share > DRIFT_THRESHOLD:
    print(f"DRIFT ALERT: {drift_share*100:.1f}% exceeds threshold {DRIFT_THRESHOLD*100:.0f}%")
    print("Recommendation: retrain model on recent data before next deployment.")
    sys.exit(1)
else:
    print(f"Drift within acceptable range. Model deployment approved.")
    sys.exit(0)
