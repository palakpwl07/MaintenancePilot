# MaintenancePilot

Predictive maintenance pipeline for industrial compressors. Predicts failure risk 48 hours in advance using real operational sensor data. Built to demonstrate production ML engineering — the model is not the product, the pipeline is.

**Live API:** https://pp263-maintenancepilot.hf.space/docs  
**GitHub:** https://github.com/palakpwl07/MaintenancePilot

---

## The Problem

Unplanned compressor failures cost industrial facilities hundreds of thousands per incident — unplanned downtime, emergency repair, cascading equipment damage. Most facilities still run scheduled maintenance: replace parts on a calendar, not on condition.

This pipeline detects the compressor entering a compensatory overwork state — elevated sustained pressure, reduced cycling variance, near-zero tower discharge — up to 48 hours before a maintenance team would otherwise notice. The output is an actionable intervention window, not an emergency.

---

## Pipeline Architecture

```
Kaggle API
    └── Polars (lazy ingestion, 1.5M rows)
            └── Feature Engineering (41 features, rolling windows)
                    └── MLflow (experiment tracking + model registry)
                            └── LightGBM Champion (CV AP 0.973, Recall 0.995)
                                    └── FastAPI (3 endpoints, typed schemas)
                                            └── Docker (containerised, one-command deploy)
                                                    └── Evidently (drift monitoring)
                                                            └── GitHub Actions (quality gate, fails on drift)
```

---

## Dataset

**MetroPT-3** — real operational data from the Air Production Unit (APU) compressor of a Portuguese metro train. February to September 2020. 1,516,948 rows at 10-second sampling intervals. 7 analogue sensors (pressure, temperature, motor current) + 8 digital sensors (binary electrical signals). 4 documented air leak failures with maintenance timestamps.

The dataset is unlabeled. Labels were constructed by backpropagating from failure timestamps: any window within 48 hours before a failure start = label 1 (pre-failure). Everything else = label 0 (normal).

Source: [Kaggle — joebeachcapital/metropt-3-dataset](https://www.kaggle.com/datasets/joebeachcapital/metropt-3-dataset)

---

## Feature Engineering

41 features constructed from 15 raw sensors. The failure signature is not a threshold breach — it is a change in behavioural pattern. The compressor stops cycling and runs continuously. Rolling statistics capture this where raw values cannot.

| Feature Group | Sensors | Statistics | Rationale |
|---|---|---|---|
| Analogue — mean + std | TP2, Motor_current, DV_pressure | 30min, 2h windows | Level shift AND variance drop are both diagnostic |
| Analogue — mean only | Oil_temperature, TP3, Reservoirs | 30min, 2h windows | Level shift is the primary signal |
| Rate of change | TP2, Motor_current, DV_pressure | 1-step diff | Catches the transition moment into failure state |
| Digital — rolling sum | LPS, COMP | 30min, 2h windows | Count of activations; "LPS fired 12x in 30 min" is operationally meaningful |
| Digital — time since event | LPS | Rows since last activation | Recency matters; sudden activation after a long quiet period is alarming |

**Window sizes:** 30 minutes (180 rows) captures acute changes. 2 hours (720 rows) captures gradual drift. Both are needed.

**Key observation:** `Motor_current_std` and `TP2_std` are stronger discriminators than their means. Reduced variance means the compressor stopped cycling — it is stuck in sustained overwork. A compressor working normally looks noisy. A compressor about to fail looks flat.

---

## Modelling

### Split Strategy

Time-based, not random. Shuffling a time series creates leakage — the model trains on future sensor readings. 

- **Train:** February 1 → June 27 (contains F1, F2, F3)
- **Validation:** June 28 → July 13
- **Test:** July 14 → September 1 (contains F4, never seen during training or tuning)

### Experiment Comparison

Three runs logged to MLflow. LightGBM is the clear champion.

| Run | Model | CV Mean AP | CV Mean Recall | Notes |
|---|---|---|---|---|
| xgboost_baseline | XGBoost | 0.867 | 0.990 | Default params — establishes floor |
| xgboost_tuned | XGBoost | 0.726 | 0.932 | Optuna tuning — AP dropped vs baseline |
| **lightgbm_tuned** | **LightGBM** | **0.973** | **0.995** | **Champion — registered in MLflow registry** |

Notable: Optuna tuning hurt XGBoost because it optimised on a single validation period AP (0.074) rather than CV AP. Hyperparameter search is only as good as the evaluation signal it optimises. LightGBM's early stopping gave better generalisation across folds.

### Evaluation: Temporal Cross-Validation

Each fold trains on historical failures and predicts an unseen future failure. This is harder than random CV and gives an honest estimate of how the model performs on a new failure event in production.

| Fold | Test Failure | AP | Precision | Recall | Missed Failures |
|---|---|---|---|---|---|
| 1 | F2 (May 29-30) | 0.729 | 0.315 | 0.997 | 49 of 15,870 |
| 2 | F3 (Jun 5-7) | 0.781 | 0.583 | 0.956 | 733 of 16,692 |
| 3 | F4 (Jul 15) | 0.540 | 0.231 | 1.000 | 0 of 12,124 |
| **Mean** | | **0.683** | **0.376** | **0.984** | |

### Threshold Selection

Default threshold of 0.5 is wrong for this problem. A missed failure costs significantly more than an unnecessary inspection. Threshold selected by minimising:

```
Cost = FN × 10 + FP × 1
```

The fn_cost=10 reflects that an undetected compressor failure causes unplanned downtime and emergency repair — conservatively 10x the cost of a false alarm that triggers an unnecessary maintenance check. The threshold minimising this cost function on the validation set is used at inference time, and is configurable per-request via the API without redeployment.

### Class Imbalance

4% positive class (pre-failure windows). Handled at the algorithm level: `class_weight='balanced'` for LightGBM, `scale_pos_weight=20` for XGBoost. SMOTE was explicitly rejected — synthetic oversampling of time series data introduces temporal leakage.

---

## API

Live at **https://pp263-maintenancepilot.hf.space/docs**

Three endpoints:

**`GET /health`** — liveness check. Returns model version, type, threshold, and CV performance metrics. Operations team can verify model quality without opening MLflow.

**`POST /predict`** — single window failure risk prediction. Input: 41 engineered features. Output: risk score, HIGH_RISK/NORMAL prediction, threshold used, model version, and plain-English interpretation for the maintenance engineer.

**`GET /model-info`** — full model card. Training period, failure events, feature list, CV metrics.

Example predict response:
```json
{
  "failure_risk_score": 0.847,
  "prediction": "HIGH_RISK",
  "threshold_used": 0.5,
  "model_version": 2,
  "model_type": "lightgbm",
  "interpretation": "Compressor shows pre-failure signature. Recommend inspection within 48 hours. Risk score 0.847 exceeds threshold 0.500."
}
```

---

## Docker

One-command deployment:

```bash
docker compose up --build
```

API available at `http://localhost:8000/docs`

The model is baked into the image — no external dependencies at runtime. Updating the production model requires registering a new version in MLflow and restarting the container. No code changes.

---

## Drift Monitoring

**Evidently AI** compares incoming sensor distributions against the training reference. Tested on the July-August deployment window vs the February-June training period.

```
Drifted features: 32/41 (78.0%)
Threshold: 50%
DRIFT ALERT: 78.0% exceeds threshold 50%
Recommendation: retrain model on recent data before next deployment.
```

78% of features drifted — primarily Oil_temperature (seasonal heat rise) and TP2/Motor_current rolling statistics (compressor cycling patterns change with ambient temperature). This is physically expected and demonstrates why drift monitoring is non-negotiable in production.

The drift check script exits with code 1 on threshold breach. GitHub Actions runs it on every push to main and every Monday morning. A push that would deploy a drifting model fails the CI pipeline before reaching production.

A model that degrades silently is worse than no model.

---

## Repository Structure

```
MaintenancePilot/
├── app/
│   ├── main.py          # FastAPI app — 3 endpoints with typed schemas
│   ├── model.py         # Model loading and inference
│   └── schemas.py       # Pydantic request/response models
├── data/
│   ├── reference_sample.csv   # Training distribution (5K rows)
│   └── production_sample.csv  # Deployment distribution (5K rows)
├── .github/
│   └── workflows/
│       └── drift_check.yml    # CI pipeline — fails on drift > 50%
├── Dockerfile
├── docker-compose.yml
├── drift_check.py       # Evidently drift report + quality gate
├── model.pkl            # LightGBM champion (joblib)
├── model_config.json    # Model metadata, threshold, feature list
└── requirements.txt
```

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Dataset | MetroPT-3 over NASA C-MAPSS | Real operational data, unlabeled, mixed sensor types. Domain breadth. |
| Problem framing | Failure prediction (binary) over RUL | Clean labels, clean evaluation. RUL without ground truth requires arbitrary assumptions. |
| Prediction horizon | 48 hours | Sufficient operational lead time. Shorter reduces false alarms but cuts response time. |
| Split strategy | Temporal CV over single split | Single split showed distribution shift artifact. CV across failure events gives honest generalisation. |
| Imbalance handling | Algorithm-level (scale_pos_weight) | SMOTE on time series creates temporal leakage. |
| Threshold | Business cost function (FN×10 + FP×1) | F1 treats FP and FN equally. Wrong for predictive maintenance. |
| Data ingestion | Polars lazy evaluation | Prevents truncation of the full dataset. Excel and pandas both cap at ~1M rows. |
| Champion model | LightGBM over XGBoost | CV AP 0.973 vs 0.867 baseline, 0.726 tuned. Early stopping generalises better across failure folds. |

---

## Stack

`LightGBM` · `XGBoost` · `Optuna` · `MLflow` · `FastAPI` · `Docker` · `Evidently AI` · `GitHub Actions` · `Polars` · `scikit-learn` · `joblib` · `Pydantic`
