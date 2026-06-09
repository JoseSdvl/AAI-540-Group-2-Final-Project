# AAI-540 Group 2 — 30-Day Hospital Readmission Risk

**Business name:** ClearPath Health Analytics
**Authors:** Jose Sandoval, Manikanta Katuri, Michael Domingo
**Course:** AAI-540 — Machine Learning Operations

End-to-end MLOps project that predicts the 30-day hospital-readmission risk of
adult Medicare patients at the point of discharge, using the CMS DE-SynPUF
synthetic claims dataset. The system is designed to run on **Amazon SageMaker**
and includes feature engineering, training, evaluation, real-time deployment,
**Model Monitor + CloudWatch + SNS alerting**, and **GitHub Actions CI/CD**.

---

## 1. What this repo contains

```
aai540-group2-final-project/
├── notebooks/
│   └── AAI-540-Group02-project.ipynb          ← main SageMaker notebook
├── src/readmit/
│   ├── data/         data ingest + 30-day readmission labeling
│   ├── features/     feature engineering (bucketing, encoding, transforms)
│   ├── models/       train / evaluate / inference (XGBoost + LR baseline)
│   ├── monitoring/   Model Monitor schedule + CloudWatch alarms + SNS
│   └── pipeline/     SageMaker Pipelines orchestration
├── tests/            pytest unit tests (features, labeling, contracts, metrics)
├── infrastructure/   CloudWatch alarm + SNS topic JSON templates
├── .github/workflows/ ci.yml (lint + tests) and cd.yml (deploy via pipeline)
├── requirements.txt
├── setup.py
└── README.md
```

## 2. Data

| Layer    | Bucket (example)                       | Format  | Purpose                          |
|----------|----------------------------------------|---------|----------------------------------|
| Raw      | `s3://cms-readmit-raw/`                | CSV     | Original DE-SynPUF files         |
| Curated  | `s3://cms-readmit-curated/`            | Parquet | Cleaned encounter records        |
| Features | `s3://cms-readmit-features/`           | Parquet | Model-ready feature matrix       |
| Model    | `s3://cms-readmit-model-artifacts/`    | .tar.gz | Trained model + metadata         |
| Monitor  | `s3://cms-readmit-monitoring/`         | JSON    | Baselines, capture, drift reports|

The ingest module (`src/data/ingest.py`) supports two modes:
- **`source=s3`** — reads CMS DE-SynPUF from the AWS Open Data Registry
- **`source=synthetic`** — generates a deterministic synthetic claims dataset
  (used by CI and for local development without S3 access)

## 3. Train / Validation / Test split

Per project requirements this submission uses **40 / 30 / 30**
(train / test / validation) with a **patient-grouped, time-aware** split so a
single beneficiary's encounters never leak across folds.

## 4. Model

- **Baseline:** scikit-learn `LogisticRegression` (interpretability + calibration)
- **Primary:** `XGBoost` (handles tabular, imbalanced, nonlinear)
- **Imbalance:** `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (LR)
- **Primary metric:** AUC-ROC (target ≥ 0.75)
- **Secondary:** PR-AUC, Recall@K, Precision, F1, Brier score, calibration curve
- **Subgroup evaluation:** age band, sex, primary diagnosis chapter

## 5. Deployment

- Real-time SageMaker endpoint (`ml.m5.large`) for discharge-time scoring
- Optional Batch Transform (`ml.m5.xlarge`) for retrospective scoring
- Data capture enabled on the endpoint for Model Monitor

## 6. Monitoring & Alerts

- **Model Monitor** baseline + hourly schedule against captured traffic
- **CloudWatch alarms** on: endpoint latency (p95), 4xx/5xx errors,
  CPU/Memory, and a **custom `ReadmitModel/AUC` metric** published by the
  scheduled eval Lambda from `src/monitoring/alerts.py`
- **SNS topic** `readmit-model-alerts` fans out email + PagerDuty/Slack

## 7. CI/CD (GitHub Actions)

- `.github/workflows/ci.yml`   — ruff + black --check + pytest + coverage
- `.github/workflows/cd.yml`   — on push to `main`, assumes an AWS OIDC role
  and runs the SageMaker Pipeline (`src/pipeline/sagemaker_pipeline.py`) which
  trains, evaluates, registers in the Model Registry (manual approval), then
  deploys to staging → production with a post-deploy smoke test.

## 8. How to reproduce

The notebook is the single source of truth for the demo — it drives the same
code paths that CI/CD runs in production. Two paths are supported:

### A. Local (development / unit tests)

```powershell
git clone https://github.com/<your-org>/aai540-group2-final-project.git
cd aai540-group2-final-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .                       # install the readmit package in editable mode

pytest -q                              # runs against the synthetic generator
```

Local mode uses the deterministic synthetic generator
(`DATA_SOURCE='synthetic'` in the notebook), so no AWS credentials are needed.

### B. SageMaker Studio (full end-to-end demo)

1. **Clone** the repo into a SageMaker Studio user directory (terminal):
   ```bash
   git clone https://github.com/<your-org>/aai540-group2-final-project.git
   cd aai540-group2-final-project
   pip install -e .
   ```
2. **Open** `notebooks/AAI-540-Group02-project.ipynb` with the
   `Data Science 3.0` (or any Python 3.10+) kernel.
3. **Configure** the first cell:
   - `DATA_SOURCE = 'cms-open'`  (curates the public CMS DE-SynPUF data)
   - `CMS_TIER    = '100k'`      (use `'1k'` for a 60-second smoke run)
   - `BUCKET = None`             (auto-uses `sagemaker.Session().default_bucket()`)
4. **Run All.** First execution curates `encounters.parquet` to
   `s3://{BUCKET}/readmit/curated/` (a few minutes); every subsequent run
   reads the cached parquet.
5. **Clean up.** Execute the final cell to delete the endpoint when you're
   done — leaving `ml.m5.large` + Model Monitor running costs ~$50/day.

Required IAM permissions for the SageMaker execution role: read access to
the user bucket, plus the managed policies `AmazonSageMakerFullAccess` and
`CloudWatchFullAccess`. Anonymous reads from `s3://synpuf-omop/` need no
credentials.

## 9. Compliance / scope

The project uses **synthetic** CMS data (DE-SynPUF), so no PHI/PII is stored
or processed. The system is a clinical-decision-support prototype and is
**not** a medical device. See the Model Card cell at the end of the notebook
for intended-use, limitations, and retraining cadence.
