# SentryNet

SentryNet is an unsupervised anomaly detector for network flow records. It learns what
**normal** traffic looks like — and only normal traffic — then flags flows that deviate from
that pattern as suspicious. No attack examples are used to train any model; one attack type
(PortScan) is withheld from every stage except the final test, to check that detection
generalises rather than memorises.

## What It Does

```
CSV network flows → derived features → preprocessing → anomaly detector → anomaly score → Normal / Suspicious
```

Each row is scored, compared against a frozen decision threshold, and labelled **Normal** or
**Suspicious**.

## Models

- **Statistical baseline** — transparent percentile / z-score rule; every alert names the
  feature that triggered it.
- **Isolation Forest** — fast tree-based detector.
- **One-Class SVM (RBF kernel)** — learns a boundary around normal traffic.

All three are trained on **Normal-only** data and reported separately (never ensembled), so
every alert is attributable to one specific method.

## Input Features

| column | meaning |
|---|---|
| `duration` | flow duration in seconds |
| `src_bytes` | bytes sent by the source |
| `dst_bytes` | bytes sent by the destination |
| `packet_count` | packets in the flow |
| `protocol` | `TCP` or `UDP` |
| `failed_logins` | failed login attempts observed in the flow |

## Quick Start — GitHub Codespaces

No local setup required.

1. Open this repository on GitHub.
2. Click **Code** → **Codespaces** → **Create codespace on main**.
3. Wait for the container to build (Python is installed and
   `pip install -r requirements.txt` runs automatically).
4. In the terminal, run:
   ```bash
   python app.py
   ```
5. A **Ports** notification appears for port **7860** — click **Open in Browser**
   (or open the **Ports** tab and click the globe icon next to `7860`).
6. Upload [`sample/sample_input.csv`](sample/sample_input.csv) and click **Score CSV**.

## Quick Start — Local Computer

Requires Python 3.12+.

```bash
git clone https://github.com/gitmuaad/SentryNet-anomaly-detection.git
cd SentryNet-anomaly-detection
python -m venv .venv
```

Activate the virtual environment — Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

## Try the Demo

The repository ships with trained models under `artifacts/`, so the app works immediately —
no training required.

Upload [`sample/sample_input.csv`](sample/sample_input.csv), pick a detector, and click
**Score CSV** to see anomaly scores, Normal/Suspicious decisions, and a score distribution
chart. The sample file contains only the six columns above (no label column).

## Reproduce Training

The trained artifacts in this repository were produced by the commands below, with a fixed
random seed for reproducibility. To reproduce them yourself:

1. Download the dataset — Kaggle, **"Cyber Security Attack Using Network Traffic"** by
   `juanschafle`, version 1 — and place it at:
   ```
   data/raw/cyber_attack_dataset_100000.csv
   ```
2. Clean the data and freeze the train/validation/test splits:
   ```bash
   python scripts/prepare_data.py
   ```
3. Train and tune all three detectors, and freeze the operating threshold:
   ```bash
   python scripts/train.py
   ```

Training fits models on **Normal rows only**; DDoS and BruteForce are used solely to tune
hyperparameters and the decision threshold on a held-out validation set. PortScan never
participates in training or tuning.

## Evaluate Models

```bash
python scripts/evaluate.py
```

Reports PR-AUC, precision, recall, F1, and per-class recall (including PortScan as the
unseen-attack score) on the held-out test set, plus false-positive counts in 1,000-row
windows, a 95%-Normal/5%-attack sensitivity scenario, a Population Stability Index (PSI)
drift check, and a synthetic evasion stress test.

Optionally, benchmark inference latency:

```bash
python scripts/benchmark_latency.py
```

## Project Structure

```
SentryNet-anomaly-detection/
  README.md
  requirements.txt
  .gitignore
  .devcontainer/devcontainer.json

  config/
    config.yaml

  src/sentrynet/
    __init__.py
    config.py
    data.py               # schema validation, deduplication, privacy review
    features.py            # derived ratio features, safe division
    preprocessing.py       # log1p, StandardScaler, OneHotEncoder
    splits.py               # Normal-only train / validation / test protocol
    baseline.py             # statistical baseline detector
    isolation_forest.py     # Isolation Forest detector
    one_class_svm.py        # One-Class SVM detector
    evaluation.py            # PR-AUC, thresholds, 1,000-row FP windows
    sensitivity.py           # 95/5 prevalence sensitivity scenario
    monitoring.py            # PSI drift monitoring
    evasion.py                # synthetic evasion stress test
    inference.py              # shared scoring path used by app.py
    persistence.py            # artifact save/load

  scripts/
    prepare_data.py    # clean + freeze splits
    train.py             # fit + tune + freeze threshold
    evaluate.py           # full evaluation report
    benchmark_latency.py  # inference latency benchmark

  app.py                # Gradio review dashboard

  artifacts/random/     # trained models the app loads (no retraining)
  sample/sample_input.csv
```

## Dataset

Kaggle — **"Cyber Security Attack Using Network Traffic"** by `juanschafle`, version 1.
100,000 **synthetic** rows across four classes: Normal, DDoS, BruteForce, PortScan. The raw
dataset is not bundled in this repository; see [Reproduce Training](#reproduce-training) for
how to obtain it.

## Limitations

- The dataset is **synthetic** — results are not claimed to transfer to production traffic.
- No timestamps exist, so there is no temporal validation and no real "alerts per hour/day"
  figure — reported windows are blocks of 1,000 rows, and any daily-volume figure is a
  labelled, user-configurable scenario, not a measurement.
- The synthetic classes occupy unusually separable numeric ranges, which inflates detection
  scores relative to real-world traffic.
- This is a **prototype review tool**, not a production intrusion detection system: no live
  packet capture, no automated retraining, no SIEM integration.
