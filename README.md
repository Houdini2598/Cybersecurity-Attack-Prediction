Here’s a complete `README.md` you can drop into your repo.

```markdown
Explainable Network Risk Prediction (PCAP/CSV → Flow IDS + Risk Score)

An end-to-end, analyst-friendly pipeline that:
- parses PCAP/CSV into per-flow records,
- predicts an attack family with a supervised model,
- scores anomaly with Isolation Forest,
- fuses signals into a 0–100 risk score, and
- explains each prediction with SHAP inside a Streamlit UI.

App entrypoint: project1.py (Streamlit) :contentReference[oaicite:0]{index=0}  
Training script: trainmodel.py (artifacts + metrics) :contentReference[oaicite:1]{index=1}

✨ Features

- Flow-based schema with timing/volume + TCP flag features (duration, packets, bytes, PPS, BPP, IAT mean/std, SYN/ACK/RST/FIN, ports, protocol). :contentReference[oaicite:2]{index=2}  
- Supervised classifier: Random Forest (`model.pkl`). :contentReference[oaicite:3]{index=3}  
- Unsupervised anomaly: Isolation Forest (`iso.pkl`) + learned threshold (`iso_meta.json`). :contentReference[oaicite:4]{index=4}  
- Transparent risk fusion: `0.7×class_risk + 0.3×anomaly_risk + heuristics` → 0–100. :contentReference[oaicite:5]{index=5}  
- Explainability: SHAP on the **transformed** design matrix with robust fallbacks. :contentReference[oaicite:6]{index=6}  
- One-click UI: Upload PCAP/PCAPNG or CSV, triage by risk, download results. :contentReference[oaicite:7]{index=7}

📦 What’s in this repo

├── project1.py               # Streamlit app (run the UI)  ← app entrypoint
├── trainmodel.py             # Train RF + ISO, save artifacts & metrics
├── data/
│   └── unsw\_flows\_labelled.csv   # Example training data (flows with labels)
├── artifacts/                # (created after training)
│   ├── unsw\_preprocessor.pkl
│   ├── unsw\_model.pkl
│   ├── unsw\_iso.pkl
│   ├── unsw\_iso\_meta.json
│   ├── metrics.json
│   ├── confusion\_matrix.csv
│   └── feature\_importances.csv
└── README.md
```

Note: The app expects **ports** to be present; the trainer can work without them if your CSV lacks `src_port`/`dst_port` (it auto-detects). :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

🚀 Quickstart

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install "scikit-learn==1.4.2" "pandas==2.2.1" "numpy==1.26.4" \
            "shap==0.44.1" "streamlit==1.32.0" "joblib==1.3.2"
```

### 2) Train (creates artifacts)

```bash
python trainmodel.py \
  --train_csv data/unsw_flows_labelled.csv \
  --out_dir artifacts --basename unsw --fast --seed 42
```

* Saves: `unsw_preprocessor.pkl`, `unsw_clf.pkl`, `unsw_iso.pkl`, `unsw_iso_meta.json`,
  `metrics.json`, `confusion_matrix.csv`, `feature_importances.csv`.&#x20;

### 3) Run the UI

```bash
streamlit run project1.py
```

* In the sidebar, set **Artifacts directory** to `artifacts/` and **Basename** to `unsw`.&#x20;

### 4) Upload data

* **CSV (flows)**: must match the schema below.
* **PCAP/PCAPNG**: requires `feature_extraction.py` (provide `pcap_to_flows(path)` that emits the same schema). If absent, upload a CSV.&#x20;

---

## 📐 Flow schema (required columns)

| Numeric                                    | Description                  |
| ------------------------------------------ | ---------------------------- |
| `duration`                                 | seconds                      |
| `packets`                                  | total packets                |
| `bytes`                                    | total bytes                  |
| `bytes_per_packet`                         | `bytes/max(packets,1)`       |
| `packets_per_second`                       | `packets/max(duration,1e-6)` |
| `iat_mean`, `iat_std`                      | inter-arrival stats          |
| `tcp_syn`, `tcp_ack`, `tcp_rst`, `tcp_fin` | TCP flag counts              |
| `src_port`, `dst_port`                     | transport ports              |

| Categorical | Notes                                        |
| ----------- | -------------------------------------------- |
| `protocol`  | `TCP`/`UDP`/`ICMP` (OneHot, unknown ignored) |

* For **training**, include `label` (e.g., Normal, DoS, Exploits, Fuzzers, Reconnaissance, …).&#x20;
* The app **requires** all numeric + `protocol` + ports.&#x20;

---

## 🧠 Models & artifacts

* **Preprocessor**: `ColumnTransformer(StandardScaler on numerics + OneHot on protocol)` → `preprocessor.pkl`. **Fit once**, reuse everywhere.&#x20;
* **Classifier**: `RandomForestClassifier(class_weight=balanced, n_estimators=300, max_depth=None, seed=42)` → `*_clf.pkl`. Grid search optional.&#x20;
* **Anomaly**: `IsolationForest(n_estimators=400, contamination=0.05, seed=42)` trained on **training Normal** only. Threshold = 95th percentile of training-normal scores → `*_iso.pkl` + `*_iso_meta.json`.&#x20;

---

## 📊 Risk scoring

**Per-flow final score (0–100):**

```
risk_score = clip(0.7*class_risk + 0.3*anomaly_risk + heuristics, 0, 100)
class_risk = 100 * P(predicted_class) * severity[predicted_class]
anomaly_risk = 100 * clip(raw_iso_score / iso_threshold, 0, 1.5)
```

**Heuristics** (bonuses): +6 sensitive `dst_port` ∈ {22,23,25,53,80,110,143,443,445,465,587,993,995,3306,3389,8080}; +8 SYN>10 & ACK=0; +6 bytes>10 MB; +8 PPS>500.&#x20;

Severity weights (excerpt): DoS 0.90, Backdoors 0.95, Exfiltration 0.95, Fuzzers 0.50, PortScan 0.50, Normal 0.10. (See code for full map.)&#x20;

---

## 🖥️ Using the Streamlit app

1. **Artifacts loaded** → green check appears.
2. **Upload** a PCAP/PCAPNG or flows CSV.
3. **Parsed flows** table appears + KPI counters (total, high/medium risk, anomalies).
4. **Top risky flows** → sort by `risk_score`.
5. **Explainability** → select a row, compute SHAP on the **transformed** row; fallback to feature importances if SHAP unavailable.&#x20;

Download your results: `wireshark_risk_results.csv`.&#x20;

---

## 📈 Training & metrics

The trainer saves:

* **`metrics.json`**: accuracy, macro/weighted precision/recall/F1, macro ROC–AUC (OVR if probs available), train time; per-class metrics.
* **`confusion_matrix.csv`**: labeled rows/cols.
* **`feature_importances.csv`**: aligned to transformed feature names.&#x20;

Use `--test_csv` to evaluate on an external test set, or `--fast` for a non–grid-searched baseline. See `python trainmodel.py -h`.&#x20;

---

## 🧩 PCAP support (optional)

Implement `feature_extraction.py` with:

```python
def pcap_to_flows(pcap_path) -> pandas.DataFrame:
    """Return DataFrame with the schema listed above."""
```

Place it next to `project1.py`. If missing, upload a CSV instead.&#x20;

---

## ✅ Reproducibility checklist

* Use the **same** `preprocessor.pkl` at train/val/app time.&#x20;
* Fix `--seed 42`.
* Ensure CSV columns match the **schema** above.
* Don’t compute stats on validation/test during fitting.
* Keep artifact **basename** consistent (e.g., `unsw_*`).&#x20;

---

## 🛠️ Troubleshooting

* **“Load error: model.pkl not found”** → Point the sidebar to `artifacts/` and set **Basename** (e.g., `unsw`). Artifacts must exist.&#x20;
* **“Missing required columns”** → Your CSV must include **all** numeric features, `protocol`, and **ports**.&#x20;
* **SHAP errors** → The app falls back to feature importances automatically; ensure features come from the **same preprocessor** and retry single-row SHAP.&#x20;
* **Few/No anomalies** → Isolation threshold is learned on Normal training flows (95th percentile). Tune `--contamination` and retrain if needed.&#x20;

---

## 📜 License

Add your chosen license (e.g., MIT) here.

## 🙌 Acknowledgments

UNSW-NB15 (flows table) used for supervised training; community tools that inspired this pipeline.
