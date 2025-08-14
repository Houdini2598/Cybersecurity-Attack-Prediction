import io
import os
import json
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# PCAP parsing is optional; CSV works even if this import fails.
try:
    from feature_extraction import pcap_to_flows
except Exception:
    pcap_to_flows = None

st.set_page_config(page_title="Wireshark Risk Prediction System", layout="wide")

# ========= Config (must match your training) =========
NUMERIC = [
    "duration", "packets", "bytes", "bytes_per_packet",
    "packets_per_second", "iat_mean", "iat_std",
    "tcp_syn", "tcp_ack", "tcp_rst", "tcp_fin",
    "src_port", "dst_port",
]
CATEGORICAL = ["protocol"]

SEVERITY = {
    "DDoS": 1.00, "DoS": 0.90, "Exfiltration": 0.95, "BruteForce": 0.70,
    "PortScan": 0.50, "WebAttack": 0.75, "Botnet": 0.85, "Normal": 0.10,
    # Add dataset-specific labels if your model has them:
    "Exploits": 0.90, "Fuzzers": 0.50, "Generic": 0.80, "Reconnaissance": 0.60,
    "Shellcode": 0.95, "Worms": 0.95, "Analysis": 0.70, "Backdoors": 0.95,
}

SENSITIVE_PORTS = {22, 23, 25, 53, 80, 110, 143, 443, 445, 465, 587, 993, 995, 3306, 3389, 8080}
HIGH_BYTES = 10_000_000  # 10 MB

# ========= Sidebar: where to load artifacts from =========
st.sidebar.header("Model Artifacts")
art_dir = st.sidebar.text_input("Artifacts directory", value=".", help="Folder with your pickles/json")
basename = st.sidebar.text_input("Basename (if used)", value="", help="e.g., 'unsw' if files are unsw_*.pkl")

def _pick_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Tried paths: {paths}")

@st.cache_resource(show_spinner=False)
def load_artifacts(art_dir: str, basename: str):
    # Try several common filenames and locations
    pre_path = _pick_first([
        os.path.join(art_dir, f"{basename}_preprocessor.pkl") if basename else "",
        os.path.join(art_dir, "preprocessor.pkl"),
        "preprocessor.pkl",
    ])
    clf_path = _pick_first([
        os.path.join(art_dir, f"{basename}_clf.pkl") if basename else "",
        os.path.join(art_dir, "model.pkl"),
        "model.pkl",
    ])

    pre = joblib.load(pre_path)
    clf = joblib.load(clf_path)

    iso, meta = None, None
    try:
        iso_path = _pick_first([
            os.path.join(art_dir, f"{basename}_iso.pkl") if basename else "",
            os.path.join(art_dir, "iso_model.pkl"),
            "iso_model.pkl",
        ])
        meta_path = _pick_first([
            os.path.join(art_dir, f"{basename}_iso_meta.json") if basename else "",
            os.path.join(art_dir, "iso_meta.json"),
            "iso_meta.json",
        ])
        iso = joblib.load(iso_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        pass

    return pre, clf, iso, meta, pre_path, clf_path

@st.cache_resource(show_spinner=False)
def build_shap_explainer(cache_key: str, _clf, _Xt_full):
    """
    cache_key: hashable token (e.g., model path + n_features)
    _clf, _Xt_full: leading underscore => Streamlit ignores them for hashing.
    Builds a TreeExplainer with a small background from the SAME transformed space.
    Uses probability space + interventional perturbation.
    """
    import shap

    n = _Xt_full.shape[0]
    if n > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=200, replace=False)
        bg = _Xt_full[idx]
    else:
        bg = _Xt_full

    if hasattr(bg, "toarray"):
        bg = bg.toarray()
    bg = np.asarray(bg, dtype=np.float64)

    return shap.TreeExplainer(
        _clf,
        data=bg,
        model_output="probability",
        feature_perturbation="interventional",
    )

def get_feature_names_safe(preprocessor, n_features, raw_feature_order):
    """
    Returns a feature-name array of length n_features.
    Tries several ways; pads or trims to match exactly.
    """
    names = None
    try:
        names = preprocessor.get_feature_names_out(raw_feature_order)
    except Exception:
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = None

    if names is None:
        names = [f"f{i}" for i in range(n_features)]
    else:
        names = list(names)

    # Enforce exact length match with transformed matrix
    if len(names) > n_features:
        names = names[:n_features]
    elif len(names) < n_features:
        names += [f"f{i}" for i in range(len(names), n_features)]

    return np.array(names, dtype=object)

def heuristic_bonus(row: pd.Series) -> float:
    bonus = 0.0
    try:
        dport = int(row.get("dst_port", 0))
    except Exception:
        dport = 0
    if dport in SENSITIVE_PORTS:
        bonus += 6
    if row.get("tcp_syn", 0) > 10 and row.get("tcp_ack", 0) == 0:
        bonus += 8
    if row.get("bytes", 0) > HIGH_BYTES:
        bonus += 6
    if row.get("packets_per_second", 0) > 500:
        bonus += 8
    return bonus

def class_risk(pred_label: str, proba_row: pd.Series) -> float:
    p = float(proba_row.get(pred_label, 0.0))
    sev = SEVERITY.get(pred_label, 0.6)
    return 100.0 * p * sev

def anomaly_risk(iso, iso_meta, Xt):
    if iso is None or iso_meta is None:
        zeros = np.zeros(Xt.shape[0])
        return zeros, zeros
    raw = -iso.decision_function(Xt)  # higher = more anomalous
    thr = max(float(iso_meta.get("iso_threshold", 1e-6)), 1e-6)
    risk = 100.0 * np.clip(raw / thr, 0, 1.5)
    return risk, raw

# ========= UI =========
st.title("Network Risk Prediction System")

status = st.empty()
try:
    preprocessor, clf, iso, iso_meta, pre_path, clf_path = load_artifacts(art_dir, basename)
    status.success(f"✅ Loaded\n- Preprocessor: `{pre_path}`\n- Classifier: `{clf_path}`")
except Exception as e:
    status.error(f"⚠️ Load error: {e}")
    st.stop()

up = st.file_uploader("Upload PCAP/PCAPNG or CSV", type=["pcap", "pcapng", "csv"])

if up:
    with st.spinner("Parsing file…"):
        if up.name.lower().endswith(".csv"):
            flows = pd.read_csv(up)
        else:
            if pcap_to_flows is None:
                st.error("PCAP parsing not available. Add feature_extraction.py or upload a CSV.")
                st.stop()
            data = up.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tf:
                tf.write(data)
                tmp_path = tf.name
            try:
                flows = pcap_to_flows(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    st.subheader("Parsed flows")
    st.write(f"{len(flows)} flows found")
    st.dataframe(flows.head(100), use_container_width=True)

    # Validate required columns
    needed = NUMERIC + CATEGORICAL
    missing = [c for c in needed if c not in flows.columns]
    if missing:
        st.error(f"Your data is missing required columns: {missing}")
        st.stop()

    # Predict
    X = flows[needed].copy()
    Xt = preprocessor.transform(X)  # may be sparse

    probs = clf.predict_proba(Xt)
    classes = list(clf.classes_)
    pred_idx = np.argmax(probs, axis=1)
    preds = np.array(classes, dtype=object)[pred_idx]
    proba_df = pd.DataFrame(probs, columns=classes)

    # Anomaly risk (optional)
    a_risk, a_raw = anomaly_risk(iso, iso_meta, Xt)

    # Risk = 70% class risk + 30% anomaly + heuristic bonuses
    class_risks = [class_risk(label, proba_df.iloc[i]) for i, label in enumerate(preds)]
    base = np.array(class_risks)
    heur = np.array([heuristic_bonus(flows.iloc[i]) for i in range(len(flows))])
    combined = 0.7 * base + 0.3 * np.clip(a_risk, 0, 100) + heur
    risk_score = np.clip(combined, 0, 100)

    # Output table
    out = flows.copy()
    out["predicted_label"] = preds
    out["risk_score"] = np.round(risk_score, 1)
    out["anomaly_score"] = np.round(a_raw, 4)
    out["class_risk"] = np.round(class_risks, 1)
    out["anomaly_risk"] = np.round(np.clip(a_risk, 0, 100), 1)

    def _topk(row, k=3):
        s = row.sort_values(ascending=False).head(k)
        return "; ".join([f"{i}:{s[i]:.2f}" for i in s.index])

    out["top3_probs"] = proba_df.apply(_topk, axis=1)

    # Summary
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total flows", len(out))
    c2.metric("High risk (≥70)", int((out["risk_score"] >= 70).sum()))
    c3.metric("Medium (40–69)", int(((out["risk_score"] >= 40) & (out["risk_score"] < 70)).sum()))
    if iso is not None and iso_meta is not None:
        anomalies = int((out["anomaly_score"] > iso_meta.get("iso_threshold", 9e9)).sum())
    else:
        anomalies = 0
    c4.metric("Anomalies (raw > thr)", anomalies)

    st.bar_chart(out["predicted_label"].value_counts())

    st.subheader("Top risky flows")
    st.dataframe(out.sort_values("risk_score", ascending=False).head(50), use_container_width=True)

    # ========= Explainability (SHAP) =========
    st.subheader("Explain a single flow (SHAP)")
    if len(out) == 0:
        st.info("No flows to explain yet.")
    else:
        max_idx = len(out) - 1
        idx = st.number_input("Row index to explain", min_value=0, max_value=max_idx, value=0, step=1, key="shap_idx")
        if st.button("Compute SHAP for selected row"):
            with st.spinner("Computing SHAP…"):
                try:
                    import shap
                    import matplotlib.pyplot as plt

                    # Build (cached) explainer with background from the CURRENT Xt
                    cache_key = f"{clf_path}|nfeat={Xt.shape[1]}"
                    explainer = build_shap_explainer(cache_key, clf, Xt)

                    # Transform ONLY this row → dense float64
                    row_X = X.iloc[[int(idx)]]
                    Xt_row = preprocessor.transform(row_X)
                    if hasattr(Xt_row, "toarray"):
                        Xt_row = Xt_row.toarray()
                    Xt_row = np.asarray(Xt_row, dtype=np.float64)

                    # Safe, aligned feature names (length matches Xt_row.shape[1])
                    feat_names = get_feature_names_safe(preprocessor, Xt_row.shape[1], NUMERIC + CATEGORICAL)

                    classes = list(clf.classes_)
                    pred_label = str(out.iloc[int(idx)]["predicted_label"])
                    if pred_label in classes:
                        cls_index = classes.index(pred_label)
                    else:
                        cls_index = int(np.argmax(clf.predict_proba(Xt_row), axis=1)[0])

                    # New SHAP API (probability, interventional). Disable strict additivity check.
                    try:
                        exp = explainer(Xt_row, check_additivity=False)
                        vals = exp.values
                        if getattr(vals, "ndim", 0) == 3:
                            shap_vals = vals[0, cls_index, :]
                        else:
                            shap_vals = vals[0]
                    except TypeError:
                        # Older SHAP fallback
                        try:
                            sv = explainer.shap_values(Xt_row, check_additivity=False)
                        except TypeError:
                            sv = explainer.shap_values(Xt_row)
                        shap_vals = sv[cls_index][0] if isinstance(sv, list) else sv[0]

                    shap_vals = np.asarray(shap_vals).reshape(-1)  # ensure 1D

                    # FINAL GUARD: if still mismatched, align by trunc/pad
                    if shap_vals.shape[0] != feat_names.shape[0]:
                        n = min(shap_vals.shape[0], feat_names.shape[0])
                        shap_vals = shap_vals[:n]
                        feat_names = feat_names[:n]

                    contrib = (
                        pd.DataFrame({"feature": feat_names, "shap_value": shap_vals})
                        .assign(abs=lambda d: d["shap_value"].abs())
                        .sort_values("abs", ascending=False)
                        .head(15)
                        .drop(columns=["abs"])
                    )

                    st.write(f"**Predicted class:** {pred_label}")
                    st.dataframe(contrib, use_container_width=True)

                    fig, ax = plt.subplots()
                    ax.barh(contrib["feature"][::-1], contrib["shap_value"][::-1])
                    ax.set_xlabel("SHAP value (impact on model output)")
                    ax.set_ylabel("Feature")
                    ax.set_title("Top SHAP contributions")
                    st.pyplot(fig)

                    st.caption("Explaining class probability using a background sample (interventional).")
                except Exception as e:
                    st.error(f"SHAP failed: {e}. Showing model feature importances instead.")
                    try:
                        # Use safe names here too
                        safe_names = get_feature_names_safe(preprocessor, Xt.shape[1], NUMERIC + CATEGORICAL)
                        importances = getattr(clf, "feature_importances_", None)
                        if importances is not None:
                            # Align lengths just in case
                            n = min(len(importances), len(safe_names))
                            imp = (pd.DataFrame({"feature": safe_names[:n], "importance": importances[:n]})
                                   .sort_values("importance", ascending=False)
                                   .head(20))
                            st.dataframe(imp, use_container_width=True)
                    except Exception:
                        st.info("Could not compute importances.")

    # Download
    st.download_button(
        "Download results CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="wireshark_risk_results.csv",
        mime="text/csv",
    )

else:
    st.info("No file uploaded yet. Try a small PCAP first (or a flows CSV).")