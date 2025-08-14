# train_model.py — minimal, robust, correct ordering
import os, json, argparse, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Base features (ports are optional; we’ll include them only if present in CSV)
NUMERIC_BASE = [
    "duration","packets","bytes","bytes_per_packet",
    "packets_per_second","iat_mean","iat_std","tcp_syn","tcp_ack","tcp_rst","tcp_fin"
]
CATEGORICAL = ["protocol"]

def load_data(path):
    df = pd.read_csv(path)
    # Decide whether to use port columns
    numeric = NUMERIC_BASE + [c for c in ["src_port","dst_port"] if c in df.columns]
    need = numeric + CATEGORICAL + ["label"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Training file missing required columns: {missing}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    df["label"] = df["label"].astype(str)
    return df, numeric

def main(argv=None):
    import time, json
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", default="", help="Optional external test CSV")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--basename", default="unsw")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metrics_json", default="metrics.json")
    ap.add_argument("--cm_csv", default="confusion_matrix.csv")
    ap.add_argument("--imp_csv", default="feature_importances.csv")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- load train ---
    df, NUMERIC = load_data(args.train_csv)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    X_all = df[NUMERIC + CATEGORICAL].copy()
    y_all = df["label"].astype(str).copy()
    classes_sorted = sorted(y_all.unique().tolist())

    # --- build pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=args.seed
    )
    pipe = Pipeline([("prep", preprocessor), ("clf", rf)])

    # --- make a proper hold-out set (unless external test supplied) ---
    if args.test_csv:
        df_test, _NUMERIC2 = load_data(args.test_csv)
        X_train, y_train = X_all, y_all
        X_val = df_test[NUMERIC + CATEGORICAL].copy()
        y_val = df_test["label"].astype(str).copy()
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        (train_idx, val_idx) = next(sss.split(X_all, y_all))
        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_val,   y_val   = X_all.iloc[val_idx],   y_all.iloc[val_idx]

    # --- fit ---
    t0 = time.time()
    if args.fast:
        print("[INFO] FAST mode: fitting baseline RF…")
        pipe.fit(X_train, y_train)
        fitted_prep = pipe.named_steps["prep"]
        fitted_clf  = pipe.named_steps["clf"]
    else:
        print("[INFO] GridSearchCV starting…")
        grid = GridSearchCV(
            pipe,
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 20],
                "clf__min_samples_split": [2, 10],
            },
            cv=3, n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)
        print("[INFO] Best params:", grid.best_params_)
        fitted_prep = grid.best_estimator_.named_steps["prep"]
        fitted_clf  = grid.best_estimator_.named_steps["clf"]
    train_time = time.time() - t0

    # --- save classifier + preprocessor ---
    pre_path = os.path.abspath(os.path.join(args.out_dir, f"{args.basename}_preprocessor.pkl"))
    clf_path = os.path.abspath(os.path.join(args.out_dir, f"{args.basename}_clf.pkl"))
    joblib.dump(fitted_prep, pre_path)
    joblib.dump(fitted_clf,  clf_path)
    print(f"[OK] Saved preprocessor → {pre_path}")
    print(f"[OK] Saved classifier   → {clf_path}")

    # --- VALIDATION (hold-out) metrics ---
    Xt_val = fitted_prep.transform(X_val)
    y_pred = fitted_clf.predict(Xt_val)
    y_proba = fitted_clf.predict_proba(Xt_val) if hasattr(fitted_clf, "predict_proba") else None

    acc = float(accuracy_score(y_val, y_pred))
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=classes_sorted)

    # Multiclass ROC-AUC (macro, OVR) if we have probabilities
    try:
        y_bin = label_binarize(y_val, classes=classes_sorted)
        roc_macro = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
    except Exception:
        roc_macro = None

    # --- feature importances (optional) ---
    try:
        importances = getattr(fitted_clf, "feature_importances_", None)
        if importances is not None:
            feat_names = getattr(fitted_prep, "get_feature_names_out")()
            # align lengths just in case
            n = min(len(importances), len(feat_names))
            imp_df = pd.DataFrame({"feature": feat_names[:n], "importance": importances[:n]})
            imp_path = os.path.join(args.out_dir, args.imp_csv)
            imp_df.sort_values("importance", ascending=False).to_csv(imp_path, index=False)
            print(f"[OK] Saved feature importances → {os.path.abspath(imp_path)}")
    except Exception as e:
        print(f"[WARN] Could not save feature importances: {e}")

    # --- save metrics & confusion matrix ---
    metrics = {
        "classes": classes_sorted,
        "samples": {"train": int(len(y_train)), "val": int(len(y_val))},
        "accuracy": acc,
        "precision_macro": report.get("macro avg", {}).get("precision"),
        "recall_macro":    report.get("macro avg", {}).get("recall"),
        "f1_macro":        report.get("macro avg", {}).get("f1-score"),
        "precision_weighted": report.get("weighted avg", {}).get("precision"),
        "recall_weighted":    report.get("weighted avg", {}).get("recall"),
        "f1_weighted":        report.get("weighted avg", {}).get("f1-score"),
        "per_class": {k: v for k, v in report.items() if k in classes_sorted},
        "roc_auc_macro_ovr": roc_macro,
        "train_time_sec": train_time,
        "params": {
            "fast": args.fast, "seed": args.seed,
            "contamination": args.contamination, "features_numeric": NUMERIC,
            "features_categorical": CATEGORICAL,
        },
    }
    mpath = os.path.join(args.out_dir, args.metrics_json)
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved metrics → {os.path.abspath(mpath)}")

    cm_df = pd.DataFrame(cm, index=classes_sorted, columns=classes_sorted)
    cmpath = os.path.join(args.out_dir, args.cm_csv)
    cm_df.to_csv(cmpath)
    print(f"[OK] Saved confusion matrix → {os.path.abspath(cmpath)}")

    # --- IsolationForest on TRAIN NORMAL samples ---
    Xt_train = fitted_prep.transform(X_train)
    normal_mask = y_train.str.lower().eq("normal").to_numpy()
    n_norm = int(normal_mask.sum())
    iso_path  = os.path.abspath(os.path.join(args.out_dir, f"{args.basename}_iso.pkl"))
    meta_path = os.path.abspath(os.path.join(args.out_dir, f"{args.basename}_iso_meta.json"))
    if n_norm >= 50:
        print(f"[INFO] Fitting IsolationForest on {n_norm} Normal training samples…")
        iso = IsolationForest(
            n_estimators=400, contamination=args.contamination, random_state=args.seed, n_jobs=-1
        )
        idx = np.flatnonzero(normal_mask)
        Xt_norm = Xt_train[idx]
        try:
            iso.fit(Xt_norm)
        except Exception as e:
            print(f"[WARN] Sparse fit failed ({e}); densifying…")
            iso.fit(Xt_norm.toarray() if hasattr(Xt_norm, "toarray") else np.asarray(Xt_norm))

        # threshold on training normals
        try:
            normal_scores = -iso.decision_function(Xt_train[idx])
        except Exception as e:
            print(f"[WARN] decision_function on sparse failed ({e}); densifying…")
            X_all = Xt_train.toarray() if hasattr(Xt_train, "toarray") else np.asarray(Xt_train)
            normal_scores = -iso.decision_function(X_all[idx])

        iso_threshold = float(np.quantile(normal_scores, 0.95))
        meta = {
            "iso_threshold": iso_threshold,
            "contamination": args.contamination,
            "feature_names": list(getattr(fitted_prep, "get_feature_names_out")()),
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "classes_": list(getattr(fitted_clf, "classes_", [])),
        }
        joblib.dump(iso, iso_path)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[OK] Saved IsolationForest → {iso_path}")
        print(f"[OK] Saved iso meta       → {meta_path}")
    else:
        print(f"[WARN] Not enough Normal samples in TRAIN ({n_norm}) for IsolationForest. Skipping.")

    print("[DONE] All artifacts & metrics saved in:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
