"""
Health Insurance Claim Fraud Detection — Flask Web Application
Run: python app.py   →   http://localhost:5000

Works in DEMO mode (heuristic) without trained models.
Train models via the notebook first for full ML predictions.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json, os, joblib, traceback
from datetime import datetime

app = Flask(__name__)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_DIR     = "models"
MODELS_LOADED = False
scaler = meta_info = xgb_model = lgb_model = mlp_model = None
iso_forest = pca_ae = meta_learner = None

def load_models():
    global scaler, meta_info, xgb_model, lgb_model, mlp_model
    global iso_forest, pca_ae, meta_learner, MODELS_LOADED
    try:
        scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        xgb_model    = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
        lgb_model    = joblib.load(os.path.join(MODEL_DIR, "lgb_model.pkl"))
        mlp_model    = joblib.load(os.path.join(MODEL_DIR, "mlp_model.pkl"))
        iso_forest   = joblib.load(os.path.join(MODEL_DIR, "iso_forest.pkl"))
        pca_ae       = joblib.load(os.path.join(MODEL_DIR, "pca_autoencoder.pkl"))
        meta_learner = joblib.load(os.path.join(MODEL_DIR, "meta_learner.pkl"))
        with open(os.path.join(MODEL_DIR, "meta.json")) as f:
            meta_info = json.load(f)
        MODELS_LOADED = True
        print("✅ All models loaded")
    except Exception as e:
        print(f"⚠️  Models not found ({e}) — running in DEMO mode")
        MODELS_LOADED = False

load_models()

# ── Heuristic scorer (demo mode) ──────────────────────────────────────────────
def heuristic_fraud_score(feat):
    score = 0.05
    if feat.get("claim_amount",     0) > 4000: score += 0.25
    if feat.get("num_procedures",   0) > 6:    score += 0.20
    if feat.get("prior_claims_6m",  0) > 6:    score += 0.20
    if feat.get("approval_ratio",   1) < 0.4:  score += 0.15
    if feat.get("lab_tests_ordered",0) > 8:    score += 0.10
    if feat.get("prescriptions",    0) > 6:    score += 0.10
    if feat.get("num_diagnoses",    0) > 5:    score += 0.10
    return min(score, 0.99)

# ── Dataset stats ──────────────────────────────────────────────────────────────
def load_stats():
    try:
        df = pd.read_csv("data/synthetic_insurance_claims.csv",
                         parse_dates=["claim_date"])
        total       = len(df)
        fraud_count = int(df["is_fraud"].sum())
        legit_count = total - fraud_count

        df["month"] = df["claim_date"].dt.to_period("M").astype(str)
        monthly = (df.groupby("month")["is_fraud"]
                     .agg(["sum","count"]).reset_index()
                     .rename(columns={"sum":"fraud","count":"total"})
                     .tail(12))

        spec = (df.groupby("provider_specialty")["is_fraud"]
                  .mean().sort_values(ascending=False).head(5))

        return {
            "total":             total,
            "fraud_count":       fraud_count,
            "legit_count":       legit_count,
            "fraud_rate":        round(fraud_count/total*100, 1),
            "avg_claim":         round(df["claim_amount"].mean(), 2),
            "avg_fraud_claim":   round(df[df["is_fraud"]==1]["claim_amount"].mean(), 2),
            "monthly_labels":    monthly["month"].tolist(),
            "monthly_fraud":     monthly["fraud"].tolist(),
            "monthly_total":     monthly["total"].tolist(),
            "spec_labels":       spec.index.tolist(),
            "spec_fraud_rates":  [round(v*100,1) for v in spec.values],
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {}

# ── Feature columns (must match notebook) ─────────────────────────────────────
DEFAULT_FEATURES = [
    "provider_specialty","patient_gender","insurance_type","diagnosis_code","procedure_code",
    "claim_amount","approved_amount","num_procedures","num_diagnoses","patient_age",
    "prior_claims_6m","days_in_hospital","emergency_flag","chronic_condition",
    "lab_tests_ordered","prescriptions","approval_ratio","claim_day_of_week","claim_month",
    "los","procedures_per_day","amount_per_procedure",
    "provider_avg_claim","provider_fraud_rate","provider_claim_count",
    "patient_avg_claim","patient_claim_count"
]
SPECIALTY_MAP = {"General Practice":0,"Cardiology":1,"Orthopedics":2,"Neurology":3,
                 "Oncology":4,"Dermatology":5,"Psychiatry":6,
                 "Emergency Medicine":7,"Radiology":8,"Surgery":9}
INSURANCE_MAP = {"HMO":0,"PPO":1,"EPO":2,"Medicare":3,"Medicaid":4}
GENDER_MAP    = {"M":0,"F":1}

def build_feature_vector(form):
    ca   = float(form.get("claim_amount",     1500))
    aa   = float(form.get("approved_amount",  1350))
    np_  = int(form.get("num_procedures",  2))
    nd   = int(form.get("num_diagnoses",   1))
    los  = int(form.get("los",             2))
    ar   = aa / (ca + 1e-9)
    ppd  = np_ / (los + 1)
    app_ = ca  / (np_ + 1)

    raw = {
        "provider_specialty":   SPECIALTY_MAP.get(form.get("specialty","General Practice"),0),
        "patient_gender":       GENDER_MAP.get(form.get("gender","M"),0),
        "insurance_type":       INSURANCE_MAP.get(form.get("insurance_type","PPO"),1),
        "diagnosis_code":       int(form.get("diagnosis_code_id", 10)),
        "procedure_code":       int(form.get("procedure_code_id", 10)),
        "claim_amount":         ca,
        "approved_amount":      aa,
        "num_procedures":       np_,
        "num_diagnoses":        nd,
        "patient_age":          int(form.get("patient_age",  45)),
        "prior_claims_6m":      int(form.get("prior_claims_6m", 1)),
        "days_in_hospital":     los,
        "emergency_flag":       int(form.get("emergency_flag", 0)),
        "chronic_condition":    int(form.get("chronic_condition", 0)),
        "lab_tests_ordered":    int(form.get("lab_tests",    2)),
        "prescriptions":        int(form.get("prescriptions",1)),
        "approval_ratio":       ar,
        "claim_day_of_week":    datetime.now().weekday(),
        "claim_month":          datetime.now().month,
        "los":                  los,
        "procedures_per_day":   ppd,
        "amount_per_procedure": app_,
        "provider_avg_claim":   float(form.get("provider_avg_claim",   1500)),
        "provider_fraud_rate":  float(form.get("provider_fraud_rate",  0.05)),
        "provider_claim_count": int(form.get("provider_claim_count",   50)),
        "patient_avg_claim":    ca,
        "patient_claim_count":  int(form.get("prior_claims_6m",1)) + 1,
    }
    feat_names = (meta_info["feature_cols"] if MODELS_LOADED and meta_info
                  else DEFAULT_FEATURES)
    X = np.array([[raw[f] for f in feat_names]], dtype=np.float32)
    return X, raw

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", stats=load_stats(), models_loaded=MODELS_LOADED)

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "GET":
        return render_template("predict_.html", models_loaded=MODELS_LOADED)

    try:
        X_raw, raw = build_feature_vector(request.form)
        scores = {}

        if MODELS_LOADED:
            X_sc = scaler.transform(X_raw)

            # Tree models: raw (unscaled) features; MLP/anomaly: scaled
            scores["XGBoost"]       = float(np.clip(xgb_model.predict_proba(X_raw)[:,1][0], 0, 1))
            scores["LightGBM"]      = float(np.clip(lgb_model.predict_proba(X_raw)[:,1][0], 0, 1))
            scores["MLP_NeuralNet"] = float(np.clip(mlp_model.predict_proba(X_sc)[:,1][0],  0, 1))

            # Isolation Forest: higher decision score = more normal, so invert
            iso_s = iso_forest.decision_function(X_sc)[0]
            scores["IsolationForest"] = float(np.clip(0.5 - iso_s, 0, 1))

            # PCA Autoencoder: reconstruction error → anomaly probability
            recon     = pca_ae.inverse_transform(pca_ae.transform(X_sc))
            recon_err = float(np.mean(np.square(X_sc - recon)))
            scores["PCA_Autoencoder"] = float(np.clip(recon_err, 0, 1))

            # Stacking meta-learner
            meta_feats = np.array([[scores["XGBoost"],
                                    scores["LightGBM"],
                                    scores["MLP_NeuralNet"]]])
            scores["Ensemble_Stacking"] = float(np.clip(
                meta_learner.predict_proba(meta_feats)[:,1][0], 0, 1))

            # Weighted ensemble — tree models get highest weight (most reliable)
            ensemble_prob = float(np.clip(
                scores["XGBoost"]           * 0.30 +
                scores["LightGBM"]          * 0.30 +
                scores["MLP_NeuralNet"]     * 0.20 +
                scores["IsolationForest"]   * 0.08 +
                scores["PCA_Autoencoder"]   * 0.04 +
                scores["Ensemble_Stacking"] * 0.08,
                0, 1))
        else:
            h = heuristic_fraud_score(raw)
            scores = {"XGBoost": h*.90, "LightGBM": h*.95, "MLP_NeuralNet": h*.85,
                      "IsolationForest": h*.70, "PCA_Autoencoder": h*.65, "Ensemble_Stacking": h}
            ensemble_prob = h

        scores["Ensemble_Final"] = ensemble_prob
        risk_level = "HIGH" if ensemble_prob >= 0.7 else "MEDIUM" if ensemble_prob >= 0.4 else "LOW"

        risk_factors = []
        if raw["claim_amount"]      > 3000: risk_factors.append(f"High claim amount (${raw['claim_amount']:,.0f})")
        if raw["num_procedures"]    > 5:    risk_factors.append(f"Excessive procedures ({raw['num_procedures']})")
        if raw["prior_claims_6m"]   > 5:    risk_factors.append(f"Many prior claims ({raw['prior_claims_6m']} in 6 months)")
        if raw["approval_ratio"]    < 0.5:  risk_factors.append(f"Low approval ratio ({raw['approval_ratio']:.0%})")
        if raw["lab_tests_ordered"] > 7:    risk_factors.append(f"Excess lab orders ({raw['lab_tests_ordered']})")
        if raw["prescriptions"]     > 5:    risk_factors.append(f"Over-prescribing ({raw['prescriptions']} Rx)")
        if raw["num_diagnoses"]     > 5:    risk_factors.append(f"Unusual diagnosis count ({raw['num_diagnoses']})")
        if not risk_factors:               risk_factors.append("No major risk indicators detected")

        return render_template("result.html",
            is_fraud      = ensemble_prob >= 0.5,
            risk_level    = risk_level,
            ensemble_prob = round(ensemble_prob * 100, 1),
            scores        = {k: round(v*100, 1) for k,v in scores.items()},
            risk_factors  = risk_factors,
            raw           = raw,
            models_loaded = MODELS_LOADED,
        )
    except Exception as e:
        traceback.print_exc()
        return render_template("predict_.html", error=str(e), models_loaded=MODELS_LOADED)

@app.route("/dashboard")
def dashboard():
    results_summary = (meta_info.get("results_summary", {})
                       if MODELS_LOADED and meta_info else {})
    return render_template("dashboard.html", stats=load_stats(),
                           results_summary=results_summary, models_loaded=MODELS_LOADED)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    X_raw, raw = build_feature_vector(data)
    prob = heuristic_fraud_score(raw)
    if MODELS_LOADED:
        prob = float(xgb_model.predict_proba(X_raw)[:,1][0])
    return jsonify({
        "fraud_probability": round(float(prob), 4),
        "is_fraud":          bool(prob >= 0.5),
        "risk_level":        "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)