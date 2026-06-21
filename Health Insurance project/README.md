# 🛡️ Health Insurance Claim Fraud Detection

## ✅ Works on Windows · Python 3.12 · CPU only · No GPU needed

**No TensorFlow. No PyTorch. No Keras.**

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Mac/Linux

# 2. Install — only standard ML packages needed
pip install -r requirements.txt

# 3. Generate synthetic dataset
cd data
python generate_data.py
cd ..

# 4. Train all models (Jupyter)
jupyter notebook fraud_detection_training.ipynb
# Run all cells — takes ~5–10 min on CPU

# 5. Launch web app
python app.py
# Visit http://localhost:5000
```

---

## Models

| # | Model | Library | Notes |
|---|---|---|---|
| 1 | **XGBoost** | `xgboost` | Gradient boosting, handles class imbalance |
| 2 | **LightGBM** | `lightgbm` | Fast gradient boosting, leaf-wise |
| 3 | **MLP Neural Network** | `scikit-learn` | 3-layer perceptron (replaces LSTM for tabular data) |
| 4 | **Isolation Forest** | `scikit-learn` | Unsupervised anomaly detection |
| 5 | **PCA Autoencoder** | `scikit-learn` | Reconstruction-error anomaly detection (replaces neural AE) |
| 6 | **Stacking Ensemble** | `scikit-learn` | LR meta-learner trained on OOF predictions |
| 7 | **Weighted Voting** | custom | AUC-weighted average of all model scores |

> Models 3 and 5 are scikit-learn equivalents of LSTM and neural Autoencoder.
> They achieve comparable accuracy on tabular data with zero extra dependencies.

---

## Project Structure

```
fraud_detection/
├── app.py                           # Flask web app
├── requirements.txt                 # pip install -r requirements.txt
├── README.md
├── fraud_detection_training.ipynb   # Training notebook (13 cells)
├── data/
│   ├── generate_data.py
│   └── synthetic_insurance_claims.csv
├── models/                          # Created after running notebook
│   ├── scaler.pkl
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── mlp_model.pkl
│   ├── iso_forest.pkl
│   ├── pca_autoencoder.pkl
│   ├── meta_learner.pkl
│   └── meta.json
└── templates/
    ├── base.html
    ├── index.html
    ├── predict.html
    ├── result.html
    └── dashboard.html
```

## API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"claim_amount\": 6000, \"num_procedures\": 12, \"prior_claims_6m\": 9}"
```
