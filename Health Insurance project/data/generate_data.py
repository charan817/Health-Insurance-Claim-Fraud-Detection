"""
Health Insurance Claim Fraud Detection - Synthetic Data Generator
Run this first to create training data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_LEGIT = 18000
N_FRAUD = 2000
N = N_LEGIT + N_FRAUD

def random_dates(start, end, n):
    delta = (end - start).days
    return [start + timedelta(days=random.randint(0, delta)) for _ in range(n)]

# ─── Base Dates ────────────────────────────────────────────────────────────────
start_date = datetime(2020, 1, 1)
end_date   = datetime(2024, 6, 30)

claim_dates = random_dates(start_date, end_date, N)

# ─── Provider / Patient pools ─────────────────────────────────────────────────
n_providers = 500
n_patients  = 8000

provider_ids = [f"PROV{str(i).zfill(4)}" for i in range(1, n_providers + 1)]
patient_ids  = [f"PAT{str(i).zfill(6)}"  for i in range(1, n_patients  + 1)]

# Mark ~5 % of providers and ~3 % of patients as suspicious
fraud_providers = set(random.sample(provider_ids, int(n_providers * 0.05)))
fraud_patients  = set(random.sample(patient_ids,  int(n_patients  * 0.03)))

specialties = [
    "General Practice", "Cardiology", "Orthopedics", "Neurology",
    "Oncology", "Dermatology", "Psychiatry", "Emergency Medicine",
    "Radiology", "Surgery"
]
diagnosis_codes = [f"ICD{str(i).zfill(4)}" for i in range(1, 201)]
procedure_codes = [f"CPT{str(i).zfill(5)}" for i in range(10000, 10201)]

# ─── Generate Legitimate Claims ───────────────────────────────────────────────
def generate_legit(n):
    rows = []
    for i in range(n):
        provider = random.choice(provider_ids)
        patient  = random.choice(patient_ids)
        specialty = random.choice(specialties)
        claim_amt  = abs(np.random.normal(1500, 800))
        approved   = abs(np.random.normal(1350, 720))
        approved   = min(approved, claim_amt)
        rows.append({
            "claim_id":           f"CLM{str(i).zfill(7)}",
            "patient_id":         patient,
            "provider_id":        provider,
            "provider_specialty": specialty,
            "claim_date":         claim_dates[i],
            "admission_date":     claim_dates[i] - timedelta(days=random.randint(0, 5)),
            "discharge_date":     claim_dates[i] + timedelta(days=random.randint(0, 7)),
            "diagnosis_code":     random.choice(diagnosis_codes),
            "procedure_code":     random.choice(procedure_codes),
            "claim_amount":       round(claim_amt, 2),
            "approved_amount":    round(approved, 2),
            "num_procedures":     random.randint(1, 4),
            "num_diagnoses":      random.randint(1, 3),
            "patient_age":        random.randint(18, 85),
            "patient_gender":     random.choice(["M", "F"]),
            "insurance_type":     random.choice(["HMO", "PPO", "EPO", "Medicare", "Medicaid"]),
            "prior_claims_6m":    random.randint(0, 3),
            "days_in_hospital":   random.randint(0, 7),
            "emergency_flag":     random.choice([0, 0, 0, 1]),
            "chronic_condition":  random.choice([0, 1]),
            "lab_tests_ordered":  random.randint(0, 5),
            "prescriptions":      random.randint(0, 4),
            "is_fraud":           0,
        })
    return rows

# ─── Generate Fraudulent Claims ────────────────────────────────────────────────
def generate_fraud(n, offset):
    rows = []
    fraud_types = ["upcoding", "phantom_billing", "duplicate", "unbundling", "unnecessary_services"]
    for i in range(n):
        fraud_type = random.choice(fraud_types)
        provider = random.choice(list(fraud_providers) + random.sample(provider_ids, 5))
        patient  = random.choice(list(fraud_patients)  + random.sample(patient_ids,  10))
        specialty = random.choice(specialties)

        # Inflate claim amount
        claim_amt = abs(np.random.normal(4500, 2000))
        approved  = abs(np.random.normal(1200, 600))  # low approval ratio
        approved  = min(approved, claim_amt)

        # Short/suspicious hospital stays for expensive procedures
        days_hosp = random.randint(0, 2) if fraud_type in ["phantom_billing","duplicate"] else random.randint(0, 5)

        # Implausible dates
        adm_offset = random.randint(-1, 0) if fraud_type == "duplicate" else random.randint(0, 3)
        adm = claim_dates[offset + i] - timedelta(days=adm_offset)
        dsc = adm + timedelta(days=days_hosp)

        rows.append({
            "claim_id":           f"FRD{str(i).zfill(7)}",
            "patient_id":         patient,
            "provider_id":        provider,
            "provider_specialty": specialty,
            "claim_date":         claim_dates[offset + i],
            "admission_date":     adm,
            "discharge_date":     dsc,
            "diagnosis_code":     random.choice(diagnosis_codes),
            "procedure_code":     random.choice(procedure_codes),
            "claim_amount":       round(claim_amt, 2),
            "approved_amount":    round(approved, 2),
            "num_procedures":     random.randint(5, 15),   # too many
            "num_diagnoses":      random.randint(4, 10),   # too many
            "patient_age":        random.randint(18, 85),
            "patient_gender":     random.choice(["M", "F"]),
            "insurance_type":     random.choice(["HMO", "PPO", "EPO", "Medicare", "Medicaid"]),
            "prior_claims_6m":    random.randint(5, 20),   # suspiciously high
            "days_in_hospital":   days_hosp,
            "emergency_flag":     random.choice([0, 1]),
            "chronic_condition":  random.choice([0, 1]),
            "lab_tests_ordered":  random.randint(8, 20),   # over-ordering
            "prescriptions":      random.randint(5, 15),   # over-prescribing
            "is_fraud":           1,
        })
    return rows

# ─── Combine & Engineer Features ──────────────────────────────────────────────
legit_rows = generate_legit(N_LEGIT)
fraud_rows = generate_fraud(N_FRAUD, N_LEGIT)

df = pd.DataFrame(legit_rows + fraud_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# Derived features
df["claim_date"]      = pd.to_datetime(df["claim_date"])
df["admission_date"]  = pd.to_datetime(df["admission_date"])
df["discharge_date"]  = pd.to_datetime(df["discharge_date"])

df["approval_ratio"]        = df["approved_amount"] / (df["claim_amount"] + 1e-9)
df["claim_day_of_week"]     = df["claim_date"].dt.dayofweek
df["claim_month"]           = df["claim_date"].dt.month
df["los"]                   = (df["discharge_date"] - df["admission_date"]).dt.days.clip(lower=0)
df["procedures_per_day"]    = df["num_procedures"] / (df["los"] + 1)
df["amount_per_procedure"]  = df["claim_amount"]   / (df["num_procedures"] + 1)

# Provider-level aggregations (risk signals)
prov_stats = df.groupby("provider_id").agg(
    provider_avg_claim   =("claim_amount", "mean"),
    provider_fraud_rate  =("is_fraud",     "mean"),
    provider_claim_count =("claim_id",     "count"),
).reset_index()
df = df.merge(prov_stats, on="provider_id", how="left")

# Patient-level aggregations
pat_stats = df.groupby("patient_id").agg(
    patient_avg_claim   =("claim_amount", "mean"),
    patient_claim_count =("claim_id",     "count"),
).reset_index()
df = df.merge(pat_stats, on="patient_id", how="left")

print(f"Dataset shape : {df.shape}")
print(f"Fraud rate    : {df['is_fraud'].mean()*100:.1f}%")
print(f"Columns       : {list(df.columns)}")

df.to_csv("synthetic_insurance_claims.csv", index=False)
print("\n✅  Saved  →  synthetic_insurance_claims.csv")
