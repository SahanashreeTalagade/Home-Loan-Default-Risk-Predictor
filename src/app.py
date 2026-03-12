import gradio as gr
import pandas as pd
import numpy as np
import joblib

# ── Load artifacts ──────────────────────────────────────────
model     = joblib.load("artifacts/xgb_model.pkl")
scaler    = joblib.load("artifacts/scaler.pkl")
columns   = joblib.load("artifacts/feature_columns.pkl")
threshold = joblib.load("artifacts/threshold.pkl")

# ── Preprocessing ────────────────────────────────────────────
def preprocess(df):
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return scaler.transform(df)

# ── Manual Input Prediction ──────────────────────────────────
def predict_manual(AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY,
                   DAYS_BIRTH, DAYS_EMPLOYED, EXT_SOURCE_1,
                   EXT_SOURCE_2, EXT_SOURCE_3, INST_LATE_COUNT,
                   BUREAU_CREDIT_DAY_OVERDUE_MEAN):

    data = {
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "AMT_CREDIT": AMT_CREDIT,
        "AMT_ANNUITY": AMT_ANNUITY,
        "DAYS_BIRTH": DAYS_BIRTH,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "INST_LATE_COUNT": INST_LATE_COUNT,
        "BUREAU_CREDIT_DAY_OVERDUE_MEAN": BUREAU_CREDIT_DAY_OVERDUE_MEAN
    }

    df = pd.DataFrame([data])
    X  = preprocess(df)

    proba = model.predict_proba(X)[:, 1][0]
    label = "🔴 HIGH RISK - LIKELY DEFAULT" if proba >= threshold else "🟢 LOW RISK - LIKELY SAFE"

    return f"{label}\n\nDefault Probability: {proba:.1%}\nThreshold Used: {threshold}"

# ── CSV Batch Prediction ─────────────────────────────────────
def predict_csv(file):
    df     = pd.read_csv(file.name)
    X      = preprocess(df)
    probas = model.predict_proba(X)[:, 1]
    labels = ["🔴 DEFAULT" if p >= threshold else "🟢 SAFE" for p in probas]

    df["Default_Probability"] = [f"{p:.1%}" for p in probas]
    df["Prediction"]          = labels
    return df

# ── Manual Input Tab ─────────────────────────────────────────
manual_tab = gr.Interface(
    fn=predict_manual,
    inputs=[
        gr.Number(label="Annual Income",              value=150000),
        gr.Number(label="Credit Amount",              value=500000),
        gr.Number(label="Annuity Amount",             value=25000),
        gr.Number(label="Days Birth (negative)",      value=-15000),
        gr.Number(label="Days Employed (negative)",   value=-2000),
        gr.Number(label="External Score 1",           value=0.5),
        gr.Number(label="External Score 2",           value=0.5),
        gr.Number(label="External Score 3",           value=0.5),
        gr.Number(label="Late Installment Count",     value=0),
        gr.Number(label="Bureau Overdue Days (Mean)", value=0),
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=4),
    title="🏦 Home Loan Default Predictor",
    description="Enter customer details below to predict default risk."
)

# ── CSV Upload Tab ───────────────────────────────────────────
csv_tab = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(label="Upload CSV File"),
    outputs=gr.Dataframe(label="Batch Prediction Results"),
    title="📂 Batch Prediction via CSV",
    description="Upload a CSV file with customer data to get predictions for all rows."
)

# ── Launch App ───────────────────────────────────────────────
app = gr.TabbedInterface(
    interface_list=[manual_tab, csv_tab],
    tab_names=["Manual Input", "CSV Upload"]
)

if __name__ == "__main__":
    app.launch()
