"""student_performance_project.py

Student Performance Prediction Using Machine Learning
Author: HARSH PRATAP SINGH
Reg No: 25BME10011
Course: Fundamentals of AI and ML, VIT Bhopal University
FACULTY: DR. VINESH KUMAR
This script:
- generates a synthetic dataset (300 entries)
- trains Linear Regression and RandomForestRegressor
- evaluates models and picks the best one
- saves model and creates graphs
- generates a multi-page PDF report with graphs embedded
"""

import os
import random
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ReportLab for PDF generation
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                PageBreak, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm

# ---------- Config ----------
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "student_performance.csv")
MODEL_PATH = os.path.join(OUT_DIR, "student_model.pkl")
GRAPH_ACTUAL_PRED = os.path.join(OUT_DIR, "actual_vs_pred.png")
GRAPH_FEATURE_IMP = os.path.join(OUT_DIR, "feature_importance.png")
PDF_PATH = os.path.join(OUT_DIR, "student_performance_report_final.pdf")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ---------- 1) Synthetic dataset generation ----------
def generate_synthetic_data(n_samples=300, random_state=SEED):
    """
    Create a realistic synthetic dataset with a controlled relationship:
    - study_hours (1-7)
    - attendance (50-100)
    - sleep_hours (4-9)
    - internet_usage (0.5-8)
    - marks (0-100) computed as weighted combination + noise
    """
    rng = np.random.RandomState(random_state)
    study_hours = np.round(rng.normal(loc=3.5, scale=1.4, size=n_samples).clip(0.5, 8), 2)
    attendance = np.round(rng.normal(loc=80, scale=10, size=n_samples).clip(50, 100), 2)
    sleep_hours = np.round(rng.normal(loc=7, scale=1, size=n_samples).clip(4, 9), 2)
    internet_usage = np.round(rng.normal(loc=3, scale=1.8, size=n_samples).clip(0, 10), 2)

    # Base marks: influence weights chosen to reflect plausible relationships
    # study_hours: strong positive, attendance: positive, sleep: moderate positive, internet: negative
    marks_base = (
        study_hours * 8.5  # weight
        + (attendance / 100) * 30
        + sleep_hours * 3
        - internet_usage * 2.5
    )

    # Add some previous-score-like variance and random noise
    noise = rng.normal(0, 6, n_samples)  # gaussian noise
    marks = (marks_base + noise).clip(0, 100).round(2)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "internet_usage": internet_usage,
        "marks": marks
    })

    return df


# ---------- 2) Save dataset ----------
df = generate_synthetic_data(300)
df.to_csv(CSV_PATH, index=False)
print(f"[+] Dataset saved to: {CSV_PATH}")
print(df.head())


# ---------- 3) Training pipeline ----------
def train_and_evaluate(csv_path):
    df = pd.read_csv(csv_path)
    X = df[["study_hours", "attendance", "sleep_hours", "internet_usage"]]
    y = df["marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=150, random_state=SEED)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)

    results = {
        "LinearRegression": {"model": lr, "mae": lr_mae, "r2": lr_r2, "pred": y_pred_lr},
        "RandomForest": {"model": rf, "mae": rf_mae, "r2": rf_r2, "pred": y_pred_rf},
        "X_test": X_test,
        "y_test": y_test
    }
    return results


results = train_and_evaluate(CSV_PATH)

print("[+] Linear Regression - MAE: {:.3f}, R2: {:.3f}".format(
    results["LinearRegression"]["mae"], results["LinearRegression"]["r2"]
))
print("[+] Random Forest - MAE: {:.3f}, R2: {:.3f}".format(
    results["RandomForest"]["mae"], results["RandomForest"]["r2"]
))


# ---------- 4) Pick best model & save ----------
best_key = "RandomForest" if results["RandomForest"]["r2"] >= results["LinearRegression"]["r2"] else "LinearRegression"
best_model = results[best_key]["model"]

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f"[+] Best model ({best_key}) saved to: {MODEL_PATH}")


# ---------- 5) Create plots ----------
def create_plots(results, out_actual_graph, out_feat_graph):
    # Actual vs Predicted (for best model)
    X_test = results["X_test"]
    y_test = results["y_test"].values
    y_pred = results[best_key]["pred"]

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, s=40)
    lims = [min(min(y_test), min(y_pred)) - 5, max(max(y_test), max(y_pred)) + 5]
    plt.plot(lims, lims, "--", linewidth=1, label="Ideal")
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.title("Actual vs Predicted Marks")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_actual_graph, dpi=150)
    plt.close()
    print(f"[+] Saved graph: {out_actual_graph}")

    # Feature Importance (if RandomForest)
    feat_importance = None
    feat_names = list(X_test.columns)
    if best_key == "RandomForest":
        feat_importance = best_model.feature_importances_
    else:
        # For linear regression use absolute coefficients scaled
        coef = np.abs(best_model.coef_)
        feat_importance = coef / coef.sum()

    # plot
    plt.figure(figsize=(7, 4))
    y_pos = np.arange(len(feat_names))
    plt.bar(y_pos, feat_importance, align='center', alpha=0.9)
    plt.xticks(y_pos, feat_names, rotation=0)
    plt.ylabel("Importance (normalized)")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(out_feat_graph, dpi=150)
    plt.close()
    print(f"[+] Saved graph: {out_feat_graph}")


create_plots(results, GRAPH_ACTUAL_PRED, GRAPH_FEATURE_IMP)


# ---------- 6) Generate PDF report ----------
def generate_pdf(pdf_path, csv_path, graph1, graph2,
                 student_name="HARSH PRATAP SINGH",
                 reg_no="25BME10011",
                 course="Fundamentals of AI and ML",
                 university="VIT Bhopal University"):
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading = ParagraphStyle('Heading', parent=styles['Heading2'], spaceAfter=6)
    normal = styles['BodyText']

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    story = []

    # Cover
    story.append(Paragraph("Student Performance Prediction Using Machine Learning", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"<b>Submitted by:</b> {student_name}<br/>"
        f"<b>Registration Number:</b> {reg_no}<br/>"
        f"<b>Course:</b> {course}<br/>"
        f"<b>University:</b> {university}<br/>"
        f"<b>Date:</b> {datetime.now().strftime('%B %Y')}"
    , normal))
    story.append(PageBreak())

    # Abstract
    story.append(Paragraph("1. ABSTRACT", heading))
    story.append(Paragraph(
        "This project trains ML regression models to predict student marks (0–100) using study hours, "
        "attendance, sleep hours and internet usage. Two models were compared and the best model was selected."
    , normal))
    story.append(PageBreak())

    # Methodology & Dataset
    story.append(Paragraph("2. METHODOLOGY & DATASET", heading))
    story.append(Paragraph(
        "Dataset: synthetic 300 samples with columns: study_hours, attendance, sleep_hours, internet_usage, marks.\n\n"
        "Methodology steps: Data generation -> Preprocessing -> Train/Test split (80/20) -> Train models -> Evaluate (MAE, R²) -> Save best model -> Visualize"
    , normal))
    story.append(PageBreak())

    # Results + Table snapshot
    story.append(Paragraph("3. RESULTS & EVALUATION", heading))
    lr_mae = results["LinearRegression"]["mae"]
    lr_r2 = results["LinearRegression"]["r2"]
    rf_mae = results["RandomForest"]["mae"]
    rf_r2 = results["RandomForest"]["r2"]

    res_table = [
        ["Model", "MAE (lower better)", "R² (higher better)"],
        ["Linear Regression", f"{lr_mae:.3f}", f"{lr_r2:.3f}"],
        ["Random Forest", f"{rf_mae:.3f}", f"{rf_r2:.3f}"],
        ["Selected (best)", best_key, ""]
    ]
    tbl = Table(res_table, colWidths=[7*cm, 4*cm, 4*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (1,1), (-1,-1), 'CENTER')
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # Add graphs
    story.append(Paragraph("Graph: Actual vs Predicted (test set)", styles['Heading3']))
    story.append(Image(graph1, width=16*cm, height=9*cm))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Graph: Feature Importance", styles['Heading3']))
    story.append(Image(graph2, width=12*cm, height=6*cm))
    story.append(PageBreak())

    # Conclusion & Future scope
    story.append(Paragraph("4. CONCLUSION & FUTURE SCOPE", heading))
    story.append(Paragraph(
        "Random Forest delivered the best accuracy for this synthetic dataset. Future work: use real institutional"
        " data, add more features (assignments, previous exam scores, socio-economic indicators), and deploy as a web app."
    , normal))

    # References & Appendix sample code snippet
    story.append(Spacer(1, 10))
    story.append(Paragraph("References:", styles['Heading3']))
    story.append(Paragraph("- scikit-learn documentation\n- Python for Data Analysis (Wes McKinney)\n- Andrew Ng ML course", normal))
    story.append(PageBreak())

    # Appendix - small sample of the dataset
    story.append(Paragraph("Appendix: Sample dataset rows", heading))
    sample = pd.read_csv(csv_path).sample(6, random_state=SEED)
    sample_table = [list(sample.columns)] + sample.values.tolist()
    t = Table(sample_table, colWidths=[3.2*cm]*5)
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(t)

    doc.build(story)
    print(f"[+] PDF generated: {pdf_path}")


generate_pdf(PDF_PATH, CSV_PATH, GRAPH_ACTUAL_PRED, GRAPH_FEATURE_IMP)
print("[*] All done. Look")