#!/usr/bin/env python3
"""
Student Result Analyzer
Single-file implementation with:
- CSV/XLSX input
- Cleaning, validation
- Analysis: totals, percentage, grades
- Visualizations (Matplotlib)
- PDF report (ReportLab)
Usage:
  python student_result_analyzer.py --input sample_data.csv --out outputs
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from datetime import datetime

# ---------- Utilities ----------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ---------- 1) Data Import ----------
def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path, engine='openpyxl')
    else:
        df = pd.read_csv(path)
    # Standardize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------- 2) Data Cleaning ----------
def clean_marks_df(df: pd.DataFrame, subject_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    invalid_cells = []
    # Replace common placeholders with NaN
    df[subject_cols] = df[subject_cols].replace(['', ' ', '.', 'NA', 'N/A', 'na', '-'], np.nan)
    # Convert to numeric, coercing errors to NaN
    for col in subject_cols:
        before_nonnum = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # detect negative or >100 numbers as invalid -> set to NaN and record
        mask_invalid_range = (df[col] < 0) | (df[col] > 100)
        if mask_invalid_range.any():
            invalid_idx = df.loc[mask_invalid_range].index.tolist()
            for i in invalid_idx:
                invalid_cells.append(f"{col}@row{int(i)}")
            df.loc[mask_invalid_range, col] = np.nan
    return df, invalid_cells

# Fill missing strategy: by default, leave NaN; optionally fill with subject mean
def fill_missing_with_mean(df: pd.DataFrame, subject_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in subject_cols:
        mean_val = df[col].mean(skipna=True)
        df[col] = df[col].fillna(round(mean_val, 2))
    return df

# ---------- 3) Analysis ----------
def compute_student_scores(df: pd.DataFrame, subject_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df['Total'] = df[subject_cols].sum(axis=1)
    df['MaxTotal'] = 100 * len(subject_cols)
    df['Percentage'] = (df['Total'] / df['MaxTotal']) * 100
    df['Percentage'] = df['Percentage'].round(2)
    df['Grade'] = df['Percentage'].apply(assign_grade)
    df['Result'] = df[subject_cols].apply(lambda row: 'Fail' if (row < 33).any() else 'Pass', axis=1)
    return df

def assign_grade(percent: float) -> str:
    if pd.isna(percent):
        return 'N/A'
    p = float(percent)
    if p >= 90:
        return 'A+'
    if p >= 80:
        return 'A'
    if p >= 70:
        return 'B+'
    if p >= 60:
        return 'B'
    if p >= 50:
        return 'C'
    if p >= 33:
        return 'D'
    return 'F'

def subject_statistics(df: pd.DataFrame, subject_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for col in subject_cols:
        stats[col] = {
            'mean': round(df[col].mean(skipna=True), 2),
            'max': float(df[col].max(skipna=True)),
            'min': float(df[col].min(skipna=True)),
            'std': round(df[col].std(skipna=True), 2),
            'missing_count': int(df[col].isna().sum())
        }
    return stats

# ---------- 4) Visualizations ----------
def plot_subject_bar(stats: Dict[str, Dict[str, float]], outpath: str):
    subjects = list(stats.keys())
    means = [stats[s]['mean'] for s in subjects]
    plt.figure(figsize=(8,5))
    plt.bar(subjects, means)
    plt.title('Subject-wise Average Marks')
    plt.ylabel('Average')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_grade_distribution(df: pd.DataFrame, outpath: str):
    grp = df['Grade'].value_counts().sort_index()
    plt.figure(figsize=(6,6))
    plt.pie(grp.values, labels=grp.index, autopct='%1.1f%%', startangle=140)
    plt.title('Grade Distribution')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_top_performers(df: pd.DataFrame, outpath: str, top_n=5):
    top = df.sort_values('Percentage', ascending=False).head(top_n)
    plt.figure(figsize=(8,5))
    plt.bar(top['Name'].astype(str), top['Percentage'])
    plt.title(f'Top {top_n} Performers')
    plt.ylabel('Percentage')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------- 5) Reporting (PDF) ----------
def create_pdf_report(summary: Dict, charts: List[str], out_pdf: str):
    c = canvas.Canvas(out_pdf, pagesize=A4)
    width, height = A4
    margin = 2*cm
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "Student Result Analyzer - Report")
    y -= 1*cm
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 1*cm

    # Summary text
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Summary")
    y -= 0.6*cm
    c.setFont("Helvetica", 10)
    lines = [
        f"Total Students: {summary.get('total_students', 'N/A')}",
        f"Passed: {summary.get('passed', 'N/A')}  |  Failed: {summary.get('failed', 'N/A')}",
        f"Average Percentage (class): {summary.get('avg_percent', 'N/A')}"
    ]
    for ln in lines:
        c.drawString(margin, y, ln)
        y -= 0.5*cm

    # Insert charts (one per page if needed)
    for chart in charts:
        if y < 6*cm:
            c.showPage()
            y = height - margin
        c.drawImage(chart, margin, y-12*cm, width=width-2*margin, height=10*cm, preserveAspectRatio=True)
        y -= 11*cm

    c.showPage()
    c.save()

# ---------- 6) Orchestration ----------
def analyze_file(input_path: str, out_dir: str, fill_missing: bool=False):
    ensure_dir(out_dir)
    charts_dir = os.path.join(out_dir, 'charts')
    ensure_dir(charts_dir)

    df = load_data(input_path)
    # Basic validation: find subject columns (assume first 2 cols are Roll, Name)
    cols = df.columns.tolist()
    if len(cols) < 3:
        raise ValueError("Input should have at least Roll, Name and one subject column.")
    # Heuristic: first two columns considered Roll and Name if named so, else use first two
    subject_cols = cols[2:]
    # Clean
    df_clean, invalid_cells = clean_marks_df(df, subject_cols)
    if fill_missing:
        df_clean = fill_missing_with_mean(df_clean, subject_cols)

    # Analysis
    df_scores = compute_student_scores(df_clean, subject_cols)
    stats = subject_statistics(df_clean, subject_cols)

    # Summary
    total_students = len(df_scores)
    passed = int((df_scores['Result'] == 'Pass').sum())
    failed = total_students - passed
    avg_percent = round(df_scores['Percentage'].mean(skipna=True), 2)

    summary = {
        'total_students': total_students,
        'passed': passed,
        'failed': failed,
        'avg_percent': avg_percent,
        'invalid_cells': invalid_cells
    }

    # Save processed CSV
    processed_csv = os.path.join(out_dir, 'processed_data.csv')
    df_scores.to_csv(processed_csv, index=False)

    # Create charts
    chart1 = os.path.join(charts_dir, 'subject_avg.png')
    chart2 = os.path.join(charts_dir, 'grade_dist.png')
    chart3 = os.path.join(charts_dir, 'top_performers.png')
    plot_subject_bar(stats, chart1)
    plot_grade_distribution(df_scores, chart2)
    plot_top_performers(df_scores, chart3, top_n=5)

    # PDF report
    charts = [chart1, chart2, chart3]
    out_pdf = os.path.join(out_dir, 'report.pdf')
    create_pdf_report(summary, charts, out_pdf)

    # Return summary and paths
    return {
        'summary': summary,
        'stats': stats,
        'processed_csv': processed_csv,
        'charts': charts,
        'report_pdf': out_pdf
    }

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Student Result Analyzer")
    p.add_argument('--input', '-i', required=True, help='Input CSV/XLSX file path')
    p.add_argument('--out', '-o', default='outputs', help='Output folder')
    p.add_argument('--fill-mean', action='store_true', help='Fill missing marks with subject mean')
    return p.parse_args()

def main():
    args = parse_args()
    print("Loading and analyzing:", args.input)
    res = analyze_file(args.input, args.out, fill_missing=args.fill_mean)
    print("Analysis done.")
    print("Summary:", res['summary'])
    print("Processed CSV at:", res['processed_csv'])
    print("Charts:", res['charts'])
    print("PDF report:", res['report_pdf'])

if __name__ == '__main__':
    main()
