# student_result_analyzer
Student Result Analyzer built with Python. Cleans and analyzes student marks using Pandas, generates charts with Matplotlib, and creates automated PDF reports. Supports CSV/Excel input, calculates totals, percentages, grades, and provides clear performance insights.
📊 Student Result Analyzer (Python)

A powerful and beginner-friendly Python-based analytics tool that helps clean, analyze, visualize, and generate reports from student result data.
Built using Pandas, NumPy, Matplotlib, and ReportLab.

🚀 Features
✅ Data Import

Supports CSV and Excel (.xlsx) files

Automatic column detection

Converts inconsistent values (".", "-", "abc") into clean format

🔧 Data Cleaning

Handles missing or invalid marks

Auto-fixes incorrect values

Option to fill missing marks using subject-wise averages

📈 Data Analysis

Calculates:

Total Marks

Percentage

Grade (A+ to F)

Pass/Fail Status

Generates subject-wise statistics (mean, max, min, std)

📊 Visualizations

Automatically creates the following charts:

Subject-wise Average Bar Chart

Grade Distribution Pie Chart

Top Performers Bar Chart

All charts are saved as PNG files.

📄 Automated PDF Report

Generates a clean, formatted PDF containing:

Class summary

Pass/Fail count

Average class performance

Embedded charts

📁 Project Structure
student_result_analyzer/
│
├─ sample_data.csv
├─ student_result_analyzer.py
├─ outputs/
│   ├─ processed_data.csv
│   ├─ charts/
│   └─ report.pdf
└─ README.md

🛠️ Installation

Install required packages:

pip install pandas numpy matplotlib reportlab openpyxl

▶️ Usage

Run the analyzer:

python student_result_analyzer.py --input sample_data.csv --out outputs


To fill missing marks using subject averages:

python student_result_analyzer.py --input sample_data.csv --out outputs --fill-mean

📝 Output Includes

Cleaned & processed CSV

Subject statistics

Summary dictionary

3 auto-generated charts

PDF report

🎯 Ideal For

School/college record analysis

Data science beginners

Python developers learning Pandas + Matplotlib

Mini-projects & academic submissions

🤝 Contribute

Feel free to suggest improvements or submit pull requests!
