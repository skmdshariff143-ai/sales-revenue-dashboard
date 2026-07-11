"""
Automated Excel Report Generator
================================
This script generates a professional multi-sheet Excel report (data_cleaning_report.xlsx)
detailing the data cleaning results. It applies formatting, styles, gridlines,
Excel formulas, and charts using xlsxwriter.

Author: Shaik Mahammad Shariff
Tech Stack: Python, pandas, xlsxwriter
"""

import os
import json
import pandas as pd
import xlsxwriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
EXCEL_PATH = os.path.join(SCRIPT_DIR, 'data_cleaning_report.xlsx')

def main():
    # 1. Load data
    summary_path = os.path.join(DATA_DIR, 'cleaning_summary.json')
    log_path = os.path.join(DATA_DIR, 'cleaning_log.csv')
    cleaned_path = os.path.join(DATA_DIR, 'cleaned_orders.csv')
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    logs_df = pd.read_csv(log_path)
    clean_df = pd.read_csv(cleaned_path)

    # Calculate log counts per Step type
    log_counts = logs_df['Step'].value_counts().reset_index()
    log_counts.columns = ['StepName', 'ModificationsCount']

    # 2. Create Excel workbook
    workbook = xlsxwriter.Workbook(EXCEL_PATH)
    
    # ── Formatting / Styles ──
    # Palette: Executive Dark Emerald Theme
    COLOR_PRIMARY = '#0f5132'     # Deep Emerald
    COLOR_SECONDARY = '#d1e7dd'   # Light Emerald Accent
    COLOR_TEXT_LIGHT = '#ffffff'
    
    fmt_title = workbook.add_format({
        'bold': True, 'font_size': 16, 'font_name': 'Segoe UI',
        'font_color': COLOR_PRIMARY, 'align': 'left', 'valign': 'vcenter'
    })
    
    fmt_subtitle = workbook.add_format({
        'font_size': 10, 'font_name': 'Segoe UI', 'font_color': '#555555',
        'italic': True, 'align': 'left', 'valign': 'vcenter'
    })
    
    fmt_header = workbook.add_format({
        'bold': True, 'font_name': 'Segoe UI', 'font_size': 11,
        'bg_color': COLOR_PRIMARY, 'font_color': COLOR_TEXT_LIGHT,
        'border': 1, 'border_color': '#dddddd', 'align': 'center', 'valign': 'vcenter'
    })
    
    fmt_cell = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10, 'border': 1,
        'border_color': '#eeeeee', 'align': 'left', 'valign': 'vcenter'
    })
    
    fmt_cell_center = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10, 'border': 1,
        'border_color': '#eeeeee', 'align': 'center', 'valign': 'vcenter'
    })
    
    fmt_cell_currency = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10, 'border': 1,
        'border_color': '#eeeeee', 'align': 'right', 'valign': 'vcenter',
        'num_format': '$#,##0.00'
    })
    
    fmt_cell_number = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10, 'border': 1,
        'border_color': '#eeeeee', 'align': 'right', 'valign': 'vcenter',
        'num_format': '#,##0'
    })
    
    fmt_cell_pct = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 10, 'border': 1,
        'border_color': '#eeeeee', 'align': 'right', 'valign': 'vcenter',
        'num_format': '0.0%'
    })
    
    fmt_metric_label = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 11, 'bold': True,
        'bg_color': COLOR_SECONDARY, 'border': 1, 'border_color': '#b5d5c5',
        'align': 'left', 'valign': 'vcenter'
    })
    
    fmt_metric_val = workbook.add_format({
        'font_name': 'Segoe UI', 'font_size': 11, 'bold': True,
        'bg_color': COLOR_SECONDARY, 'border': 1, 'border_color': '#b5d5c5',
        'align': 'right', 'valign': 'vcenter'
    })

    # ═══════════════════════════════════════════════════════
    # SHEET 1: Summary Dashboard
    # ═══════════════════════════════════════════════════════
    ws_summary = workbook.add_worksheet('Executive Summary')
    ws_summary.hide_gridlines(2) # show gridlines on screen, hide on print
    
    # Headers
    logo_path = os.path.join(SCRIPT_DIR, '..', 'task1-sales-dashboard', 'logo.png')
    if os.path.exists(logo_path):
        ws_summary.insert_image('A1', logo_path, {'x_scale': 0.15, 'y_scale': 0.15, 'x_offset': 5, 'y_offset': 5})
    ws_summary.write('A2', '📊 Data Quality & Cleaning Automation Report', fmt_title)
    ws_summary.write('A3', 'Automated summary of 11-step data cleaning pipeline execution', fmt_subtitle)
    
    # Summary Metrics Cards
    metrics = [
        ("Raw Messy Rows", summary["initial_rows"], "#,##0"),
        ("Cleaned Rows", summary["final_rows"], "#,##0"),
        ("Total Issues Resolved", summary["quality_metrics"]["total_issues_resolved"], "#,##0"),
        ("Initial Quality Score", summary["quality_metrics"]["initial_quality_score_pct"] / 100.0, "0.0%"),
        ("Final Quality Score", summary["quality_metrics"]["final_quality_score_pct"] / 100.0, "0.0%")
    ]
    
    row_offset = 5
    for idx, (label, val, num_fmt) in enumerate(metrics):
        ws_summary.write(row_offset + idx, 0, label, fmt_metric_label)
        ws_summary.write(row_offset + idx, 1, val, fmt_metric_val)
        
        # Apply specific cell format to metric values
        if '%' in num_fmt:
            ws_summary.write_number(row_offset + idx, 1, val, fmt_cell_pct)
        else:
            ws_summary.write_number(row_offset + idx, 1, int(val), fmt_cell_number)

    # Write modifications log breakdown table
    ws_summary.write('A12', 'Resolution Breakdown by Step', workbook.add_format({'bold': True, 'font_name': 'Segoe UI', 'font_size': 12}))
    
    ws_summary.write('A13', 'Cleaning Step', fmt_header)
    ws_summary.write('B13', 'Issues Resolved', fmt_header)
    
    table_offset = 14
    for idx, row in log_counts.iterrows():
        ws_summary.write(table_offset + idx, 0, row['StepName'], fmt_cell)
        ws_summary.write_number(table_offset + idx, 1, int(row['ModificationsCount']), fmt_cell_number)
        
    # Add a visual chart on the summary sheet
    chart = workbook.add_chart({'type': 'bar'})
    chart.add_series({
        'categories': ['Executive Summary', 14, 0, 14 + len(log_counts) - 1, 0],
        'values':     ['Executive Summary', 14, 1, 14 + len(log_counts) - 1, 1],
        'fill':       {'color': COLOR_PRIMARY},
        'border':     {'color': '#ffffff'}
    })
    chart.set_title({'name': 'Modifications Count per Pipeline Step', 'name_font': {'name': 'Segoe UI', 'size': 11, 'bold': True}})
    chart.set_x_axis({'name': 'Resolution Count', 'name_font': {'name': 'Segoe UI', 'size': 9}})
    chart.set_y_axis({'name': 'Cleaning Step', 'name_font': {'name': 'Segoe UI', 'size': 9}})
    chart.set_legend({'position': 'none'})
    chart.set_size({'width': 480, 'height': 300})
    
    ws_summary.insert_chart('D5', chart)

    # Auto-fit column widths
    ws_summary.set_column('A:A', 35)
    ws_summary.set_column('B:B', 18)

    # ═══════════════════════════════════════════════════════
    # SHEET 2: Cleaned Orders
    # ═══════════════════════════════════════════════════════
    ws_clean = workbook.add_worksheet('Cleaned Orders')
    ws_clean.hide_gridlines(2)
    
    headers = list(clean_df.columns)
    for col_idx, h in enumerate(headers):
        ws_clean.write(0, col_idx, h, fmt_header)
        
    for row_idx, row in clean_df.iterrows():
        ws_clean.write(row_idx + 1, 0, row['OrderID'], fmt_cell_center)
        ws_clean.write(row_idx + 1, 1, row['CustomerName'], fmt_cell)
        ws_clean.write(row_idx + 1, 2, row['OrderDate'], fmt_cell_center)
        ws_clean.write(row_idx + 1, 3, row['Region'], fmt_cell_center)
        ws_clean.write(row_idx + 1, 4, row['Category'], fmt_cell_center)
        ws_clean.write_number(row_idx + 1, 5, int(row['Quantity']), fmt_cell_number)
        ws_clean.write_number(row_idx + 1, 6, float(row['UnitPrice']), fmt_cell_currency)
        ws_clean.write_number(row_idx + 1, 7, float(row['Sales']), fmt_cell_currency)
        
    # Auto-fit column widths dynamically
    for col_idx, col in enumerate(headers):
        max_len = max(clean_df[col].astype(str).map(len).max(), len(col)) + 4
        ws_clean.set_column(col_idx, col_idx, max(max_len, 12))

    # ═══════════════════════════════════════════════════════
    # SHEET 3: Modifications Log
    # ═══════════════════════════════════════════════════════
    ws_logs = workbook.add_worksheet('Audit Cleaning Log')
    ws_logs.hide_gridlines(2)
    
    log_headers = list(logs_df.columns)
    for col_idx, h in enumerate(log_headers):
        ws_logs.write(0, col_idx, h, fmt_header)
        
    for row_idx, row in logs_df.iterrows():
        ws_logs.write(row_idx + 1, 0, row['Step'], fmt_cell)
        ws_logs.write_number(row_idx + 1, 1, int(row['RowIndex']), fmt_cell_number)
        ws_logs.write(row_idx + 1, 2, row['ColumnName'], fmt_cell_center)
        ws_logs.write(row_idx + 1, 3, str(row['OriginalValue']), fmt_cell)
        ws_logs.write(row_idx + 1, 4, str(row['CleanedValue']), fmt_cell)
        ws_logs.write(row_idx + 1, 5, row['Description'], fmt_cell)

    # Set column widths
    ws_logs.set_column('A:A', 25)
    ws_logs.set_column('B:B', 12)
    ws_logs.set_column('C:C', 15)
    ws_logs.set_column('D:D', 20)
    ws_logs.set_column('E:E', 20)
    ws_logs.set_column('F:F', 45)

    workbook.close()
    print(f"[EXCEL] Generated final data cleaning Excel report at: {EXCEL_PATH}")

if __name__ == '__main__':
    main()
