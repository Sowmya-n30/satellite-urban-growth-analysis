"""
Report generation for urban growth analysis results.

Generates:
    - HTML/PDF summary reports
    - Markdown statistics tables
    - Urban growth statistics dashboards
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Vijayawada Urban Growth Analysis Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1 {{ color: #2C3E50; border-bottom: 3px solid #3498DB; padding-bottom: 10px; }}
  h2 {{ color: #2980B9; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #3498DB; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 12px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 16px;
                  box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin: 8px; display: inline-block; width: 180px; }}
  .metric-value {{ font-size: 2em; font-weight: bold; color: #E74C3C; }}
  .metric-label {{ color: #7F8C8D; font-size: 0.85em; }}
  .card-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
  footer {{ margin-top: 40px; color: #7F8C8D; font-size: 0.8em; border-top: 1px solid #ddd; padding-top: 10px; }}
</style>
</head>
<body>
<h1>🛰 Vijayawada Urban Growth Analysis Report</h1>
<p>Generated: {timestamp}</p>

<h2>📊 Summary</h2>
<div class="card-row">
  <div class="metric-card">
    <div class="metric-value">{urban_area_t1}</div>
    <div class="metric-label">Urban Area 2020 (km²)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{urban_area_t2}</div>
    <div class="metric-label">Urban Area 2023 (km²)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">+{urban_growth}</div>
    <div class="metric-label">Urban Growth (km²)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{growth_pct}%</div>
    <div class="metric-label">Growth Rate</div>
  </div>
</div>

<h2>🏙 Land Cover Statistics</h2>
<table>
  <tr><th>Class</th><th>2020 Area (km²)</th><th>2020 %</th><th>2023 Area (km²)</th><th>2023 %</th><th>Change (km²)</th></tr>
  {land_cover_rows}
</table>

<h2>🤖 Model Performance</h2>
<table>
  <tr><th>Metric</th><th>Non-Urban</th><th>Semi-Urban</th><th>Urban</th><th>Mean</th></tr>
  {model_metrics_rows}
</table>

<h2>📍 Analysis Details</h2>
<ul>
  <li><b>Region:</b> Vijayawada, Andhra Pradesh, India</li>
  <li><b>Data source:</b> Sentinel-2 MSI (10m resolution)</li>
  <li><b>Analysis period:</b> 2020 – 2023</li>
  <li><b>Model:</b> {model_name}</li>
  <li><b>Classes:</b> Non-Urban, Semi-Urban, Urban</li>
</ul>

<footer>
  Satellite Urban Growth Analysis System | Vijayawada | {timestamp}
</footer>
</body>
</html>
"""


def generate_html_report(
    change_stats: dict,
    model_metrics: dict = None,
    model_name: str = "U-Net",
    output_path: str = "results/reports/urban_growth_report.html",
):
    """
    Generate an HTML report from change detection and model metrics.

    Args:
        change_stats (dict): Output from postprocessing.compute_change_statistics().
        model_metrics (dict | None): Segmentation metrics dict.
        model_name (str): Model name for the report.
        output_path (str): Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Land cover table rows
    classes = ["Non-Urban", "Semi-Urban", "Urban"]
    lc_rows = ""
    for cls in classes:
        t1 = change_stats.get("t1_stats", {}).get(cls, {})
        t2 = change_stats.get("t2_stats", {}).get(cls, {})
        a1 = t1.get("area_km2", 0)
        p1 = t1.get("percentage", 0)
        a2 = t2.get("area_km2", 0)
        p2 = t2.get("percentage", 0)
        delta = round(a2 - a1, 3)
        sign = "+" if delta >= 0 else ""
        lc_rows += f"<tr><td>{cls}</td><td>{a1}</td><td>{p1}%</td><td>{a2}</td><td>{p2}%</td><td>{sign}{delta}</td></tr>\n"

    # Model metrics table rows
    mm_rows = ""
    if model_metrics:
        for metric in ["iou", "dice", "f1", "precision", "recall"]:
            per_class = model_metrics.get(f"per_class_{metric}", [0, 0, 0])
            mean_val = model_metrics.get(f"mean_{metric}", 0)
            mm_rows += (
                f"<tr><td>{metric.upper()}</td>"
                + "".join(f"<td>{v:.4f}</td>" for v in per_class)
                + f"<td><b>{mean_val:.4f}</b></td></tr>\n"
            )

    html = REPORT_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        urban_area_t1=change_stats.get("urban_area_t1_km2", "N/A"),
        urban_area_t2=change_stats.get("urban_area_t2_km2", "N/A"),
        urban_growth=change_stats.get("urban_growth_km2", "N/A"),
        growth_pct=change_stats.get("urban_growth_percent", "N/A"),
        land_cover_rows=lc_rows,
        model_metrics_rows=mm_rows,
        model_name=model_name,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved → %s", output_path)
    return output_path


def generate_markdown_report(
    change_stats: dict,
    model_metrics: dict = None,
    output_path: str = "results/reports/urban_growth_report.md",
):
    """
    Generate a Markdown summary report.

    Args:
        change_stats (dict): Change statistics dictionary.
        model_metrics (dict | None): Segmentation metrics.
        output_path (str): Output Markdown file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 🛰 Vijayawada Urban Growth Analysis Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## 📊 Urban Growth Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Urban Area 2020 | {change_stats.get('urban_area_t1_km2', 'N/A')} km² |",
        f"| Urban Area 2023 | {change_stats.get('urban_area_t2_km2', 'N/A')} km² |",
        f"| Urban Growth | +{change_stats.get('urban_growth_km2', 'N/A')} km² |",
        f"| Growth Rate | {change_stats.get('urban_growth_percent', 'N/A')}% |",
        f"| New Urban Area | {change_stats.get('new_urban_area_km2', 'N/A')} km² |",
        "",
        "## 🤖 Model Metrics",
    ]

    if model_metrics:
        lines += [
            "",
            "| Metric | Non-Urban | Semi-Urban | Urban | Mean |",
            "|--------|-----------|------------|-------|------|",
        ]
        for metric in ["iou", "dice", "f1"]:
            per_class = model_metrics.get(f"per_class_{metric}", [0, 0, 0])
            mean_val = model_metrics.get(f"mean_{metric}", 0)
            row = f"| {metric.upper()} | " + " | ".join(f"{v:.4f}" for v in per_class) + f" | **{mean_val:.4f}** |"
            lines.append(row)

    lines += [
        "",
        "## 📍 Analysis Details",
        "",
        "- **Region:** Vijayawada, Andhra Pradesh, India",
        "- **Data Source:** Sentinel-2 MSI (10m resolution)",
        "- **Analysis Period:** 2020 – 2023",
        "- **Classes:** Non-Urban, Semi-Urban, Urban",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report saved → %s", output_path)
    return output_path


def save_statistics_json(stats: dict, output_path: str = "results/reports/statistics.json"):
    """
    Save analysis statistics as JSON.

    Args:
        stats (dict): Statistics dictionary.
        output_path (str): Output JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Statistics JSON saved → %s", output_path)
