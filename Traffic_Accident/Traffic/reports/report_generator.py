"""
reports/report_generator.py
Generates downloadable CSV and PDF reports for predictions.
PDF generation uses reportlab (lightweight, no external binary dependency).
"""

import io
import sys
import os
import datetime as dt

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_single_prediction_pdf(
    user_full_name: str,
    prediction_dict: dict,
    record: dict,
    date: dt.date,
    time_: dt.time,
    city: str,
) -> bytes:
    """Builds a single-prediction PDF report and returns raw bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], textColor=colors.HexColor("#1f3b57")
    )
    story = []

    story.append(Paragraph(config.APP_NAME, title_style))
    story.append(Paragraph("Accident Severity Prediction Report", styles["Heading2"]))
    story.append(Spacer(1, 12))

    meta_table = Table(
        [
            ["Generated For", user_full_name],
            ["Generated At", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Prediction Date", str(date)],
            ["Prediction Time", str(time_)],
            ["City", city],
        ],
        colWidths=[5 * cm, 10 * cm],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Prediction Inputs", styles["Heading3"]))
    input_rows = [["Field", "Value"]] + [[k, str(v)] for k, v in record.items()]
    input_table = Table(input_rows, colWidths=[6 * cm, 9 * cm])
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(input_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Prediction Result", styles["Heading3"]))
    result_rows = [
        ["Predicted Severity", prediction_dict["severity"]],
        ["Confidence Score", f"{prediction_dict['confidence'] * 100:.1f}%"],
        ["Model Used", prediction_dict["model_used"]],
    ]
    result_table = Table(result_rows, colWidths=[6 * cm, 9 * cm])
    sev_color = colors.HexColor(config.SEVERITY_COLORS.get(prediction_dict["severity"], "#999999"))
    result_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (1, 0), (1, 0), sev_color),
                ("TEXTCOLOR", (1, 0), (1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 16))

    if prediction_dict.get("shap_top_features"):
        story.append(Paragraph("Top Contributing Factors (Explainable AI)", styles["Heading3"]))
        shap_rows = [["Feature", "Relative Impact"]] + [
            [f["feature"], f"{f['impact']:.4f}"] for f in prediction_dict["shap_top_features"]
        ]
        shap_table = Table(shap_rows, colWidths=[7 * cm, 8 * cm])
        shap_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(shap_table)
        story.append(Spacer(1, 16))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def build_history_pdf(user_full_name: str, df: pd.DataFrame) -> bytes:
    """Builds a summary PDF for a filtered prediction history table."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(config.APP_NAME, styles["Title"]),
        Paragraph("Prediction History Report", styles["Heading2"]),
        Paragraph(f"Generated for: {user_full_name}", styles["Normal"]),
        Paragraph(f"Generated at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Paragraph(f"Total Records: {len(df)}", styles["Normal"]),
        Spacer(1, 12),
    ]

    display_cols = ["Date", "City", "Predicted Severity", "Confidence", "Model Used"]
    display_cols = [c for c in display_cols if c in df.columns]
    table_data = [display_cols] + df[display_cols].astype(str).values.tolist()

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    buf.seek(0)
    return buf.read()
