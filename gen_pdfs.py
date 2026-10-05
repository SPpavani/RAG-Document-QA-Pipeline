"""Generate a small synthetic PDF so the pipeline can be run end to end.

The company and all numbers are fictional. The facts are chosen to match the
questions in eval/questions.json.

Usage: python gen_pdfs.py
"""
from pathlib import Path

from fpdf import FPDF

OUT = Path("data/raw/pdfs/ai_strategy_2024.pdf")

SECTIONS = [
    (
        "Acme Corp - AI Strategy 2024 (synthetic sample)",
        "Acme Corp plans to invest 12 million dollars in artificial intelligence "
        "during 2024. The strategy rests on three pillars: customer support "
        "automation, demand forecasting, and internal knowledge search.",
    ),
    (
        "Customer Support Automation",
        "The support copilot targets a 30 percent reduction in average ticket "
        "handling time. The pilot starts in the Hyderabad support center in the "
        "second quarter of 2024.",
    ),
    (
        "Demand Forecasting",
        "Forecasting models will use three years of historical sales data. The "
        "target is to cut forecast error from 18 percent to 11 percent by the "
        "end of 2024.",
    ),
    (
        "Governance and Risk",
        "All AI systems must pass a privacy review before launch. Customer data "
        "is never sent to external models without anonymisation. The AI "
        "Governance Board, chaired by the Chief Data Officer, meets every month.",
    ),
    (
        "Team and Hiring",
        "The AI team will grow from 8 to 20 engineers in 2024. Priority roles "
        "are ML engineers, data engineers and an AI product manager.",
    ),
    (
        "Timeline",
        "Q1 2024: governance framework approved. Q2 2024: support copilot pilot. "
        "Q3 2024: forecasting models in production. Q4 2024: company-wide "
        "knowledge search launch.",
    ),
]


def main() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for title, body in SECTIONS:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 10, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=12)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, body)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Created {OUT}")


if __name__ == "__main__":
    main()
