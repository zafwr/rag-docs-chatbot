"""Generate the fictional sample handbook used for demos and tests.

Run from the repo root:  python sample_docs/make_sample_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    (
        "Lumen Labs Employee Handbook (sample)",
        [
            "Lumen Labs is a fictional software company used to demonstrate this chatbot.",
            "Working hours are flexible. Core hours, when everyone should be reachable, are 11:00 to 16:00 on weekdays.",
            "Employees can work remotely up to three days per week. Fridays are remote by default.",
        ],
    ),
    (
        "Leave and holidays",
        [
            "Full-time employees receive 20 days of paid annual leave per year.",
            "Unused leave can be carried over to the next year, up to a maximum of 5 days.",
            "Sick leave does not require a doctor's note for absences of up to 2 consecutive days.",
            "Leave requests should be submitted in the HR portal at least 10 working days in advance.",
        ],
    ),
    (
        "Expenses and equipment",
        [
            "Every new employee receives a laptop and a one-time home office budget of 500 euros.",
            "Travel expenses are reimbursed within 14 days after the receipt is uploaded to the finance portal.",
            "Meals during business travel are covered up to 40 euros per day.",
            "Software licenses must be requested from the IT team rather than purchased personally.",
        ],
    ),
]


def main() -> None:
    output = Path(__file__).parent / "lumen-labs-handbook.pdf"
    styles = getSampleStyleSheet()
    story = []
    for index, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 12))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if index < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(output), pagesize=A4, title="Lumen Labs Employee Handbook (sample)").build(story)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
