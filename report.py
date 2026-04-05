from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(filename, detections):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Tomato Disease Detection Report", styles['Title']))
    content.append(Spacer(1, 20))

    for d in detections:
        text = f"Disease: {d[0]} | Confidence: {d[1]:.2f}"
        content.append(Paragraph(text, styles['Normal']))
        content.append(Spacer(1, 10))

    doc.build(content)
