import streamlit as st
from openai import OpenAI
import os
import io
import base64
import textwrap

# Optional libraries for document/image parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    A4 = None
    canvas = None

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = st.secrets["OPENAI_VECTOR_STORE_ID"] # set this in your env variables

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_CHOICES = {
    "Standard - GPT-5.4": "gpt-5.4",
    "Mittel - GPT-5.5": "gpt-5.5",
    "Maximal - GPT-5.5 Pro": "gpt-5.5-pro",
}

EXPORT_CHOICES = {
    "Nicht speichern": None,
    "Als TXT speichern": "txt",
    "Als Word-Dokument speichern": "docx",
    "Als PDF speichern": "pdf",
}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("CHECK-Ki Chatbot")

# File uploads: allow attaching photos and documents
uploaded_files = st.file_uploader("Fotos/Dokumente anhängen (optional)", type=None, accept_multiple_files=True)

def choose_model(model_label):
    return MODEL_CHOICES[model_label]


def build_conversation_text(messages):
    if not messages:
        return "Keine Nachrichten vorhanden."

    lines = []
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        model = message.get("model")
        model_suffix = f" ({model})" if model else ""
        lines.append(f"{role}{model_suffix}:")
        lines.append(message["content"])
        lines.append("")

    return "\n".join(lines).strip()


def build_docx_export(messages):
    if Document is None:
        return None

    doc = Document()
    doc.add_heading("CHECK-Ki Chatbot Conversation", level=1)
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        model = message.get("model")
        heading = f"{role} ({model})" if model else role
        doc.add_heading(heading, level=2)
        doc.add_paragraph(message["content"])

    output = io.BytesIO()
    doc.save(output)
    output.seek(0)
    return output.getvalue()


def build_pdf_export(messages):
    if A4 is None or canvas is None:
        return None

    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4)
    width, height = A4
    left_margin = 50
    top_margin = height - 50
    line_height = 14
    y = top_margin

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y, "CHECK-Ki Chatbot Conversation")
    y -= line_height * 2

    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        model = message.get("model")
        heading = f"{role} ({model})" if model else role

        if y < 70:
            pdf.showPage()
            y = top_margin

        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(left_margin, y, heading)
        y -= line_height
        pdf.setFont("Helvetica", 10)

        for paragraph in message["content"].splitlines() or [""]:
            for line in textwrap.wrap(paragraph, width=95) or [""]:
                if y < 50:
                    pdf.showPage()
                    y = top_margin
                    pdf.setFont("Helvetica", 10)
                pdf.drawString(left_margin, y, line)
                y -= line_height
            y -= 4

        y -= line_height

    pdf.save()
    output.seek(0)
    return output.getvalue()


def extract_text_from_file(uploaded_file):
    name = uploaded_file.name
    lower = name.lower()
    data = uploaded_file.getvalue()

    # Text-like files
    if lower.endswith(('.txt', '.md', '.csv')):
        try:
            return data.decode('utf-8')
        except Exception:
            return data.decode('latin-1', errors='ignore')

    # PDF
    if lower.endswith('.pdf') and PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = []
            for p in reader.pages:
                try:
                    text.append(p.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(text)
        except Exception:
            return f"(Konnte PDF {name} nicht extrahieren.)"

    # Images: try OCR if available
    if lower.endswith(('.png', '.jpg', '.jpeg')) and Image is not None:
        if pytesseract is not None:
            try:
                img = Image.open(io.BytesIO(data))
                text = pytesseract.image_to_string(img)
                return text or f"(Bild {name} ohne erkennbaren Text.)"
            except Exception:
                return f"(OCR für Bild {name} fehlgeschlagen.)"
        else:
            return f"(Bild {name} angehängt; OCR nicht verfügbar in der Umgebung.)"

    # Fallback: include filename and size
    return f"(Datei {name} angehängt — Typ unbekannt oder nicht unterstützte Extraktion. Größe: {len(data)} bytes)"

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

with st.container(border=True):
    model_col, save_col = st.columns(2)
    with model_col:
        selected_model_label = st.selectbox(
            "Model-Selector",
            options=list(MODEL_CHOICES.keys()),
            index=0,
            key="model_selector",
        )
    with save_col:
        selected_export_label = st.selectbox(
            "Chat speichern",
            options=list(EXPORT_CHOICES.keys()),
            index=0,
            disabled=not st.session_state.messages,
            key="export_selector",
        )

        selected_export = EXPORT_CHOICES[selected_export_label]
        if selected_export == "txt":
            st.download_button(
                "Download",
                data=build_conversation_text(st.session_state.messages).encode("utf-8"),
                file_name="check-ki-chat.txt",
                mime="text/plain",
            )
        elif selected_export == "docx":
            docx_export = build_docx_export(st.session_state.messages)
            if docx_export is not None:
                st.download_button(
                    "Download",
                    data=docx_export,
                    file_name="check-ki-chat.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.button("Download", disabled=True)
        elif selected_export == "pdf":
            pdf_export = build_pdf_export(st.session_state.messages)
            if pdf_export is not None:
                st.download_button(
                    "Download",
                    data=pdf_export,
                    file_name="check-ki-chat.pdf",
                    mime="application/pdf",
                )
            else:
                st.button("Download", disabled=True)

    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Frage stellen:")
        submitted = st.form_submit_button("Senden")

if submitted and prompt.strip():
    prompt = prompt.strip()
    previous_messages = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt})

    # -----------------------------
    # STEP 1: Vector Search
    # -----------------------------
    try:
        search_res = client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=prompt
        )
    except Exception as e:
        st.error(f"Search error: {e}")
        st.stop()

    # -----------------------------
    # STEP 2: Extract context
    # -----------------------------
    context_text = ""

    for hit in search_res.data:
        if hasattr(hit, "content") and hit.content:
            for part in hit.content:
                # OpenAI spec: part.type == "text"
                if part.type == "text" and hasattr(part, "text"):
                    context_text += part.text + "\n\n"

    if not context_text:
        context_text = "(Kein relevanter Kontext gefunden.)"

    # -----------------------------
    # Attachments: extract text from uploaded files
    # -----------------------------
    attachments_text = ""
    if uploaded_files:
        for f in uploaded_files:
            try:
                extracted = extract_text_from_file(f)
            except Exception as e:
                extracted = f"(Fehler beim Verarbeiten von {f.name}: {e})"
            attachments_text += f"--- Datei: {f.name} ---\n{extracted}\n\n"
        if not attachments_text:
            attachments_text = "(Hochgeladene Dateien enthalten keinen extrahierbaren Text.)"

    # -----------------------------
    # STEP 3: Ask GPT
    # -----------------------------
    final_prompt = f"""
You are a helpful and knowledgeable assistant.
Use the provided context (if meaningful) to answer the user's question.
If the context does not help, answer the question normally.

BISHERIGER CHAT:
{build_conversation_text(previous_messages)}

KONTEXT:
{context_text}

DATEIEN (aus Hochladen):
{attachments_text}

FRAGE:
{prompt}

ANTWORT:
"""

    try:
        selected_model = choose_model(selected_model_label)
        st.info(f"Verwendetes Modell: {selected_model}")
        response = client.responses.create(
            model=selected_model,
            instructions="You are a helpful assistant.",
            input=final_prompt
        )
        answer = response.output_text
    except Exception as e:
        st.error(f"GPT error: {e}")
        st.stop()

    # -----------------------------
    # STEP 4: Show assistant answer
    # -----------------------------
    st.session_state.messages.append({"role": "assistant", "content": answer, "model": selected_model})
    st.rerun()
