import streamlit as st
from openai import OpenAI
import os
import io
import base64

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

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = st.secrets["OPENAI_VECTOR_STORE_ID"] # set this in your env variables

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("CHECK-Ki Chatbot")

# File uploads: allow attaching photos and documents
uploaded_files = st.file_uploader("Fotos/Dokumente anhängen (optional)", type=None, accept_multiple_files=True)

# User prompt
prompt = st.text_input("Frage stellen:")


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

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

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

KONTEXT:
{context_text}

DATEIEN (aus Hochladen):
{attachments_text}

FRAGE:
{prompt}

ANTWORT:
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ]
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        st.error(f"GPT error: {e}")
        st.stop()

    # -----------------------------
    # STEP 4: Show assistant answer
    # -----------------------------
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
