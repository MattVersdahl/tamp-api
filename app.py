from flask import Flask, jsonify, request, redirect
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
import os
import logging
import fitz  # PyMuPDF
import datetime
from dotenv import load_dotenv
import sqlite3
from openai import OpenAI
import faiss
import numpy as np

# Load .env for local dev
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Google OAuth
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:5000/oauth2callback")
TOKEN_FILE = "token.json"

CREDENTIALS = None

# SQLite DB
DB_FILE = "tamps.db"

# FAISS index
FAISS_INDEX_FILE = "faiss.index"
embedding_dim = 1536  # OpenAI text-embedding-3-small

if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    logging.info("✅ Loaded FAISS index from file")
else:
    index = faiss.IndexFlatL2(embedding_dim)
    logging.warning("⚠️ No FAISS index found, created new one")

# Embedding client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Init SQLite
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            file_id TEXT,
            file_name TEXT,
            chunk_index INTEGER,
            chunk_text TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_chunks_to_db(file_id, file_name, chunks):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.executemany(
        "INSERT INTO chunks (file_id, file_name, chunk_index, chunk_text) VALUES (?, ?, ?, ?)",
        [(file_id, file_name, i, chunk) for i, chunk in enumerate(chunks)]
    )
    conn.commit()
    conn.close()

def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")

# ✅ NEW: Batched FAISS insert
def add_to_faiss(chunks, batch_size=20):
    total_vectors = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            vectors = [np.array(item.embedding, dtype="float32") for item in resp.data]
            index.add(np.vstack(vectors))
            total_vectors += len(vectors)
            logging.info(f"✅ Added {len(vectors)} vectors to FAISS (batch {i//batch_size + 1})")
        except Exception as e:
            logging.error(f"❌ Failed to embed batch {i//batch_size + 1}: {e}", exc_info=True)
    faiss.write_index(index, FAISS_INDEX_FILE)
    return total_vectors

def search_faiss(query, top_k=5):
    q_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        c.execute("SELECT file_name, chunk_text FROM chunks LIMIT 1 OFFSET ?", (idx,))
        row = c.fetchone()
        if row:
            results.append({"fileName": row[0], "snippet": row[1][:500]})
    conn.close()
    return results

def load_credentials():
    global CREDENTIALS
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
                CREDENTIALS = Credentials.from_authorized_user_info(data, SCOPES)
                logging.info("✅ Loaded credentials from token.json")
        except Exception as e:
            logging.error(f"❌ Failed to load credentials: {e}")
            CREDENTIALS = None
    else:
        logging.warning("⚠️ token.json not found")
        CREDENTIALS = None

load_credentials()

@app.route('/authorize', methods=['GET'])
def authorize():
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uris": [REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    authorization_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    code = request.args.get('code')
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uris": [REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)
    creds = flow.credentials

    with open(TOKEN_FILE, "w") as token:
        token.write(creds.to_json())

    global CREDENTIALS
    CREDENTIALS = creds

    return "✅ Authentication successful! You can now call /buildIndex."

@app.route('/status', methods=['GET'])
def status():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = c.fetchone()[0]
    conn.close()
    return jsonify({
        "status": "ok",
        "credentials_loaded": CREDENTIALS is not None and CREDENTIALS.valid,
        "chunks_indexed": total_chunks,
        "faiss_entries": index.ntotal,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    })

@app.route('/listTampsPdfs', methods=['GET'])
def list_tamps_pdfs():
    try:
        service = build("drive", "v3", credentials=CREDENTIALS)
        results = service.files().list(
            q="name contains 'TAMP' and mimeType='application/pdf'",
            fields="files(id, name)"
        ).execute()
        return jsonify(results.get("files", []))
    except Exception as e:
        logging.error(f"❌ Error listing PDFs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/buildIndex', methods=['POST'])
def build_index():
    try:
        if not CREDENTIALS or not CREDENTIALS.valid:
            return jsonify({"error": "Google credentials missing or invalid. Please re-authenticate at /authorize."}), 401

        service = build("drive", "v3", credentials=CREDENTIALS)
        results = service.files().list(
            q="name contains 'TAMP' and mimeType='application/pdf'",
            fields="files(id, name)"
        ).execute()
        files = results.get("files", [])

        if not files:
            return jsonify({"error": "No TAMP PDFs found in Google Drive"}), 404

        docs_indexed = 0
        chunks_indexed = 0

        for f in files:
            file_id = f["id"]
            name = f["name"]

            logging.info(f"📄 Downloading {name}")
            request_drive = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request_drive)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            fh.seek(0)
            doc = fitz.open(stream=fh, filetype="pdf")

            text = ""
            for page in doc:
                try:
                    text += page.get_text()
                except Exception as e:
                    logging.warning(f"⚠️ Failed to read page in {name}: {e}")

            chunks = chunk_text(text)
            save_chunks_to_db(file_id, name, chunks)
            chunks_indexed += len(chunks)
            docs_indexed += 1

            # Add to FAISS with batching
            add_to_faiss(chunks)

        return jsonify({
            "documentsIndexed": docs_indexed,
            "chunksIndexed": chunks_indexed,
            "message": "Index built and saved to FAISS"
        })

    except Exception as e:
        logging.error("❌ Fatal error in buildIndex", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get("q")
        if not query:
            return jsonify({"error": "query required"}), 400
        results = search_faiss(query)
        return jsonify({"query": query, "results": results})
    except Exception as e:
        logging.error("❌ Fatal error in ask", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/checkOpenAI', methods=['GET'])
def check_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set in environment"}), 500

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello from the TAMP API test!"}],
            max_tokens=20
        )
        return jsonify({"success": True, "message": resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
