from flask import Flask, jsonify, request, redirect
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io, os, json, logging, fitz, datetime
import sqlite3
from dotenv import load_dotenv
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

# Embeddings + FAISS setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_DIM = 1536  # ada-002 / gpt-4o-mini embeddings
INDEX_FILE = "tamps.index"
META_FILE = "tamps_meta.json"

index = None
metadata = []

def embed_text(text):
    """Get embedding vector from OpenAI"""
    emb = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    ).data[0].embedding
    return np.array(emb, dtype="float32")

def load_faiss():
    """Load FAISS index + metadata if available"""
    global index, metadata
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        logging.info("✅ Loaded FAISS index")
    else:
        index = faiss.IndexFlatL2(EMBED_DIM)
        logging.info("⚠️ No FAISS index found, created new one")
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
        logging.info(f"✅ Loaded metadata: {len(metadata)} entries")
    else:
        metadata = []

# Load FAISS at startup
load_faiss()

# ---------------- DB Setup (still store raw chunks for backup/debug) ----------------
DB_FILE = "tamps.db"

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

# ---------------- Google Auth ----------------
def load_credentials():
    """Always load credentials from token.json"""
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

# ---------------- Flask Routes ----------------
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
        "faiss_entries": len(metadata),
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
        logging.error(f"❌ Error listing PDFs: {e}")
        return jsonify({"error": str(e)}), 500

def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

@app.route('/buildIndex', methods=['POST'])
def build_index():
    global index, metadata
    try:
        if not CREDENTIALS or not CREDENTIALS.valid:
            return jsonify({"error": "Google credentials missing or invalid. Please re-authenticate at /authorize."}), 401

        # reset
        index = faiss.IndexFlatL2(EMBED_DIM)
        metadata = []

        service = build("drive", "v3", credentials=CREDENTIALS)
        results = service.files().list(
            q="name contains 'TAMP' and mimeType='application/pdf'",
            fields="files(id, name)"
        ).execute()
        files = results.get("files", [])

        for f in files:
            file_id = f["id"]
            name = f["name"]

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

            # embed + add to FAISS
            for chunk in chunks:
                emb = embed_text(chunk)
                index.add(np.array([emb]))
                metadata.append({
                    "fileId": file_id,
                    "fileName": name,
                    "snippet": chunk[:500]
                })

        # save FAISS + metadata
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "w") as f:
            json.dump(metadata, f)

        return jsonify({
            "documentsIndexed": len(files),
            "chunksIndexed": len(metadata),
            "message": "Index built and saved to FAISS"
        })
    except Exception as e:
        logging.error(f"❌ Fatal error in buildIndex: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get("q")
    if not query:
        return jsonify({"error": "query required"}), 400

    # embed query
    q_emb = embed_text(query)

    # search in FAISS
    D, I = index.search(np.array([q_emb]), k=5)  # top 5
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return jsonify({"query": query, "results": results})

@app.route('/checkOpenAI', methods=['GET'])
def check_openai():
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500
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
