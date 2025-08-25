import os
import io
import json
import logging
import pickle
from flask import Flask, request, jsonify, redirect
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import fitz  # PyMuPDF
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
INDEX_FILE = "faiss.index"
CORPUS_FILE = "corpus.pkl"

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    logging.info("✅ FAISS index loaded")
else:
    index = faiss.IndexFlatL2(1536)
    logging.warning("⚠️ No FAISS index found, created new one")

if os.path.exists(CORPUS_FILE):
    with open(CORPUS_FILE, "rb") as f:
        corpus_texts = pickle.load(f)
else:
    corpus_texts = []

creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    logging.info("✅ Loaded credentials from token.json")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def add_to_faiss(chunks, batch_size=20):
    global index, corpus_texts
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_vecs = [d.embedding for d in resp.data]
        vectors.extend(batch_vecs)
        index.add(np.array(batch_vecs).astype("float32"))
        corpus_texts.extend(batch)
        logging.info(f"✅ Added {len(batch)} vectors to FAISS (batch {i//batch_size+1})")

    # Save index + corpus
    faiss.write_index(index, INDEX_FILE)
    with open(CORPUS_FILE, "wb") as f:
        pickle.dump(corpus_texts, f)

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "faiss_entries": index.ntotal,
        "chunks_indexed": len(corpus_texts),
        "credentials_loaded": creds is not None,
    })

@app.route("/authorize")
def authorize():
    flow = Flow.from_client_secrets_file(
        "credentials.json", SCOPES,
        redirect_uri="https://tamp-api.onrender.com/oauth2callback"
    )
    auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true")
    return redirect(auth_url)

@app.route("/oauth2callback")
def oauth2callback():
    flow = Flow.from_client_secrets_file(
        "credentials.json", SCOPES,
        redirect_uri="https://tamp-api.onrender.com/oauth2callback"
    )
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials
    with open("token.json", "w") as token:
        token.write(creds.to_json())
    return "✅ Authentication successful! You can now call /buildIndex."

@app.route("/buildIndex", methods=["POST"])
def build_index():
    global creds
    if not creds:
        return jsonify({"error": "Not authorized"}), 401

    service = build("drive", "v3", credentials=creds)
    results = service.files().list(
        q="mimeType='application/pdf'",
        pageSize=40
    ).execute()
    files = results.get("files", [])

    docs_processed = 0
    chunks_processed = 0

    for file in files:
        logging.info(f"📄 Downloading {file['name']}")
        request_dl = service.files().get_media(fileId=file["id"])
        fh = io.BytesIO()
        downloader = build("drive", "v3", credentials=creds)._http.request(request_dl.uri)
        fh.write(downloader[1])
        fh.seek(0)

        doc = fitz.open(stream=fh, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")

        chunks = chunk_text(text)
        add_to_faiss(chunks)
        docs_processed += 1
        chunks_processed += len(chunks)

    return jsonify({
        "message": "Index built and saved to FAISS",
        "documentsIndexed": docs_processed,
        "chunksIndexed": chunks_processed
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("q")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    q_vector = get_embedding(query)
    D, I = index.search(np.array([q_vector]).astype("float32"), 5)

    results = []
    for idx in I[0]:
        if idx < len(corpus_texts):
            results.append(corpus_texts[idx])

    return jsonify({"query": query, "results": results})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("q")
    if not query:
        return jsonify({"error": "Missing question"}), 400

    q_vector = get_embedding(query)
    D, I = index.search(np.array([q_vector]).astype("float32"), 5)

    results = []
    for idx in I[0]:
        if idx < len(corpus_texts):
            results.append(corpus_texts[idx])

    if not results:
        return jsonify({"answer": "No relevant context found."})

    context = "\n\n".join(results)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for transportation asset management."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = resp.choices[0].message["content"]
    return jsonify({"query": query, "answer": answer, "sources": results})

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
