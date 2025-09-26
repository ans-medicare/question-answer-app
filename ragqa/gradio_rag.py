
import os
import faiss
import json
import torch
import glob
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np

# ================= CONFIG =================
VECTOR_STORE_PATH = "vector_store.index"
METADATA_PATH = "vector_store_meta.json"
DOCS_FOLDER = "/data"
TOP_K = 5  # initial search before strict filtering

# ================= LOAD MODELS =================
print("🔄 Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
qa_model_name = 'google/flan-t5-base'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name).to(device)
qa_model.eval()

# ================= HELPERS =================
def parse_qa_text(text):
    qas = []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for block in blocks:
        if block.startswith("Q.") and "A." in block:
            q_part, a_part = block.split("A.", 1)
            question = " ".join(q_part.replace("Q.", "").splitlines()).strip()
            answer = " ".join(a_part.splitlines()).strip()
            qas.append({"question": question, "answer": answer})
    return qas

def load_and_prepare_documents(folder_path):
    qas = []
    for file_path in glob.glob(f"{folder_path}/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        pairs = parse_qa_text(text)
        for qa in pairs:
            qas.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "source": os.path.basename(file_path)
            })
    print(f"✅ Loaded {len(qas)} Q&A pairs")
    return qas

def build_or_load_faiss(qas):
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = []
        if metadata:
            print(f"📂 Loaded FAISS index with {len(metadata)} Q&A pairs")
            index = faiss.read_index(VECTOR_STORE_PATH)
            return index, metadata

    questions = [qa["question"] for qa in qas]
    embeddings = embed_model.encode(questions, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, VECTOR_STORE_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(qas, f, indent=2)
    print(f"✅ Built FAISS index with {len(qas)} Q&A pairs")
    return index, qas

def retrieve_answers(query, index, metadata, top_k=TOP_K):
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])
    return results

def filter_strict_answers(query, retrieved):
    query_lower = query.lower()
    filtered = [r for r in retrieved if query_lower in r['question'].lower()]
    return filtered if filtered else retrieved

# ================= LOAD / BUILD VECTOR STORE =================
qas = load_and_prepare_documents(DOCS_FOLDER)
index, metadata = build_or_load_faiss(qas)

# ================= GRADIO TYPEWRITER FUNCTION =================
def qa_typewriter(user_question):
    retrieved = retrieve_answers(user_question, index, metadata)
    retrieved = filter_strict_answers(user_question, retrieved)
    top_answer = retrieved[0]['answer']

    output_text = ""
    for char in top_answer:
        output_text += char
        time.sleep(0.01)  # typing speed
        yield output_text

# ================= GRADIO BLOCKS APP =================
sample_questions = [
    "What is MPPP360?",
    "What is A&G360?",
    "Explain eEnroll360.",
    "What does Revenue360 do?"
]

with gr.Blocks(css="""
    body {background-color: #f2f2f7; font-family: 'Arial', sans-serif;}
    .gradio-container {max-width: 500px; margin:auto; padding:20px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1);}
    #get_answer_btn {
        background-color: orange !important;
        color: white !important;
        border: none !important;
        transition: background-color 0.3s;
        border-radius: 12px;
    }
    #get_answer_btn:hover {
        background-color: darkorange !important;
    }
    #answer_box textarea {
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        min-height: 180px !important;  
        max-height: 400px !important;
        overflow-y: auto !important;
        resize: vertical !important;
    }
""") as demo:

    gr.Markdown("## 📱 Medicare 360 Q&A")
    
    question_input = gr.Textbox(
        lines=2,
        placeholder="Ask a question about Medicare 360 products...",
        label="Your Question",
        elem_id="question_box"
    )
    answer_output = gr.Textbox(
        label="Answer",
        interactive=False,
        lines=6,
        elem_id="answer_box"
    )
    gr.Examples(
        examples=[[q] for q in sample_questions],
        inputs=question_input
    )
    submit_btn = gr.Button("Get Answer", elem_id="get_answer_btn")
    submit_btn.click(fn=qa_typewriter, inputs=question_input, outputs=answer_output, queue=True)


demo.launch()