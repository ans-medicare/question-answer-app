# gradio_rag_autoload_app_text_to_voice_slowtype.py

import os
from pathlib import Path
import gradio as gr
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
import tempfile
import pyttsx3

# ----------------------------
# CONFIG
# ----------------------------
DOCUMENT_FOLDER = "./data"       # Folder with all your PDFs/TXTs
FAISS_INDEX_PATH = "faiss_index"
MODEL_NAME = "google/flan-t5-base"  # Use base model for better detail
CHUNK_SIZE = 600                     # Larger chunks for complete coverage
CHUNK_OVERLAP = 100
TOP_K = 4                           # More chunks for comprehensive answers

# ----------------------------
# Global objects
# ----------------------------
vectorstore = None
qa_chain = None
# Removed global tts_engine - now created fresh for each request

# ----------------------------
# TTS Engine is now initialized per request to avoid hanging
# ----------------------------

# ----------------------------
# Load embeddings once
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Custom prompt for detailed answers
# ----------------------------
def get_detailed_prompt():
    from langchain.prompts import PromptTemplate
    
    template = """Based on the following context, provide a complete and comprehensive answer to the question.
    Include all relevant details, features, and information available in the context.
    If the context contains structured information (bullet points, lists, features), include ALL of them.
    Stay focused on the specific topic asked about.

    Context: {context}

    Question: {question}

    Complete Answer: """
    
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ----------------------------
# STEP 1: Load all documents from folder and create / load vectorstore
# ----------------------------
def initialize_vectorstore():
    global vectorstore, qa_chain
    os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

    # Load existing vectorstore if available
    if Path(FAISS_INDEX_PATH).exists():
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = []
        for fname in os.listdir(DOCUMENT_FOLDER):
            filepath = os.path.join(DOCUMENT_FOLDER, fname)
            if filepath.endswith(".pdf"):
                loader = PyMuPDFLoader(filepath)
            elif filepath.endswith(".txt"):
                loader = TextLoader(filepath)
            else:
                continue
            try:
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

        if not docs:
            raise ValueError(f"No documents found in {DOCUMENT_FOLDER}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n## ",      # Section breaks with headers (highest priority)
                "\n## ",        # Preserve section headers
                "\n\n### ",     # Subsection breaks
                "\n### ",       # Preserve subsection headers
                "\n\n",         # Paragraph breaks
                "\n- ",         # Bullet points
                "\n",           # Line breaks
                ". ",           # Sentences
                " "             # Words (last resort)
            ],
            keep_separator=True,
            length_function=len,
            is_separator_regex=False
        )
        docs_chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(docs_chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)

    # Load Flan-T5 and build QA chain
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    text_gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=150,             # Reduced for faster generation
        do_sample=True,                 # Enable sampling for more varied responses
        temperature=0.2,                # Lower temperature for more focused responses
        top_p=0.85,                     # Slightly more focused nucleus sampling
        pad_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 3,         # Reduced for faster retrieval
                "fetch_k": 20   # Reduced for faster selection
            }
        ),
        return_source_documents=False,   # Don't return sources for speed
        chain_type="stuff",              # Keep related information together
        chain_type_kwargs={
            "prompt": get_detailed_prompt()  # Custom prompt for structured answers
        }
    )

    print(f"✅ Vectorstore initialized with {len(vectorstore.index_to_docstore_id)} chunks.")


# ----------------------------
# STEP 2: Ask a question with streaming response
# ----------------------------
import time

def ask_question_streaming(query):
    """Generator function that yields text progressively for typewriter effect"""
    global qa_chain
    if qa_chain is None:
        yield "⚠️ QA system not initialized.", None
        return

    try:
        # Get the complete answer first
        response = qa_chain(query)
        answer = response["result"]
        
        # Check if we got a good structured answer
        if len(answer.strip()) < 30 or "don't have enough information" in answer.lower():
            # Try with more context-seeking query
            contextual_query = f"What information is available about {query}? Include all details and structure."
            response = qa_chain(contextual_query)
            answer = response["result"]
        
        # Stream the text letter by letter
        displayed_text = ""
        for char in answer:
            displayed_text += char
            time.sleep(0.02)  # Adjust speed here (0.02 = 50 chars/second)
            yield displayed_text, None
        
        # Generate audio file after streaming is complete
        audio_file = text_to_speech(answer)
        yield displayed_text, audio_file
        
    except Exception as e:
        error_msg = f"❌ Error processing question: {str(e)}"
        yield error_msg, None

def ask_question(query):
    """Non-streaming version for compatibility"""
    global qa_chain
    if qa_chain is None:
        return "⚠️ QA system not initialized.", None

    try:
        # Try direct query first to capture complete structured information
        response = qa_chain(query)
        answer = response["result"]
        
        # Check if we got a good structured answer
        if len(answer.strip()) < 30 or "don't have enough information" in answer.lower():
            # Try with more context-seeking query
            contextual_query = f"What information is available about {query}? Include all details and structure."
            response = qa_chain(contextual_query)
            answer = response["result"]
        
        # Generate audio file
        audio_file = text_to_speech(answer)
        
        return answer, audio_file
    except Exception as e:
        return f"❌ Error processing question: {str(e)}", None


# ----------------------------
# STEP 3: Text to Speech function (optimized and fixed)
# ----------------------------
def text_to_speech(text):
    try:
        # Limit text length for faster processing
        if len(text) > 800:
            # Take first 800 characters and find the last complete sentence
            truncated = text[:800]
            last_period = truncated.rfind('.')
            if last_period > 500:  # Ensure we have substantial content
                text = truncated[:last_period + 1]
            else:
                text = truncated + "..."
        
        # Initialize TTS engine fresh for each request to avoid hanging
        try:
            local_tts = pyttsx3.init()
            
            # Set properties for faster processing
            voices = local_tts.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        local_tts.setProperty('voice', voice.id)
                        break
                else:
                    # Use the first available voice
                    local_tts.setProperty('voice', voices[0].id)
            
            local_tts.setProperty('rate', 160)    # Slower, more comfortable speech rate
            local_tts.setProperty('volume', 0.9)
            
            # Create a temporary file for the audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            # Generate speech and save to file
            local_tts.save_to_file(text, temp_file.name)
            local_tts.runAndWait()
            
            # Clean up the TTS engine
            try:
                local_tts.stop()
            except:
                pass
            
            return temp_file.name
            
        except Exception as tts_error:
            print(f"❌ TTS Error: {tts_error}")
            return None
            
    except Exception as e:
        print(f"❌ Error in text_to_speech: {e}")
        return None


# ----------------------------
# STEP 4: Clear function
# ----------------------------
def clear_fields():
    return "", "", None


# ----------------------------
# Initialize vectorstore at startup
# ----------------------------
initialize_vectorstore()
# Removed TTS initialization - now done per request

# ----------------------------
# GRADIO UI with Streaming Support
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📚 Medicare : Question and Answers with Voice")
    gr.Markdown("*Ask detailed questions and get comprehensive answers in both text and voice with typewriter effect.*")

    query_input = gr.Textbox(
        label="Ask a question", 
        placeholder="Type your question here for a detailed explanation...",
        lines=2
    )
    
    with gr.Row():
        ask_button = gr.Button("Ask", variant="primary")
        clear_button = gr.Button("Clear")
    
    answer_output = gr.Textbox(
        label="Detailed Answer", 
        lines=8,
        max_lines=15
    )
    
    audio_output = gr.Audio(
        label="Voice Answer",
        type="filepath",
        interactive=False
    )

    # Use streaming for typewriter effect
    ask_button.click(
        fn=ask_question_streaming, 
        inputs=query_input, 
        outputs=[answer_output, audio_output],
        show_progress=True  # Shows progress indicator while streaming
    )
    clear_button.click(fn=clear_fields, inputs=[], outputs=[query_input, answer_output, audio_output])

demo.launch()