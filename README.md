# Medicare Q&A Application with Voice Support

A comprehensive Question & Answer application built with RAG (Retrieval-Augmented Generation) technology that provides detailed Medicare information with both text and voice responses.

## Features

- 🤖 **Advanced RAG System**: Uses Flan-T5 model for intelligent question answering
- 📚 **Document Processing**: Supports PDF and TXT files for knowledge base
- 🔍 **Smart Retrieval**: FAISS vector database for efficient document search
- 🎯 **Streaming Response**: Typewriter effect for real-time text display
- 🔊 **Text-to-Speech**: Voice output with customizable speech settings
- 🌐 **Web Interface**: Clean, intuitive Gradio-based UI
- 📊 **Comprehensive Answers**: Structured responses with all relevant details

## Architecture

The application follows a modular architecture:

```
gradio_rag.py
├── Configuration Settings
├── Document Loading & Processing
├── Vector Store Management (FAISS)
├── Question Answering Chain (LangChain + Flan-T5)
├── Text-to-Speech Engine (pyttsx3)
└── Gradio Web Interface
```

## Requirements

### System Requirements
- Python 3.8+
- Windows (for pyttsx3 TTS engine)
- Minimum 4GB RAM (8GB+ recommended for better performance)
- GPU support optional (CPU mode supported)

### Python Dependencies
```
gradio>=4.0.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1
transformers>=4.30.0
torch>=2.0.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
pymupdf>=1.20.0
pyttsx3>=2.90
```

## Installation

1. **Clone or download the application files**

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

   Or install manually:
   ```powershell
   pip install gradio langchain langchain-community langchain-huggingface transformers torch faiss-cpu sentence-transformers pymupdf pyttsx3
   ```

## Setup

### 1. Document Preparation
- Create a `data` folder in the same directory as the script
- Add your Medicare documents (PDF or TXT files) to the `data` folder
- Supported formats: `.pdf`, `.txt`

### 2. Configuration (Optional)
You can modify these settings in the script:

```python
DOCUMENT_FOLDER = "./data"           # Path to your documents
FAISS_INDEX_PATH = "faiss_index"     # Vector database storage
MODEL_NAME = "google/flan-t5-base"   # Language model
CHUNK_SIZE = 600                     # Document chunk size
CHUNK_OVERLAP = 100                  # Overlap between chunks
```

## Usage

### Running the Application

1. **Start the application**:
   ```powershell
   python gradio_rag.py
   ```

2. **Access the web interface**:
   - Open your browser and go to the displayed URL (typically `http://127.0.0.1:7860`)

3. **Ask questions**:
   - Type your Medicare-related question in the text box
   - Click "Ask" to get both text and voice responses
   - Use "Clear" to reset the interface

### Example Questions
- "What are the Medicare prescription payment plan benefits?"
- "How does Medicare Part D work?"
- "What is the coverage gap in Medicare?"
- "Tell me about Medicare enrollment periods"

## Features in Detail

### 🤖 RAG System
- **Model**: Google Flan-T5-base for comprehensive text generation
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for fast similarity search
- **Chunking**: Intelligent document splitting with context preservation

### 🔍 Smart Document Processing
- **Multi-format Support**: PDF and TXT files
- **Intelligent Chunking**: Preserves document structure (headers, sections, bullet points)
- **Context Preservation**: Maintains relationships between related information

### 🎯 Response Generation
- **Streaming Output**: Real-time text display with typewriter effect
- **Comprehensive Answers**: Includes all relevant details from source documents
- **Fallback Logic**: Re-queries with different strategies if initial response is insufficient

### 🔊 Text-to-Speech
- **Voice Selection**: Automatically selects female voice when available
- **Customizable Settings**: Adjustable speech rate and volume
- **Audio Export**: Generates WAV files for each response
- **Error Handling**: Graceful fallback when TTS is unavailable

### 🌐 Web Interface
- **Clean Design**: Modern, responsive Gradio interface
- **Real-time Updates**: Streaming text with progress indicators
- **Audio Playback**: Integrated audio player for voice responses
- **User-friendly Controls**: Clear buttons and intuitive layout

## File Structure

```
question-answer-app/
├── ragqa/
│   ├── gradio_rag.py          # Main application file
│   ├── README.md              # This documentation
│   ├── requirements.txt       # Python dependencies
│   ├── data/                  # Document storage (create this)
│   │   ├── document1.pdf
│   │   └── document2.txt
│   └── faiss_index/          # Vector database (auto-generated)
│       ├── index.faiss
│       └── index.pkl
```

## Configuration Options

### Model Settings
```python
MODEL_NAME = "google/flan-t5-base"    # Language model
max_new_tokens = 150                  # Response length limit
temperature = 0.2                     # Response creativity
top_p = 0.85                         # Nucleus sampling
```

### Retrieval Settings
```python
CHUNK_SIZE = 600                     # Document chunk size
CHUNK_OVERLAP = 100                  # Overlap between chunks
k = 3                               # Number of chunks to retrieve
fetch_k = 20                        # Initial retrieval pool
```

### TTS Settings
```python
rate = 160                          # Speech rate (words per minute)
volume = 0.9                        # Voice volume (0.0-1.0)
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'gradio'"**
   - Install required packages: `pip install -r requirements.txt`
   - Ensure virtual environment is activated

2. **"No documents found in ./data"**
   - Create `data` folder and add PDF/TXT files
   - Check file permissions and formats

3. **TTS not working**
   - Ensure Windows TTS engine is available
   - Check audio drivers and system settings

4. **Slow performance**
   - Reduce `CHUNK_SIZE` and `max_new_tokens`
   - Use GPU if available (change `device=-1` to `device=0`)

5. **Memory issues**
   - Reduce document size or number of documents
   - Increase system RAM or use smaller model

### Performance Optimization

1. **For better speed**:
   - Use GPU acceleration
   - Reduce chunk size and retrieval parameters
   - Limit document collection size

2. **For better quality**:
   - Use larger models (flan-t5-large)
   - Increase chunk size and overlap
   - Add more relevant documents

## Advanced Usage

### Custom Prompts
Modify the `get_detailed_prompt()` function to customize response style:

```python
def get_detailed_prompt():
    template = """Your custom prompt template here...
    Context: {context}
    Question: {question}
    Answer: """
    return PromptTemplate(template=template, input_variables=["context", "question"])
```

### Adding New Document Types
Extend the `initialize_vectorstore()` function to support additional formats:

```python
elif filepath.endswith(".docx"):
    from langchain.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(filepath)
```

## API Reference

### Main Functions

- `initialize_vectorstore()`: Sets up document processing and vector storage
- `ask_question_streaming(query)`: Generates streaming text responses
- `ask_question(query)`: Non-streaming question answering
- `text_to_speech(text)`: Converts text to audio file
- `clear_fields()`: Resets the interface

### Configuration Variables

- `DOCUMENT_FOLDER`: Path to document storage
- `FAISS_INDEX_PATH`: Vector database location
- `MODEL_NAME`: Hugging Face model identifier
- `CHUNK_SIZE`: Document chunk size in characters
- `CHUNK_OVERLAP`: Overlap between consecutive chunks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check individual package licenses for dependencies.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Ensure all dependencies are properly installed
4. Check system requirements

## Version History

- **v1.0**: Initial release with basic RAG functionality
- **v1.1**: Added streaming responses and improved TTS
- **v1.2**: Enhanced document processing and error handling

---

**Note**: This application is designed for Medicare information processing. Ensure your documents contain accurate and up-to-date Medicare information for best results.
