# Medicare 360 Q&A System 🏥

A Retrieval-Augmented Generation (RAG) system built with Gradio that provides intelligent question-answering capabilities for Medicare 360 products and services.

## 🌟 Features

- **Smart Q&A System**: Uses FAISS vector search with semantic similarity matching
- **Typewriter Effect**: Streaming responses for better user experience  
- **Pre-built Examples**: Quick access to common Medicare 360 questions
- **Fast Retrieval**: Optimized FAISS indexing for quick document search
- **Custom UI**: Orange-themed interface with modern styling
- **Intelligent Filtering**: Strict answer filtering for relevant responses

## 🛠️ Technologies Used

- **Gradio**: Web interface framework
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embedding generation (all-MiniLM-L6-v2)
- **HuggingFace Transformers**: Language model for answer generation
- **Google Flan-T5**: Base model for text generation
- **PyTorch**: Deep learning framework

## 📁 Project Structure

```
question-answer-app/
├── ragqa/
│   └── gradio_rag.py          # Main application file
├── data/
│   └── *.txt                  # Q&A text files
├── vector_store.index         # FAISS vector index (generated)
├── vector_store_meta.json     # Metadata for vector store (generated)
└── README.md                  # This file
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd question-answer-app
   ```

2. **Install dependencies**
   ```bash
   pip install gradio faiss-cpu torch transformers sentence-transformers numpy
   ```

   For GPU support (optional):
   ```bash
   pip install faiss-gpu
   ```

3. **Prepare your data**
   - Place your Q&A text files in the `/data` folder
   - Format should be:
     ```
     Q. What is MPPP360?
     A. MPPP360 is a comprehensive Medicare plan management system...

     Q. What is A&G360?
     A. A&G360 is an analytics and growth platform...
     ```

## 🎯 Usage

### Running the Application

```bash
python ragqa/gradio_rag.py
```

The application will:
1. Load and process Q&A documents from the `/data` folder
2. Build or load the FAISS vector index
3. Launch the Gradio web interface
4. Open automatically in your browser at `http://localhost:7860`

### Using the Interface

1. **Ask Questions**: Type your Medicare 360 related questions in the input box
2. **Get Answers**: Click "Get Answer" to receive typewriter-style responses
3. **Use Examples**: Click on pre-built example questions for quick testing
4. **Browse Results**: Answers are retrieved from your document corpus using semantic search

## 📊 Sample Questions

The application comes with pre-built examples:
- "What is MPPP360?"
- "What is A&G360?"
- "Explain eEnroll360."
- "What does Revenue360 do?"

## ⚙️ Configuration

You can modify the following parameters in `gradio_rag.py`:

```python
# Vector store paths
VECTOR_STORE_PATH = "vector_store.index"
METADATA_PATH = "vector_store_meta.json"

# Data folder
DOCS_FOLDER = "/data"

# Search parameters
TOP_K = 5  # Number of documents to retrieve

# Model configuration
qa_model_name = 'google/flan-t5-base'
embed_model = 'sentence-transformers/all-MiniLM-L6-v2'
```

## 🔧 How It Works

1. **Document Processing**: Parses Q&A text files and extracts question-answer pairs
2. **Embedding Generation**: Creates vector embeddings for all questions using Sentence Transformers
3. **Vector Index**: Builds FAISS index for fast similarity search
4. **Query Processing**: 
   - Encodes user query into vector representation
   - Searches FAISS index for most similar questions
   - Filters results for strict relevance
   - Returns the best matching answer
5. **Response Display**: Streams answer with typewriter effect

## 📋 Data Format

Your text files should follow this format:

```
Q. Question 1 here?
A. Answer 1 here with detailed explanation.

Q. Question 2 here?
A. Answer 2 here with comprehensive details.

Q. Question 3 here?
A. Answer 3 here.
```

## 🎨 UI Customization

The interface uses custom CSS styling:
- Orange-themed buttons with hover effects
- Responsive design with rounded corners
- Auto-scrolling answer boxes
- Modern typography and spacing

## 🚨 Troubleshooting

### Common Issues

1. **No documents found**
   - Ensure `.txt` files are in the `/data` folder
   - Check file encoding is UTF-8
   - Verify Q&A format is correct

2. **Model loading errors**
   - Check internet connection for first-time model downloads
   - Ensure sufficient disk space for model files
   - Verify PyTorch installation

3. **FAISS index issues**
   - Delete `vector_store.index` and `vector_store_meta.json` to rebuild
   - Check file permissions in the project directory

### Performance Tips

- **GPU Usage**: Install `faiss-gpu` for faster similarity search
- **Model Selection**: Use larger models like `flan-t5-large` for better quality
- **Batch Processing**: Process documents in batches for large datasets

## 🔄 Updates and Maintenance

- **Rebuilding Index**: Delete vector store files to rebuild with new data
- **Model Updates**: Update model versions in the configuration
- **Adding Data**: Simply add new `.txt` files to the `/data` folder and restart

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the configuration options

---

**Made with ❤️ for Medicare 360 Q&A**
2. Review the configuration options
3. Ensure all dependencies are properly installed
4. Check system requirements

## Version History

- **v1.0**: Initial release with basic RAG functionality
- **v1.1**: Added streaming responses and improved TTS
- **v1.2**: Enhanced document processing and error handling

---

**Note**: This application is designed for Medicare information processing. Ensure your documents contain accurate and up-to-date Medicare information for best results.
