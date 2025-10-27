# 🚀 QueryPDF using RAG

A RAG (Retrieval-Augmented Generation) application for intelligent PDF document querying. This application combines local embeddings, vector search, and advanced LLMs to provide accurate, context-aware responses from your PDF documents.

## ✨ Features

- 📄 **PDF Document Ingestion** - Automatic chunking and indexing of PDF files
- 🔍 **Semantic Search** - Vector-based similarity search using Qdrant
- 🤖 **AI-Powered Q&A** - Query your documents using Groq's GPT-OSS-20B model
- ⚡ **Event-Driven Architecture** - Built with Inngest for reliable, observable workflows
- 💻 **Streamlit UI** - User-friendly web interface for document upload and querying
- 🏠 **Local Embeddings** - Uses SentenceTransformers for privacy-preserving embeddings
- 📊 **Rate Limiting & Throttling** - Built-in protection against abuse
- 🎯 **Source Tracking** - Track which documents inform each answer

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│  (Frontend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│  + Inngest      │
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
┌─────────┐ ┌──────────┐ ┌─────────┐
│ Sentence│ │  Qdrant  │ │  Groq   │
│Transfrmrs│ │  (Vector│ │  (LLM)  │
│ (Embed) │ │   DB)    │ │         │
└─────────┘ └──────────┘ └─────────┘
```

### Components

- **Streamlit App** (`streamlit_app.py`) - Upload PDFs and query documents
- **FastAPI Server** (`main.py`) - Backend server with Inngest functions
- **Data Loader** (`data_loader.py`) - PDF parsing and text chunking with SentenceTransformers
- **Vector Database** (`vector_db.py`) - Qdrant integration for semantic search
- **Custom Types** (`custom_types.py`) - Pydantic models for type safety

## 📋 Prerequisites

- Python 3.8+
- Docker Desktop (for running Qdrant locally)
- Groq API key ([Get one here](https://console.groq.com/))
- Optional: OpenAI API key (if switching to OpenAI models)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "ChatPDF using RAG"
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `fastapi` - Web framework
- `inngest` - Event orchestration
- `llama-index-core` & `llama-index-readers-file` - PDF processing
- `qdrant-client` - Vector database client
- `sentence-transformers` - Local embeddings (all-MiniLM-L6-v2)
- `groq` - LLM inference (GPT-OSS-20B)
- `streamlit` - Web UI
- `uvicorn` - ASGI server
- Other dependencies

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Required: Groq API Key for LLM inference
GROQ_API_KEY=your_groq_api_key_here

# Optional: For using OpenAI instead of Groq
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Override Inngest API endpoint
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

### 5. Start Qdrant (Vector Database)

**Option A: Using Docker (Recommended)**

```bash
docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

**Option B: Using Qdrant Cloud**

1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster and update `vector_db.py` with your cluster URL

## 🚀 Running the Application

### Start the FastAPI Server (with Inngest)

```bash
uvicorn main:app --reload --port 8000
```

This starts:
- FastAPI server on `http://localhost:8000`
- Inngest dev server on `http://127.0.0.1:8288`

### Start the Streamlit UI

In a new terminal:

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 📖 Usage

### 1. Upload PDFs

1. Open the Streamlit app
2. Click "Upload a PDF to Ingest"
3. Select a PDF file from your computer
4. Wait for ingestion to complete (you'll see a success message)

### 2. Query Documents

1. Scroll to "Ask a question about your PDFs"
2. Enter your question in the text input
3. Adjust the number of chunks to retrieve (default: 5)
4. Click "Ask"
5. View the AI-generated answer with source citations

### API Usage

You can also interact with the API directly:

**Ingest a PDF:**

```python
import inngest

client = inngest.Inngest(app_id="chatPDF_RAG_App")
await client.send(
    inngest.Event(
        name="rag/ingest_pdf",
        data={
            "pdf_path": "/path/to/document.pdf",
            "source_id": "unique-source-id"
        }
    )
)
```

**Query Documents:**

```python
await client.send(
    inngest.Event(
        name="rag/query_pdf_ai",
        data={
            "question": "What is the main topic of this document?",
            "top_k": 5
        }
    )
)
```

## 🔍 How It Works

### Ingestion Pipeline

1. **Load PDF** - PDF is parsed using LlamaIndex PDFReader
2. **Chunk Text** - Text is split into overlapping chunks (1000 chars, 200 overlap)
3. **Generate Embeddings** - Each chunk is embedded using SentenceTransformers (all-MiniLM-L6-v2, 384 dims)
4. **Store in Qdrant** - Embeddings are stored with metadata (source, text) in Qdrant

### Query Pipeline

1. **Embed Question** - User question is converted to embedding vector
2. **Similarity Search** - Qdrant finds top-k most similar chunks
3. **Context Retrieval** - Relevant chunks are assembled as context
4. **LLM Generation** - Groq's GPT-OSS-20B generates answer from context
5. **Return Results** - Answer + sources are returned to user

### Features & Protection

- **Throttling**: Max 2 PDF ingests per minute
- **Rate Limiting**: 1 ingest per source per 4 hours
- **Step Functions**: Observable, resumable workflows with Inngest
- **Error Handling**: Automatic retries and error recovery

## 📁 Project Structure

```
.
├── main.py                 # FastAPI server with Inngest functions
├── streamlit_app.py        # Streamlit UI for uploading & querying
├── data_loader.py          # PDF loading, chunking, and embeddings
├── vector_db.py            # Qdrant vector database integration
├── custom_types.py         # Pydantic models
├── requirements.txt        # Python dependencies
├── qdrant_storage/         # Local Qdrant data storage
├── uploads/                # Uploaded PDF files
└── README.md               # This file
```

## 🛠️ Configuration

### Embedding Model

Edit `data_loader.py` to change the embedding model:

```python
EMBED_MODEL = "all-MiniLM-L6-v2"  # Fast, 384 dimensions
# Alternatives: "all-mpnet-base-v2" (768 dims, slower but better)
```

### LLM Model

Edit `main.py` to change the Groq model:

```python
completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",  # Current model
    # Alternatives: "llama3-70b-8192", "mixtral-8x7b-32768"
    ...
)
```

### Chunking Strategy

Edit `data_loader.py`:

```python
splitter = SentenceSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

## 🔧 Troubleshooting

### Qdrant Connection Error

```bash
# Make sure Docker Desktop is running
docker ps

# Check if Qdrant is running
curl http://localhost:6333/health
```

### Embedding Model Download

First run will download the model (~100MB). This happens automatically and only once.

### API Key Issues

Verify your `.env` file has the correct API keys:

```bash
# Check environment variables
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GROQ_API_KEY'))"
```

### Port Already in Use

Change ports in `uvicorn` and `streamlit run` commands:

```bash
uvicorn main:app --reload --port 8001
streamlit run streamlit_app.py --server.port 8502
```

## 📊 Performance

- **Ingestion Speed**: ~50-100 pages per minute (depends on PDF complexity)
- **Query Latency**: ~2-5 seconds (includes embedding, search, and LLM generation)
- **Memory Usage**: ~500MB (base) + model size (~100MB for all-MiniLM-L6-v2)
- **Storage**: ~1-2KB per chunk in Qdrant

## 🚀 Deployment

### Production Checklist

- [ ] Set `is_production=True` in `main.py`
- [ ] Use Qdrant Cloud for production database
- [ ] Configure proper API keys in environment
- [ ] Set up monitoring and logging
- [ ] Configure rate limits for your use case
- [ ] Use production-grade embedding models
- [ ] Set up proper error handling and alerts

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🙏 Acknowledgments

- [Inngest](https://www.inngest.com/) - Event orchestration platform
- [Qdrant](https://qdrant.tech/) - Vector search engine
- [Groq](https://groq.com/) - High-performance LLM inference
- [SentenceTransformers](https://www.sbert.net/) - Semantic embeddings
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLMs

---

**Happy Document Querying! 📚✨**
