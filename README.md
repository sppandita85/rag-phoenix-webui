# NorthBayPoC RAG System

A complete **RAG (Retrieval-Augmented Generation)** system with Web UI integration, built for processing PDF documents and providing intelligent question-answering capabilities.

## 🚀 Features

- **Document Processing**: PDF parsing and intelligent chunking using docling
- **Vector Storage**: PostgreSQL with pgvector for efficient similarity search
- **RAG Engine**: LlamaIndex-powered retrieval and generation
- **Web Interface**: Modern chat UI using Open WebUI
- **API Backend**: FastAPI with OpenAI-compatible endpoints
- **Local Processing**: Ollama integration for privacy-focused AI
- **RAG Evaluation**: RAGAs-powered response quality assessment
- **Monitoring**: Phoenix integration for comprehensive RAG system observability

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Files   │    │  Document       │    │  Vector Store   │
│   (PDF/DOCX)    │───▶│  Processor      │───▶│  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   RAG Engine    │    │   Web UI        │
                       │  (LlamaIndex)   │◄──▶│  (Open WebUI)   │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
NorthBayPoC/
├── core/                    # Core RAG functionality
│   ├── document_processor/  # PDF parsing, chunking
│   ├── vector_store/        # Database operations
│   └── rag_engine/          # RAG query processing
├── api/                     # FastAPI backend
│   └── simple_api.py        # Main API server
├── webui/                   # Web UI integration
├── config/                  # Configuration files
├── docs/                    # Documentation
├── tests/                   # Test files
├── inputs/                  # Document input directory
├── requirements.txt         # Python dependencies
├── main.py                  # Project entry point
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites

- **Python**: 3.12+
- **PostgreSQL**: 13+ with pgvector extension
- **Ollama**: For local LLM inference

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NorthBayPoC
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama models**
   ```bash
   ollama pull llama3.2
   ollama pull qllama/bge-large-en-v1.5
   ```

5. **Configure PostgreSQL**
   ```sql
   CREATE EXTENSION vector;
   ```

## 🚀 Quick Start

### 1. Start the RAG API
```bash
source venv/bin/activate
uvicorn api.simple_api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Web UI
```bash
source venv/bin/activate
open-webui serve --port 8080
```

### 3. Access the System
- **Web UI**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Phoenix Dashboard**: http://localhost:6006 (optional monitoring)

### 3. Start Phoenix Monitoring (Optional)
```bash
source venv/bin/activate
python launch_phoenix.py
```

## 📚 Usage

### Adding Documents
Place PDF files in the `inputs/input_data/` directory. The system will automatically process them when you make your first query.

### Making Queries
1. Open the Web UI at http://localhost:8080
2. Start a new chat
3. Ask questions about your documents
4. Get AI-powered responses based on your content

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# RAG query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "user_id": "user", "session_id": "session"}'

# OpenAI-compatible endpoint
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Monitoring and Evaluation
```bash
# Launch Phoenix monitoring (optional)
python launch_phoenix.py

# Check evaluation status
curl http://localhost:8000/models | python -m json.tool
```

## ⚙️ Configuration

### Web UI Settings
1. **Ollama Configuration**:
   - Base URL: `http://localhost:11434`
   - Enable Ollama API: ✅

2. **OpenAI API Configuration**:
   - Base URL: `http://localhost:8000`
   - API Key: (leave empty for local RAG API)

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ragdb

# Ollama
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

## 🧪 Testing

Run the test suite:
```bash
cd tests
pytest
```

## 📚 Documentation

- **README.md**: This file - project overview and setup
- **docs/TROUBLESHOOTING.md**: Comprehensive troubleshooting guide for common issues
- **docs/PHOENIX_TRACING_QUICK_REFERENCE.md**: Emergency fix for Phoenix tracing issues
- **docs/RAGAS_API_KEY_SOLUTION.md**: Complete guide for fixing RAGAs API key issues
- **API Documentation**: Available at http://localhost:8000/docs when API is running

## 📊 RAG Evaluation & Monitoring

### RAGAs Metrics
The system automatically evaluates RAG responses using:
- **Faithfulness**: How well the answer follows the provided context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Relevancy**: How relevant the retrieved context is
- **Context Recall**: How well the context covers the question
- **Answer Correctness**: Accuracy of the generated answer
- **Answer Similarity**: Consistency with expected responses

### Phoenix Monitoring
- **Real-time Tracing**: Track RAG queries and responses
- **Performance Metrics**: Monitor response times and quality
- **System Health**: Comprehensive system status monitoring
- **Dashboard**: Visual interface at http://localhost:6006

## 🔧 Troubleshooting

### Phoenix Tracing Issues

If you encounter "Non-hexadecimal digit found" errors or don't see traces in Phoenix UI, follow these steps:

#### **Problem Symptoms:**
- ❌ Phoenix dashboard shows no traces
- ❌ API logs show "Failed to trace request to Phoenix: Non-hexadecimal digit found"
- ❌ Web UI queries not appearing in Phoenix monitoring

#### **Root Causes:**
1. **Wrong Import Path**: API trying to import from non-existent `llama_index.llama_store`
2. **Invalid Trace ID Format**: Using custom formatted strings instead of UUIDs
3. **Invalid Span ID Generation**: Appending `_span` to trace IDs
4. **Missing Data Directory**: LlamaModel looking for wrong input path

#### **Complete Solution:**

##### **1. Fix Import Path in `api/simple_api.py`:**
```python
# Before (BROKEN):
from llama_index.llama_store.llama_model import LlamaModel

# After (FIXED):
from core.vector_store.llama_model_simple import LlamaModelSimple as LlamaModel
```

##### **2. Fix Trace ID Generation:**
```python
# Before (BROKEN):
trace_id = f"rag_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.question) % 10000}"

# After (FIXED):
import uuid
trace_id = str(uuid.uuid4())
```

##### **3. Fix Span ID Generation in `core/rag_engine/integration.py`:**
```python
# Before (BROKEN):
context = SpanContext(
    trace_id=trace_id,
    span_id=f"{trace_id}_span"  # Invalid format
)

# After (FIXED):
import uuid
span_id = str(uuid.uuid4())
context = SpanContext(
    trace_id=trace_id,
    span_id=span_id  # Proper UUID format
)
```

##### **4. Fix Data Directory Path in `core/vector_store/llama_model_simple.py`:**
```python
# Before (BROKEN):
def __init__(self, model_name='BAAI/bge-small-en-v1.5', data_directory='inputs/CV'):

# After (FIXED):
def __init__(self, model_name='BAAI/bge-small-en-v1.5', data_directory='inputs/input_data'):
```

#### **Verification Steps:**
1. **Check Phoenix Dashboard**: http://localhost:6006/ should show traces
2. **Check API Logs**: No more "Non-hexadecimal digit found" errors
3. **Test Web UI**: Queries should generate proper trace IDs (UUID format)
4. **Verify RAG System**: Documents should be processed and queries answered

#### **Key Technical Details:**
- **Phoenix Version**: 11.26.0
- **Required Format**: Both `trace_id` and `span_id` must be valid hexadecimal UUIDs
- **Phoenix Expects**: OpenTelemetry-compliant Span objects
- **Error Indicator**: "Non-hexadecimal digit found" means ID format is incompatible

### Other Common Issues

#### **RAG System Not Available (503 Error)**
- **Cause**: LlamaModel not initialized or documents not loaded
- **Solution**: Check if `inputs/input_data/` contains PDF documents
- **Verify**: Run `ls -la inputs/input_data/` to confirm files exist

#### **Import Errors**
- **Cause**: Missing dependencies or wrong import paths
- **Solution**: Ensure virtual environment is activated and dependencies installed
- **Verify**: Run `pip list | grep -E "(llama|phoenix|ragas)"`

#### **Phoenix Connection Issues**
- **Cause**: Phoenix service not running
- **Solution**: Start Phoenix with `python launch_phoenix.py`
- **Verify**: Check http://localhost:6006/ is accessible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Review the documentation
- Open an issue on GitHub

---

**Built with ❤️ using FastAPI, LlamaIndex, and Open WebUI**
