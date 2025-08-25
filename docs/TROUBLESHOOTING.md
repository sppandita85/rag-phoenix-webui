# Troubleshooting Guide

## 🚨 Critical Issues & Solutions

### 1. Phoenix Tracing Not Working

**Problem**: Phoenix dashboard shows no traces, API logs show "Non-hexadecimal digit found" errors.

**Impact**: No monitoring of RAG system performance, missing observability data.

**Root Causes**:
1. Wrong import path for LlamaModel
2. Invalid trace ID format (not UUID)
3. Invalid span ID generation
4. Missing data directory

**Complete Solution**:

#### **Step 1: Fix Import Path**
**File**: `api/simple_api.py`
```python
# ❌ BROKEN - This import path doesn't exist
from llama_index.llama_store.llama_model import LlamaModel

# ✅ FIXED - Use the correct path
from core.vector_store.llama_model_simple import LlamaModelSimple as LlamaModel
```

#### **Step 2: Fix Trace ID Generation**
**File**: `api/simple_api.py`
```python
# ❌ BROKEN - Custom format not compatible with Phoenix
trace_id = f"rag_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.question) % 10000}"

# ✅ FIXED - Use proper UUID format
import uuid
trace_id = str(uuid.uuid4())
```

#### **Step 3: Fix Span ID Generation**
**File**: `core/rag_engine/integration.py`
```python
# ❌ BROKEN - Invalid span ID format
context = SpanContext(
    trace_id=trace_id,
    span_id=f"{trace_id}_span"  # Phoenix rejects this format
)

# ✅ FIXED - Generate proper UUID for span ID
import uuid
span_id = str(uuid.uuid4())
context = SpanContext(
    trace_id=trace_id,
    span_id=span_id  # Valid hexadecimal format
)
```

#### **Step 4: Fix Data Directory Path**
**File**: `core/vector_store/llama_model_simple.py`
```python
# ❌ BROKEN - Directory doesn't exist
def __init__(self, model_name='BAAI/bge-small-en-v1.5', data_directory='inputs/CV'):

# ✅ FIXED - Use correct directory path
def __init__(self, model_name='BAAI/bge-small-en-v1.5', data_directory='inputs/input_data'):
```

**Verification**:
```bash
# Check if directory exists
ls -la inputs/input_data/

# Should show your PDF files
# Example: CV-Data and AI Specialist _v_0.2.pdf
```

### 2. RAG System Not Available (503 Error)

**Problem**: API returns "503: RAG system not available" for all queries.

**Root Cause**: LlamaModel not initialized or documents not loaded.

**Solution**:
1. Ensure `inputs/input_data/` contains PDF documents
2. Check LlamaModel initialization in API logs
3. Verify document loading process

**Debug Commands**:
```bash
# Check document directory
ls -la inputs/input_data/

# Check API health
curl http://localhost:8000/health

# Check API logs for LlamaModel errors
# Look for "Warning: LlamaModel not available" messages
```

### 3. Web UI Not Responding

**Problem**: Web UI accepts questions but doesn't return responses.

**Root Cause**: Usually related to RAG system issues above.

**Solution**: Fix the underlying RAG system issues first, then verify Web UI connectivity.

**Debug Commands**:
```bash
# Check Web UI status
curl http://localhost:8080/

# Check API connectivity from Web UI
curl http://localhost:8000/health

# Verify both services are running
ps aux | grep -E "(uvicorn|open-webui)"
```

## 🔍 Diagnostic Commands

### Check Service Status
```bash
# Check all running services
ps aux | grep -E "(uvicorn|open-webui|phoenix)"

# Check port usage
lsof -i :8000  # RAG API
lsof -i :8080  # Web UI
lsof -i :6006  # Phoenix
```

### Check Phoenix Integration
```bash
# Test Phoenix client directly
source venv/bin/activate
python -c "
import phoenix as px
client = px.Client()
print('Phoenix version:', client.get_phoenix_version())
print('✅ Phoenix client working')
"

# Test tracing components
python -c "
from phoenix.trace.schemas import SpanContext
context = SpanContext(trace_id='1234567890abcdef', span_id='abcdef1234567890')
print('✅ SpanContext creation working')
"
```

### Check RAG System
```bash
# Test LlamaModel import
source venv/bin/activate
python -c "
from core.vector_store.llama_model_simple import LlamaModelSimple
print('✅ LlamaModel import working')
"

# Test document loading
python -c "
import os
doc_dir = 'inputs/input_data'
if os.path.exists(doc_dir):
    files = os.listdir(doc_dir)
    print(f'✅ Documents found: {files}')
else:
    print(f'❌ Directory not found: {doc_dir}')
"
```

## 📋 Prevention Checklist

### Before Starting Services
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] PostgreSQL running with pgvector extension
- [ ] Ollama models downloaded
- [ ] Documents placed in `inputs/input_data/`

### After Starting Services
- [ ] RAG API responds to health check (`/health` endpoint)
- [ ] Web UI accessible at http://localhost:8080
- [ ] Phoenix dashboard accessible at http://localhost:6006
- [ ] No import errors in API logs
- [ ] LlamaModel initializes successfully

### When Testing
- [ ] Web UI accepts questions
- [ ] RAG API processes queries
- [ ] Phoenix shows traces
- [ ] No "Non-hexadecimal digit found" errors
- [ ] Proper UUID trace IDs generated

## 🚀 Quick Recovery Steps

If Phoenix tracing stops working:

1. **Restart API**: `pkill -f uvicorn && sleep 3 && source venv/bin/activate && uvicorn api.simple_api:app --host 0.0.0.0 --port 8000 --reload --loop asyncio`

2. **Restart Phoenix**: `pkill -f phoenix && sleep 3 && source venv/bin/activate && python launch_phoenix.py &`

3. **Verify Tracing**: Make a test query and check Phoenix dashboard

4. **Check Logs**: Look for any new error messages

## 📚 Reference

- **Phoenix Version**: 11.26.0
- **Required Format**: UUID-compliant trace and span IDs
- **Error Indicator**: "Non-hexadecimal digit found" = ID format incompatible
- **Key Principle**: Phoenix requires strict OpenTelemetry compliance

---

**Remember**: The most common Phoenix tracing issue is ID format incompatibility. Always use `uuid.uuid4()` for trace and span IDs!
