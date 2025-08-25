# Phoenix Tracing Quick Reference

## 🚨 Emergency Fix for Phoenix Tracing Issues

**When you see**: "Non-hexadecimal digit found" errors or no traces in Phoenix UI

**Quick Fix**: Update these 4 files with the exact code below

---

## 📝 File Fixes

### 1. `api/simple_api.py`
```python
# Add this import at the top
import uuid

# Replace trace ID generation in both endpoints
trace_id = str(uuid.uuid4())
```

### 2. `core/rag_engine/integration.py`
```python
# Add this import at the top
import uuid

# In trace_request method, replace:
span_id = str(uuid.uuid4())
context = SpanContext(
    trace_id=trace_id,
    span_id=span_id
)

# In trace_error method, replace:
span_id = str(uuid.uuid4())
context = SpanContext(
    trace_id=trace_id,
    span_id=span_id
)
```

### 3. `core/vector_store/llama_model_simple.py`
```python
# Fix the data directory path
def __init__(self, model_name='BAAI/bge-small-en-v1.5', data_directory='inputs/input_data'):
```

### 4. `api/simple_api.py` (Import Fix)
```python
# Replace the broken import
from core.vector_store.llama_model_simple import LlamaModelSimple as LlamaModel
```

---

## ✅ Verification

After applying fixes:

1. **Restart API**: `pkill -f uvicorn && sleep 3 && source venv/bin/activate && uvicorn api.simple_api:app --host 0.0.0.0 --port 8000 --reload --loop asyncio`

2. **Test Query**: Make a request to the API

3. **Check Phoenix**: http://localhost:6006/ should show traces

4. **Verify Trace IDs**: Should look like `6a5e20f6-d6a4-4364-941d-ec3689e21433`

---

## 🔑 Key Points

- **Always use `uuid.uuid4()`** for trace and span IDs
- **Phoenix requires hexadecimal format** - no custom strings
- **Error "Non-hexadecimal digit found"** = ID format incompatible
- **Both `trace_id` and `span_id`** must be valid UUIDs

---

## 📞 If Still Not Working

1. Check `docs/TROUBLESHOOTING.md` for detailed steps
2. Verify Phoenix is running: `ps aux | grep phoenix`
3. Check API logs for new error messages
4. Ensure all services are restarted after changes

---

**Remember**: This fix ensures Phoenix tracing works permanently! 🎯
