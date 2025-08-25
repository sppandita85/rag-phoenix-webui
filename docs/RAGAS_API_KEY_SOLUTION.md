# RAGAs API Key Issue - Complete Solution Guide

## 🚨 **Problem**
RAGAs evaluation requires an OpenAI API key for LLM-based metrics like:
- **Faithfulness**: Evaluates if the answer is faithful to the retrieved context
- **Answer Relevancy**: Evaluates if the answer is relevant to the question

**Error Message**: `The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable`

## 🔑 **Solution 1: Add OpenAI API Key (Recommended)**

### **Step 1: Get OpenAI API Key**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-`)

### **Step 2: Set Environment Variable**

#### **Option A: Temporary (Current Session)**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

#### **Option B: Permanent (Add to Shell Profile)**
```bash
# For zsh (macOS/Linux)
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.zshrc
source ~/.zshrc

# For bash (Linux)
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### **Option C: Create .env File**
```bash
# Create .env file in project root
echo 'OPENAI_API_KEY=sk-your-api-key-here' > .env

# Load it in your shell
source .env
```

### **Step 3: Verify the Fix**
```bash
# Test RAGAs evaluation
python -c "
from core.rag_engine.evaluation import RAGEvaluator
evaluator = RAGEvaluator(project_name='test_project')
result = evaluator.evaluate_rag_response(
    question='What is machine learning?',
    answer='Machine learning is a subset of AI.',
    context='Machine learning is a subset of artificial intelligence.',
    metadata={'source': 'test'}
)
print('✅ RAGAs working:', result['scores'])
"
```

## 🚀 **Solution 2: Use Local Ollama Models (Advanced)**

### **Prerequisites**
- Ollama running with `llama3.2` model
- No dependency conflicts

### **Implementation**
This requires modifying the evaluation module to use Ollama LLM instead of OpenAI. However, due to version compatibility issues, this approach is more complex and may require additional development.

## 🔧 **Solution 3: Disable RAGAs Evaluation (Current State)**

### **What Happens Now**
- RAG system works normally
- Phoenix tracing works perfectly
- Web UI functions correctly
- **Only RAGAs evaluation is disabled**

### **Benefits**
- No API costs
- No external dependencies
- System remains fully functional
- Can be enabled later when needed

## 📊 **Current Status After Fix**

### **With OpenAI API Key:**
- ✅ RAG System: Working
- ✅ Phoenix Tracing: Working
- ✅ RAGAs Evaluation: Working with quality metrics
- ✅ Web UI: Working
- ✅ Performance Metrics: Available

### **Without OpenAI API Key (Current):**
- ✅ RAG System: Working
- ✅ Phoenix Tracing: Working
- ⚠️ RAGAs Evaluation: Disabled (no quality metrics)
- ✅ Web UI: Working
- ✅ Basic Performance Metrics: Available

## 🧪 **Testing Your Fix**

### **1. Check Environment Variable**
```bash
echo $OPENAI_API_KEY
# Should show: sk-your-api-key-here
```

### **2. Test RAGAs Import**
```bash
source venv/bin/activate
python -c "import ragas; print('✅ RAGAs version:', ragas.__version__)"
```

### **3. Test RAGAs Evaluation**
```bash
python -c "
from core.rag_engine.evaluation import RAGEvaluator
evaluator = RAGEvaluator(project_name='test_project')
result = evaluator.evaluate_rag_response(
    question='Test question',
    answer='Test answer',
    context='Test context',
    metadata={'source': 'test'}
)
print('✅ Evaluation result:', result)
"
```

### **4. Test API Endpoint**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Test RAGAs", "user_id": "test", "session_id": "test"}' \
  | python -m json.tool
```

Look for `"evaluation_available": true` and proper scores in the response.

## 🚨 **Troubleshooting**

### **API Key Not Working**
```bash
# Check if environment variable is set
env | grep OPENAI_API_KEY

# Restart your terminal/shell
# Or restart the API service
pkill -f uvicorn
source venv/bin/activate
uvicorn api.simple_api:app --host 0.0.0.0 --port 8000 --reload --loop asyncio
```

### **Still Getting API Key Error**
```bash
# Check if the variable is loaded in the Python process
python -c "import os; print('API Key:', os.getenv('OPENAI_API_KEY', 'NOT SET'))"

# Verify the key format (should start with 'sk-')
echo $OPENAI_API_KEY | head -c 5
```

### **RAGAs Import Errors**
```bash
# Reinstall RAGAs
pip uninstall ragas
pip install ragas==0.3.2

# Check for conflicts
pip check
```

## 💰 **Cost Considerations**

### **OpenAI API Costs**
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **RAGAs Evaluation**: Typically 100-500 tokens per evaluation
- **Estimated Cost**: $0.01-0.05 per 100 evaluations

### **Cost Optimization**
- Use GPT-3.5-turbo instead of GPT-4
- Limit evaluation frequency
- Cache evaluation results
- Use batch evaluation

## 🎯 **Recommendation**

### **For Development/Testing:**
- Use Solution 1 (OpenAI API Key)
- Get a free tier API key ($18 credit)
- Enable full RAGAs evaluation

### **For Production/Privacy:**
- Use Solution 2 (Ollama integration)
- Requires additional development
- No external API calls

### **For Immediate Use:**
- Use Solution 3 (Current state)
- System works perfectly
- Enable evaluation later when needed

---

## 📝 **Quick Fix Summary**

```bash
# 1. Get OpenAI API key from https://platform.openai.com/api-keys
# 2. Set environment variable
export OPENAI_API_KEY="sk-your-api-key-here"

# 3. Restart API service
pkill -f uvicorn
source venv/bin/activate
uvicorn api.simple_api:app --host 0.0.0.0 --port 8000 --reload --loop asyncio

# 4. Test with a query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Test", "user_id": "test", "session_id": "test"}'
```

**Result**: RAGAs evaluation will work, providing quality metrics for your RAG system! 🎉
