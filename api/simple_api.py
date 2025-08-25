#!/usr/bin/env python3
"""
Enhanced FastAPI RAG API Bridge for Open WebUI Integration
Now uses real LlamaModel to query ingested documents from PostgreSQL
Includes OpenAI-compatible endpoints for Open WebUI compatibility
"""

import os
import sys
import time
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid

# Disable uvicorn's uvloop to prevent conflicts with RAGAs
os.environ["UVICORN_USE_UVLOOP"] = "0"

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the real LlamaModel
try:
    from core.vector_store.llama_model_simple import LlamaModelSimple as LlamaModel
    LLAMA_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LlamaModel not available: {e}")
    LLAMA_MODEL_AVAILABLE = False

# Import RAG evaluation and monitoring
try:
    from core.rag_engine.integration import rag_integration
    RAG_EVALUATION_AVAILABLE = True
    print("✅ RAG evaluation and monitoring system loaded")
except ImportError as e:
    print(f"Warning: RAG evaluation not available: {e}")
    RAG_EVALUATION_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced RAG API Bridge",
    description="Advanced RAG API with OpenAI compatibility",
    version="2.0.0"
)

# Add CORS middleware for Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
llama_model = None

def _ensure_llama_model():
    """Lazily initialize LlamaModel if not already initialized."""
    global llama_model
    
    if llama_model is None and LLAMA_MODEL_AVAILABLE:
        try:
            print("🚀 Lazy initializing LlamaModel...")
            llama_model = LlamaModel()
            print("✅ LlamaModel instantiated successfully")
            
            # Create the index and query engine
            print("🔧 Creating index and query engine...")
            llama_model.create_index()
            print("✅ Index and query engine created successfully")
            
            return True
        except Exception as e:
            print(f"❌ Failed to initialize LlamaModel: {e}")
            print(f"   Error details: {type(e).__name__}: {str(e)}")
            llama_model = None
            return False
    
    return llama_model is not None

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    user_id: str = "default"
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    context: str
    sources: list
    confidence: float
    model_used: str
    trace_id: str
    performance_metrics: dict

# OpenAI-compatible models
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "llama3.2"
    messages: list[OpenAIMessage]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"

class OpenAIResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: dict

class EvaluationRequest(BaseModel):
    question: str
    expected_answer: str
    user_id: str = "default"

class EvaluationResponse(BaseModel):
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    overall_score: float
    feedback: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG API Bridge on startup."""
    # Don't initialize LlamaModel during startup - do it lazily when first needed
    print("⏳ LlamaModel will be initialized on first use (lazy loading)")
    llama_model = None
    
    # Start RAG monitoring if available
    if RAG_EVALUATION_AVAILABLE:
        try:
            rag_integration.start_monitoring()
            print("✅ RAG monitoring system started")
        except Exception as e:
            print(f"⚠️ Failed to start RAG monitoring: {e}")
    else:
        print("⚠️ RAG monitoring not available")

@app.get("/")
async def root():
    """Health check endpoint."""
    monitoring_status = {}
    if RAG_EVALUATION_AVAILABLE:
        monitoring_status = rag_integration.get_system_status()
    
    return {
        "status": "healthy",
        "message": "Enhanced RAG API Bridge is running",
        "llama_model_loaded": llama_model is not None,
        "rag_evaluation_available": RAG_EVALUATION_AVAILABLE,
        "monitoring_status": monitoring_status,
        "endpoints": ["/chat", "/chat/completions", "/v1/chat/completions", "/evaluate", "/models", "/health", "/docs"]
    }

# OpenAI-compatible endpoint - this is what Open WebUI expects
@app.post("/chat/completions", response_model=OpenAIResponse)
async def openai_chat_completions(request: OpenAIRequest):
    """OpenAI-compatible chat completions endpoint."""
    global llama_model
    
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    
    try:
        # Extract user message from OpenAI format
        user_message = ""
        for message in request.messages:
            if message.role == "user":
                user_message = message.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        print(f"🔍 OpenAI-compatible request: {user_message}")
        print(f"📊 Trace ID: {trace_id}")
        
        # Trace the request to Phoenix regardless of outcome
        if RAG_EVALUATION_AVAILABLE:
            try:
                rag_integration.trace_request(
                    trace_id=trace_id,
                    question=user_message,
                    endpoint="/chat/completions",
                    user_id="openai_user",
                    session_id=trace_id
                )
            except Exception as trace_error:
                print(f"⚠️ Phoenix tracing failed: {trace_error}")
        
        if llama_model is not None or _ensure_llama_model():
            # Use real LlamaModel for document queries
            try:
                print("📚 Querying real documents from PostgreSQL...")
                
                answer = llama_model.query(user_message)
                
                # Calculate performance metrics
                end_time = time.time()
                response_time = end_time - start_time
                
                # Create OpenAI-compatible response
                response = OpenAIResponse(
                    id=f"chatcmpl-{trace_id}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        OpenAIChoice(
                            message=OpenAIMessage(
                                role="assistant",
                                content=str(answer)
                            )
                        )
                    ],
                    usage={
                        "prompt_tokens": len(user_message.split()),
                        "completion_tokens": len(str(answer).split()),
                        "total_tokens": len(user_message.split()) + len(str(answer).split())
                    }
                )
                
                print(f"✅ OpenAI-compatible response generated from documents")
                print(f"📊 Performance: {response_time:.2f}s response time")
                
                return response
                
            except Exception as e:
                print(f"⚠️ LlamaModel query failed: {e}")
                
                # Trace the error to Phoenix
                if RAG_EVALUATION_AVAILABLE:
                    try:
                        rag_integration.trace_error(
                            trace_id=trace_id,
                            error=str(e),
                            error_type=type(e).__name__,
                            endpoint="/chat/completions"
                        )
                    except Exception as trace_error:
                        print(f"⚠️ Phoenix error tracing failed: {trace_error}")
                
                # Return error in OpenAI format
                raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
        else:
            # Trace the 503 error to Phoenix
            if RAG_EVALUATION_AVAILABLE:
                try:
                    rag_integration.trace_error(
                        trace_id=trace_id,
                        error="RAG system not available",
                        error_type="ServiceUnavailable",
                        endpoint="/chat/completions"
                    )
                except Exception as trace_error:
                    print(f"⚠️ Phoenix error tracing failed: {trace_error}")
            
            raise HTTPException(status_code=503, detail="RAG system not available")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in OpenAI endpoint: {e}")
        
        # Trace the error to Phoenix
        if RAG_EVALUATION_AVAILABLE:
            try:
                rag_integration.trace_error(
                    trace_id=trace_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    endpoint="/chat/completions"
                )
            except Exception as trace_error:
                print(f"⚠️ Phoenix error tracing failed: {trace_error}")
        
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Alternative OpenAI endpoint path
@app.post("/v1/chat/completions", response_model=OpenAIResponse)
async def openai_v1_chat_completions(request: OpenAIRequest):
    """Alternative OpenAI-compatible endpoint path."""
    return await openai_chat_completions(request)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Enhanced chat endpoint for RAG queries."""
    global llama_model
    
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    
    try:
        print(f"🔍 Processing question: {request.question}")
        print(f"📊 Trace ID: {trace_id}")
        
        # Trace the request to Phoenix regardless of outcome
        if RAG_EVALUATION_AVAILABLE:
            try:
                rag_integration.trace_request(
                    trace_id=trace_id,
                    question=request.question,
                    endpoint="/chat",
                    user_id=request.user_id,
                    session_id=request.session_id
                )
            except Exception as trace_error:
                print(f"⚠️ Phoenix tracing failed: {trace_error}")
        
        if llama_model is not None or _ensure_llama_model():
            # Use real LlamaModel for document queries
            try:
                print("📚 Querying real documents from PostgreSQL...")
                
                answer = llama_model.query(request.question)
                
                # Calculate performance metrics
                end_time = time.time()
                response_time = end_time - start_time
                
                # Evaluate RAG response if evaluation is available
                evaluation_result = None
                if RAG_EVALUATION_AVAILABLE:
                    try:
                        # Extract context from the answer (simplified - in real implementation, get actual context)
                        context = str(answer)[:500]  # First 500 chars as context
                        evaluation_result = rag_integration.process_rag_query(
                            question=request.question,
                            context=context,
                            answer=str(answer),
                            metadata={
                                "endpoint": "/chat",
                                "trace_id": trace_id,
                                "response_time": response_time
                            }
                        )
                        print("✅ RAG response evaluated successfully")
                    except Exception as eval_error:
                        print(f"⚠️ RAG evaluation failed: {eval_error}")
                
                performance_metrics = {
                    "response_time_ms": round(response_time * 1000, 2),
                    "query_length": len(request.question),
                    "response_length": len(str(answer)),
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_available": RAG_EVALUATION_AVAILABLE
                }
                
                # Add evaluation scores to performance metrics if available
                if evaluation_result and evaluation_result.get("success"):
                    evaluation_scores = evaluation_result.get("evaluation", {}).get("scores", {})
                    if evaluation_scores:
                        performance_metrics["evaluation_scores"] = evaluation_scores
                        performance_metrics["evaluation_method"] = evaluation_result.get("evaluation", {}).get("evaluation_method", "unknown")
                
                response = ChatResponse(
                    answer=str(answer),
                    context="Context extracted from your ingested documents",
                    sources=["Your PostgreSQL document database"],
                    confidence=0.90,
                    model_used="llama3.2 + PostgreSQL",
                    trace_id=trace_id,
                    performance_metrics=performance_metrics
                )
                
                print(f"✅ Real RAG response generated from documents")
                print(f"📊 Performance: {response_time:.2f}s response time")
                
                return response
                
            except Exception as e:
                print(f"⚠️ LlamaModel query failed: {e}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error details: {str(e)}")
                
                # Trace the error to Phoenix
                if RAG_EVALUATION_AVAILABLE:
                    try:
                        rag_integration.trace_error(
                            trace_id=trace_id,
                            error=str(e),
                            error_type=type(e).__name__,
                            endpoint="/chat"
                        )
                    except Exception as trace_error:
                        print(f"⚠️ Phoenix error tracing failed: {trace_error}")
                
                # Return error response instead of fallback
                raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
        else:
            # Trace the 503 error to Phoenix
            if RAG_EVALUATION_AVAILABLE:
                try:
                    rag_integration.trace_error(
                        trace_id=trace_id,
                        error="RAG system not available",
                        error_type="ServiceUnavailable",
                        endpoint="/chat"
                    )
                except Exception as trace_error:
                    print(f"⚠️ Phoenix error tracing failed: {trace_error}")
            
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        print(f"✅ Response generated successfully")
        return response
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        
        # Trace the error to Phoenix
        if RAG_EVALUATION_AVAILABLE:
            try:
                rag_integration.trace_error(
                    trace_id=trace_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    endpoint="/chat"
                )
            except Exception as trace_error:
                print(f"⚠️ Phoenix error tracing failed: {trace_error}")
        
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_response(request: EvaluationRequest):
    """Evaluate RAG response quality using heuristic metrics."""
    try:
        print(f"🔍 Evaluating response for: {request.question}")
        
        # Simple heuristic evaluation (in production, use more sophisticated methods)
        relevance_score = min(0.9, len(request.question) / 100)  # Simple relevance based on question length
        accuracy_score = 0.85  # Placeholder for actual accuracy measurement
        completeness_score = 0.88  # Placeholder for completeness measurement
        
        overall_score = (relevance_score + accuracy_score + completeness_score) / 3
        
        feedback = f"Response evaluation completed. Overall quality: {overall_score:.2f}/1.0"
        
        return EvaluationResponse(
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            overall_score=overall_score,
            feedback=feedback
        )
        
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.get("/models")
async def get_models():
    """Get available models information."""
    return {
        "llm_model": "llama3.2" if llama_model else "fallback",
        "embedding_model": "qllama/bge-large-en-v1.5" if llama_model else "none",
        "vector_store": "PostgreSQL with pgvector" if llama_model else "none",
        "status": "fully_loaded" if llama_model else "fallback_mode",
        "rag_evaluation": "enabled" if RAG_EVALUATION_AVAILABLE else "disabled",
        "phoenix_tracing": "enabled" if RAG_EVALUATION_AVAILABLE and rag_integration.evaluator.is_phoenix_available() else "disabled"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed system status."""
    monitoring_status = {}
    if RAG_EVALUATION_AVAILABLE:
        monitoring_status = rag_integration.get_system_status()
    
    return {
        "api_status": "healthy",
        "llama_model_loaded": llama_model is not None,
        "database_connected": llama_model is not None,
        "models_available": {
            "llm": "llama3.2" if llama_model else "none",
            "embeddings": "qllama/bge-large-en-v1.5" if llama_model else "none"
        },
        "rag_evaluation_available": RAG_EVALUATION_AVAILABLE,
        "phoenix_tracing": "enabled" if RAG_EVALUATION_AVAILABLE and rag_integration.evaluator.is_phoenix_available() else "disabled",
        "openai_compatibility": "enabled",
        "lazy_loading": "enabled" if llama_model is None else "disabled",
        "monitoring_status": monitoring_status
    }

# Traces endpoint removed

if __name__ == "__main__":
    print("🚀 Starting Enhanced RAG API Bridge...")
    print("📡 API will be available at: http://localhost:8000")
    print("🔗 Open WebUI can connect to: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)
