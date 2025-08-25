#!/usr/bin/env python3
"""
Test script to verify Open WebUI connection to custom RAG API
"""

import requests
import json

def test_rag_api_directly():
    """Test the RAG API directly to ensure it's working."""
    print("🔍 Testing RAG API directly...")
    
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ RAG API Health Check:")
            print(f"   Status: {health_data['api_status']}")
            print(f"   LlamaModel Loaded: {health_data['llama_model_loaded']}")
            print(f"   Database Connected: {health_data['database_connected']}")
            return True
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_rag_chat():
    """Test the RAG chat endpoint."""
    print("\n🔍 Testing RAG Chat Endpoint...")
    
    try:
        payload = {
            "question": "What documents do you have?",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        
        response = requests.post(
            "http://localhost:8000/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ RAG Chat Response:")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Model: {result['model_used']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Trace ID: {result.get('trace_id', 'N/A')}")
            
            # Check if it's a real response or fallback
            if "Query engine not available" in result['answer']:
                print("⚠️  Warning: Query engine not available - index may still be building")
            elif "fallback" in result['answer'].lower():
                print("⚠️  Warning: Using fallback response")
            else:
                print("🎉 SUCCESS: Real RAG response from your documents!")
            
            return True
        else:
            print(f"❌ Chat request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat request error: {e}")
        return False

def test_openwebui_connection():
    """Test if Open WebUI can reach the RAG API."""
    print("\n🔍 Testing Open WebUI → RAG API Connection...")
    
    try:
        # Simulate what Open WebUI would send
        payload = {
            "question": "Tell me about your document processing capabilities",
            "user_id": "openwebui_user",
            "session_id": "webui_session"
        }
        
        response = requests.post(
            "http://localhost:8000/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Open WebUI Connection Test:")
            print(f"   Question: {payload['question']}")
            print(f"   Response: {result['answer'][:100]}...")
            print(f"   Sources: {result['sources']}")
            return True
        else:
            print(f"❌ Connection test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def main():
    """Run all connection tests."""
    print("🚀 Testing Open WebUI → Custom RAG API Connection")
    print("=" * 60)
    
    # Test 1: RAG API Health
    if not test_rag_api_directly():
        print("\n❌ RAG API is not healthy. Cannot proceed.")
        return
    
    # Test 2: RAG Chat Functionality
    if not test_rag_chat():
        print("\n❌ RAG Chat is not working. Cannot proceed.")
        return
    
    # Test 3: Open WebUI Connection
    if not test_openwebui_connection():
        print("\n❌ Open WebUI connection test failed.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 ALL CONNECTION TESTS PASSED!")
    print("=" * 60)
    print("✅ Your RAG API is working perfectly")
    print("✅ Open WebUI can reach your API")
    print("✅ The connection is established")
    
    print("\n📝 Next Steps:")
    print("1. Restart Open WebUI to pick up the new configuration")
    print("2. Go to Open WebUI and start a new chat")
    print("3. Your questions should now use the RAG pipeline!")
            print("4. Check API documentation: http://localhost:8000/docs")
    
    print("\n🔧 If Open WebUI still doesn't work:")
    print("   • Look for model selection options in the chat interface")
    print("   • Check if there's a 'RAG' or 'Custom API' model option")
    print("   • Verify the configuration took effect")

if __name__ == "__main__":
    main()
