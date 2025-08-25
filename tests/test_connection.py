#!/usr/bin/env python3
"""
Test script to validate Open WebUI → RAG API connection
"""

import requests
import json
import time

def test_rag_api():
    """Test if RAG API is accessible"""
    print("🔍 Testing RAG API accessibility...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ RAG API is running and accessible")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ RAG API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to RAG API: {e}")
        return False
    
    return True

def test_openwebui():
    """Test if Open WebUI is accessible"""
    print("\n🔍 Testing Open WebUI accessibility...")
    
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print("✅ Open WebUI is running and accessible")
        else:
            print(f"❌ Open WebUI returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Open WebUI: {e}")
        return False
    
    return True

def test_rag_chat():
    """Test RAG API chat endpoint"""
    print("\n🔍 Testing RAG API chat functionality...")
    
    try:
        payload = {
            "question": "What documents do you have?",
            "user_id": "test_user"
        }
        
        response = requests.post(
            "http://localhost:8000/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ RAG API chat endpoint working")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Model: {result['model_used']}")
            print(f"   Confidence: {result['confidence']}")
        else:
            print(f"❌ Chat endpoint returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return False
    
    return True

def simulate_openwebui_request():
    """Simulate what Open WebUI would send to your API"""
    print("\n🔍 Simulating Open WebUI request to RAG API...")
    
    try:
        # Simulate Open WebUI chat request
        payload = {
            "question": "Tell me about your document processing capabilities",
            "user_id": "openwebui_user",
            "session_id": "test_session_123"
        }
        
        response = requests.post(
            "http://localhost:8000/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Open WebUI → RAG API connection successful!")
            print(f"   Question: {payload['question']}")
            print(f"   Response: {result['answer'][:100]}...")
            print(f"   Sources: {result['sources']}")
            return True
        else:
            print(f"❌ Connection failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    """Run all connection tests"""
    print("🚀 Testing Open WebUI → RAG API Connection")
    print("=" * 50)
    
    # Test 1: RAG API accessibility
    if not test_rag_api():
        print("\n❌ RAG API test failed. Cannot proceed.")
        return
    
    # Test 2: Open WebUI accessibility  
    if not test_openwebui():
        print("\n❌ Open WebUI test failed. Cannot proceed.")
        return
    
    # Test 3: RAG API chat functionality
    if not test_rag_chat():
        print("\n❌ RAG API chat test failed. Cannot proceed.")
        return
    
    # Test 4: Simulate Open WebUI connection
    if simulate_openwebui_request():
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Open WebUI can successfully connect to your RAG API")
        print("✅ Your custom API bridge is working correctly")
        print("\n📝 Next steps:")
        print("   1. Configure Open WebUI to use http://localhost:8000")
        print("   2. Test with real questions in Open WebUI")
        print("   3. Verify responses come from your RAG pipeline")
    else:
        print("\n❌ Open WebUI connection test failed")
        print("   Check your API configuration and try again")

if __name__ == "__main__":
    main()
