#!/usr/bin/env python3
"""
Test script to check API connection and LlamaModel status
"""

import requests
import json

def test_api_endpoints():
    """Test all API endpoints to identify the issue."""
    
    base_url = "http://localhost:8000"
    
    print("🔍 Testing API endpoints...")
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 2: Health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test 3: Models endpoint
    try:
        response = requests.get(f"{base_url}/models")
        print(f"✅ Models endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
    
    # Test 4: Chat endpoint (simple)
    try:
        payload = {
            "question": "What documents do you have access to?",
            "user_id": "test"
        }
        response = requests.post(f"{base_url}/chat", json=payload)
        print(f"✅ Chat endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
    
    # Test 5: OpenAI-compatible endpoint
    try:
        payload = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "What documents do you have access to?"}
            ],
            "temperature": 0.7
        }
        response = requests.post(f"{base_url}/chat/completions", json=payload)
        print(f"✅ OpenAI endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ OpenAI endpoint error: {e}")

if __name__ == "__main__":
    test_api_endpoints()
