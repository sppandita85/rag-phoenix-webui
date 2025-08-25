#!/usr/bin/env python3
"""
Test script to debug index creation
"""

import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    print("📚 Importing LlamaModel...")
    from llama_index.llama_store.llama_model import LlamaModel
    
    print("🚀 Creating LlamaModel instance...")
    model = LlamaModel()
    
    print("🔧 Creating index...")
    model.create_index()
    
    print("✅ Index created successfully!")
    print(f"Query engine available: {model.query_engine is not None}")
    
    if model.query_engine:
        print("🔍 Testing query...")
        response = model.query("What documents do you have?")
        print(f"Response: {str(response)[:200]}...")
    else:
        print("❌ Query engine not available")
        
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
