#!/usr/bin/env python3
"""
Test script to test LlamaModel directly
"""

import sys
import os

# Add the llama_index directory to the path
sys.path.append('llama_index')

try:
    from llama_store.llama_model import LlamaModel
    
    print("🚀 Testing LlamaModel directly...")
    
    # Initialize the model
    llama_model = LlamaModel()
    print("✅ LlamaModel initialized")
    
    # Check if index exists
    if llama_model.index:
        print("✅ Index exists")
    else:
        print("❌ Index not created")
    
    # Check if query engine exists
    if llama_model.query_engine:
        print("✅ Query engine exists")
    else:
        print("❌ Query engine not created")
    
    # Try to create index if it doesn't exist
    if not llama_model.index:
        print("🔧 Creating index...")
        llama_model.create_index()
        print("✅ Index created")
    
    # Try to query
    if llama_model.query_engine:
        print("🔍 Testing query...")
        response = llama_model.query("What documents do you have access to?")
        print(f"✅ Query successful: {response}")
    else:
        print("❌ Query engine not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"   Type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
