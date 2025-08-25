#!/usr/bin/env python3
"""
Debug script to test LlamaModel import in API context
"""

import os
import sys

# Add the project root to the path (same as API)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

print(f"🔍 Project root: {project_root}")
print(f"🔍 Python path: {sys.path[:3]}...")

try:
    print("📚 Attempting to import LlamaModel...")
    from llama_index.llama_store.llama_model import LlamaModel
    print("✅ LlamaModel imported successfully")
    
    print("🚀 Attempting to instantiate LlamaModel...")
    model = LlamaModel()
    print("✅ LlamaModel instantiated successfully")
    
    print("🔍 Testing a simple query...")
    response = model.query("What documents do you have?")
    print(f"✅ Query successful: {str(response)[:100]}...")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
