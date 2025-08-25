#!/usr/bin/env python3
"""
Test script for RAGAs and Phoenix integration
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_rag_evaluation():
    """Test RAG evaluation functionality."""
    print("🧪 Testing RAG Evaluation Integration...")
    
    try:
        from core.rag_engine.evaluation import rag_evaluator
        
        print(f"✅ RAG Evaluator imported successfully")
        print(f"   - RAGAs available: {rag_evaluator.is_ragas_available()}")
        print(f"   - Phoenix available: {rag_evaluator.is_phoenix_available()}")
        
        # Test single response evaluation
        if rag_evaluator.is_ragas_available():
            print("\n📊 Testing single response evaluation...")
            result = rag_evaluator.evaluate_rag_response(
                question="What is the main topic?",
                context="This is a test context about artificial intelligence and machine learning.",
                answer="The main topic is artificial intelligence and machine learning.",
                metadata={"test": True}
            )
            print(f"   ✅ Evaluation result: {result.get('scores', {})}")
        else:
            print("   ⚠️ RAGAs not available, skipping evaluation test")
        
        # Test Phoenix integration
        if rag_evaluator.is_phoenix_available():
            print(f"\n📊 Phoenix integration:")
            print(f"   - URL: {rag_evaluator.get_phoenix_url()}")
            print(f"   - Handler: {rag_evaluator.get_phoenix_handler() is not None}")
        else:
            print("   ⚠️ Phoenix not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG evaluation: {e}")
        return False

def test_rag_integration():
    """Test RAG integration functionality."""
    print("\n🧪 Testing RAG Integration...")
    
    try:
        from core.rag_engine.integration import rag_integration
        
        print(f"✅ RAG Integration imported successfully")
        
        # Test system status
        status = rag_integration.get_system_status()
        print(f"   - Project: {status.get('project_name')}")
        print(f"   - RAGAs: {status.get('ragas_available')}")
        print(f"   - Phoenix: {status.get('phoenix_available')}")
        print(f"   - Status: {status.get('status')}")
        
        # Test query processing
        print("\n📊 Testing query processing...")
        result = rag_integration.process_rag_query(
            question="Test question?",
            context="Test context",
            answer="Test answer",
            metadata={"test": True}
        )
        print(f"   ✅ Query processed: {result.get('success')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG integration: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAG Integration Test Suite")
    print("=" * 40)
    
    # Test evaluation
    eval_success = test_rag_evaluation()
    
    # Test integration
    integration_success = test_rag_integration()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"   - RAG Evaluation: {'✅ PASS' if eval_success else '❌ FAIL'}")
    print(f"   - RAG Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if eval_success and integration_success:
        print("\n🎉 All tests passed! RAG integration is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
