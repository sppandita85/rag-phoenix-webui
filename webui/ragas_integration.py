 #!/usr/bin/env python3
"""
RAGAs Integration for RAG System Evaluation and Tracing
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    AnswerRelevancy,
    ContextRelevance,
    context_recall
)
import os
from typing import List, Dict, Any

class RAGAsIntegration:
    def __init__(self, project_name: str = "NorthBayPoC"):
        """Initialize RAGAs integration for RAG system evaluation."""
        self.project_name = project_name
        
    def evaluate_rag_system(self, questions: List[str], contexts: List[str], 
                           answers: List[str]) -> Dict[str, float]:
        """Evaluate RAG system using RAGAs metrics."""
        try:
            # Create dataset for evaluation
            from datasets import Dataset
            eval_dataset = Dataset.from_dict({
                "question": questions,
                "contexts": contexts,
                "answer": answers
            })
            
            # Run evaluation
            results = evaluate(
                eval_dataset,
                metrics=[
                    faithfulness,
                    AnswerRelevancy(),
                    ContextRelevance(),
                    context_recall
                ]
            )
            
            print(f"✅ RAGAs evaluation completed successfully")
            return results.to_dict()
            
        except Exception as e:
            print(f"❌ Error in RAGAs evaluation: {e}")
            return {}
    
    def trace_rag_query(self, question: str, context: str, answer: str, 
                       metadata: Dict[str, Any] = None):
        """Trace a RAG query for debugging and analysis."""
        try:
            print(f"🔍 Tracing RAG query: {question[:100]}...")
            print(f"   Context length: {len(context)} characters")
            print(f"   Answer length: {len(answer)} characters")
            
            if metadata:
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
                    
            print(f"✅ RAG query traced successfully")
            
        except Exception as e:
            print(f"❌ Error tracing RAG query: {e}")
    
    def get_evaluation_summary(self, results: Dict[str, float]) -> str:
        """Generate a summary of evaluation results."""
        if not results:
            return "No evaluation results available"
        
        summary = f"RAG System Evaluation Summary for {self.project_name}:\n"
        summary += "=" * 50 + "\n"
        
        for metric, score in results.items():
            if isinstance(score, (int, float)):
                summary += f"{metric}: {score:.3f}\n"
            else:
                summary += f"{metric}: {score}\n"
        
        return summary
