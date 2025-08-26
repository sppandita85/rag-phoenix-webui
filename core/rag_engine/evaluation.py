#!/usr/bin/env python3
"""
RAG Evaluation and Monitoring Module
Integrates RAGAs and Phoenix for comprehensive RAG system evaluation
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize flags
RAGAS_AVAILABLE = False
PHOENIX_AVAILABLE = False

# Import the separate RAGAs evaluator to avoid uvloop conflicts
try:
    from core.rag_engine.ragas_evaluator import ragas_evaluator
    RAGAS_AVAILABLE = ragas_evaluator.is_available()
    if RAGAS_AVAILABLE:
        print("✅ RAGAs evaluator imported successfully")
        logger.info("✅ RAGAs evaluator imported successfully")
    else:
        print("⚠️ RAGAs evaluator not available")
        logger.warning("⚠️ RAGAs evaluator not available")
except ImportError as e:
    print(f"❌ RAGAs evaluator import failed: {e}")
    logger.warning(f"❌ RAGAs evaluator import failed: {e}")
    RAGAS_AVAILABLE = False

# Try to import Phoenix
try:
    import phoenix as px

    # Try different import paths for Phoenix callback handler
    try:
        from llama_index.callbacks.arize_phoenix import ArizePhoenixCallbackHandler
        PHOENIX_CALLBACK_AVAILABLE = True
    except ImportError:
        try:
            from llama_index.callbacks import ArizePhoenixCallbackHandler
            PHOENIX_CALLBACK_AVAILABLE = True
        except ImportError:
            PHOENIX_CALLBACK_AVAILABLE = False
            logger.warning("⚠️ Phoenix callback handler not available")

    PHOENIX_AVAILABLE = True
    logger.info("✅ Phoenix imported successfully")

except ImportError as e:
    logger.warning(f"⚠️ Phoenix not available: {e}")
except Exception as e:
    logger.warning(f"⚠️ Phoenix import failed: {e}")

class RAGEvaluator:
    """Comprehensive RAG evaluation using RAGAs and Phoenix."""

    def __init__(self, project_name: str = "NorthBayPoC"):
        """Initialize the RAG evaluator."""
        self.project_name = project_name
        self.phoenix_client = None
        self.phoenix_handler = None

        # Initialize Phoenix if available
        if PHOENIX_AVAILABLE:
            self._init_phoenix()

    def _init_phoenix(self):
        """Initialize Phoenix client and callback handler."""
        try:
            # Initialize Phoenix client
            self.phoenix_client = px.Client()
            logger.info("✅ Phoenix client initialized successfully")

            # Initialize callback handler for LlamaIndex if available
            if PHOENIX_CALLBACK_AVAILABLE:
                try:
                    self.phoenix_handler = ArizePhoenixCallbackHandler(
                        project_name=self.project_name
                    )
                    logger.info("✅ Phoenix callback handler initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Phoenix callback handler: {e}")
                    self.phoenix_handler = None
            else:
                logger.warning("⚠️ Phoenix callback handler not available")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Phoenix: {e}")
            self.phoenix_client = None
            self.phoenix_handler = None

    def evaluate_rag_response(
        self,
        question: str,
        context: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single RAG response using RAGAs."""
        if not RAGAS_AVAILABLE or not RAGAS_METRICS:
            return {
                "error": "RAGAs not available",
                "scores": {"placeholder": 0.0},
                "question": question,
                "context_length": len(context),
                "answer_length": len(answer),
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }

        try:
            # Try to use custom Ollama-based evaluation if available
            try:
                from core.rag_engine.ollama_llm_wrapper import OllamaLLMWrapper
                ollama_llm = OllamaLLMWrapper("llama3.2")
                
                # Perform basic evaluation using Ollama
                scores = self._evaluate_with_ollama(ollama_llm, question, answer, context)
                
                evaluation_result = {
                    "scores": scores,
                    "question": question,
                    "context_length": len(context),
                    "answer_length": len(answer),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {},
                    "evaluation_method": "ollama_custom"
                }
                
                logger.info(f"✅ RAG response evaluated using Ollama LLM")
                return evaluation_result
                
            except Exception as e:
                logger.warning(f"⚠️ Custom Ollama evaluation failed: {e}, falling back to basic evaluation")
                
                # Fallback to basic evaluation without LLM
                scores = self._basic_evaluation(question, answer, context)
                
                evaluation_result = {
                    "scores": scores,
                    "question": question,
                    "context_length": len(context),
                    "answer_length": len(answer),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {},
                    "evaluation_method": "basic_heuristic"
                }
                
                logger.info(f"✅ RAG response evaluated using basic heuristics")
                return evaluation_result

        except Exception as e:
            logger.error(f"❌ Error evaluating RAG response: {e}")
            return {
                "error": str(e),
                "scores": {"error": 0.0},
                "question": question,
                "context_length": len(context),
                "answer_length": len(answer),
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
    
    def _evaluate_with_ollama(self, ollama_llm, question: str, answer: str, context: str) -> Dict[str, float]:
        """Evaluate response quality using Ollama LLM"""
        try:
            # Faithfulness evaluation: Check if answer is faithful to context
            faithfulness_prompt = f"""Given the context: "{context}"

And the answer: "{answer}"

Rate how faithful the answer is to the context on a scale of 0.0 to 1.0, where:
- 1.0 = Completely faithful, all information can be directly inferred from context
- 0.5 = Partially faithful, some information matches context
- 0.0 = Not faithful, information contradicts or is not in context

Return only the number (e.g., 0.8):"""

            faithfulness_response = ollama_llm.complete(faithfulness_prompt)
            faithfulness_score = self._extract_score(faithfulness_response)
            
            # Answer relevancy evaluation: Check if answer is relevant to question
            relevancy_prompt = f"""Given the question: "{question}"

And the answer: "{answer}"

Rate how relevant the answer is to the question on a scale of 0.0 to 1.0, where:
- 1.0 = Highly relevant, directly answers the question
- 0.5 = Somewhat relevant, partially answers the question
- 0.0 = Not relevant, doesn't answer the question

Return only the number (e.g., 0.9):"""

            relevancy_response = ollama_llm.complete(relevancy_prompt)
            relevancy_score = self._extract_score(relevancy_response)
            
            return {
                "faithfulness": faithfulness_score,
                "answer_relevancy": relevancy_score,
                "overall_quality": (faithfulness_score + relevancy_score) / 2
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Ollama evaluation failed: {e}")
            return self._basic_evaluation(question, answer, context)
    
    def _basic_evaluation(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """Basic evaluation using heuristics when LLM is not available"""
        scores = {}
        
        # Basic faithfulness: Check if answer words appear in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        common_words = answer_words.intersection(context_words)
        
        if len(answer_words) > 0:
            faithfulness = min(1.0, len(common_words) / len(answer_words))
        else:
            faithfulness = 0.0
        
        # Basic relevancy: Check if question words appear in answer
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        relevant_words = question_words.intersection(answer_words)
        
        if len(question_words) > 0:
            relevancy = min(1.0, len(relevant_words) / len(question_words))
        else:
            relevancy = 0.0
        
        scores["faithfulness"] = round(faithfulness, 2)
        scores["answer_relevancy"] = round(relevancy, 2)
        scores["overall_quality"] = round((faithfulness + relevancy) / 2, 2)
        
        return scores
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|\d+\.\d+', response)
            if numbers:
                score = float(numbers[0])
                return min(1.0, max(0.0, score))  # Clamp between 0.0 and 1.0
            else:
                # If no number found, try to extract from text
                if "high" in response.lower() or "good" in response.lower():
                    return 0.8
                elif "medium" in response.lower() or "okay" in response.lower():
                    return 0.5
                elif "low" in response.lower() or "poor" in response.lower():
                    return 0.2
                else:
                    return 0.5  # Default score
        except:
            return 0.5  # Default score on error

    def evaluate_rag_system(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate multiple RAG responses for system-level metrics."""
        if not RAGAS_AVAILABLE or not RAGAS_METRICS:
            return {
                "error": "RAGAs not available",
                "system_metrics": {"placeholder": 0.0},
                "summary": {
                    "total_samples": len(questions),
                    "average_scores": {"placeholder": 0.0},
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_complete": True
                }
            }

        try:
            # Create dataset
            if RAGAS_DATASET_AVAILABLE:
                dataset = Dataset.from_dict({
                    "question": questions,
                    "contexts": contexts,
                    "answer": answers
                })
            else:
                # Fallback if Dataset is not available
                return {
                    "error": "RAGAs Dataset not available",
                    "system_metrics": {"placeholder": 0.0},
                    "summary": {
                        "total_samples": len(questions),
                        "average_scores": {"placeholder": 0.0},
                        "timestamp": datetime.now().isoformat(),
                        "evaluation_complete": True
                    }
                }

            # Run evaluation with available metrics
            results = evaluate(dataset, RAGAS_METRICS)

            # Extract and format results
            system_results = {}
            for metric_name, metric_result in results.items():
                if hasattr(metric_result, 'score'):
                    system_results[metric_name] = float(metric_result.score)
                else:
                    system_results[metric_result] = metric_result

            # Calculate summary statistics
            summary = {
                "total_samples": len(questions),
                "average_scores": system_results,
                "timestamp": datetime.now().isoformat(),
                "evaluation_complete": True
            }

            # Trace system evaluation to Phoenix
            if self.phoenix_client and self.phoenix_handler:
                self._trace_system_evaluation(summary)

            logger.info(f"✅ RAG system evaluation completed for {len(questions)} samples")
            return {
                "system_metrics": system_results,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"❌ Error evaluating RAG system: {e}")
            return {
                "error": str(e),
                "system_metrics": {"error": 0.0},
                "summary": {
                    "total_samples": len(questions),
                    "average_scores": {"error": 0.0},
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_complete": False
                }
            }

    def _trace_evaluation(self, evaluation_result: Dict[str, Any]):
        """Trace evaluation results to Phoenix."""
        try:
            if not self.phoenix_client:
                return

            # Import Phoenix components
            from phoenix.trace.schemas import Span, SpanContext, SpanKind, SpanStatusCode, SpanEvent
            from phoenix.trace import TraceDataset
            from datetime import datetime
            
            # Create span context
            context = SpanContext(
                trace_id=f"eval_{hash(evaluation_result['question']) % 10000}",
                span_id=f"eval_span_{hash(evaluation_result['question']) % 10000}"
            )
            
            # Create span events
            events = [
                SpanEvent(
                    name="evaluation_completed",
                    timestamp=datetime.now(),
                    attributes={
                        "timestamp": evaluation_result["timestamp"],
                        "project": self.project_name
                    }
                )
            ]
            
            # Create span
            span = Span(
                name="RAG Response Evaluation",
                context=context,
                span_kind=SpanKind.UNKNOWN,
                parent_id=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status_code=SpanStatusCode.OK,
                status_message="Evaluation completed successfully",
                attributes={
                    "timestamp": evaluation_result["timestamp"],
                    "project": self.project_name,
                    "event": "evaluation_completed"
                },
                events=events,
                conversation=None
            )
            
            # Create trace dataset and log to Phoenix
            trace_dataset = TraceDataset.from_spans([span])
            self.phoenix_client.log_traces(trace_dataset, project_name=self.project_name)

            logger.debug("✅ Evaluation traced to Phoenix")

        except Exception as e:
            logger.warning(f"⚠️ Failed to trace evaluation to Phoenix: {e}")

    def _trace_system_evaluation(self, summary: Dict[str, Any]):
        """Trace system evaluation results to Phoenix."""
        try:
            if not self.phoenix_client:
                return

            # Import Phoenix components
            from phoenix.trace.schemas import Span, SpanContext, SpanKind, SpanStatusCode, SpanEvent
            from phoenix.trace import TraceDataset
            from datetime import datetime
            
            # Create span context
            context = SpanContext(
                trace_id=f"sys_eval_{hash(summary['timestamp']) % 10000}",
                span_id=f"sys_eval_span_{hash(summary['timestamp']) % 10000}"
            )
            
            # Create span events
            events = [
                SpanEvent(
                    name="system_evaluation_completed",
                    timestamp=datetime.now(),
                    attributes={
                        "timestamp": summary["timestamp"],
                        "project": self.project_name
                    }
                )
            ]
            
            # Create span attributes
            attributes = {
                "total_samples": summary["total_samples"],
                "average_scores": summary["average_scores"],
                "evaluation_complete": summary["evaluation_complete"],
                "timestamp": summary["timestamp"],
                "project": self.project_name
            }
            
            # Create span
            span = Span(
                name="RAG System Evaluation",
                context=context,
                span_kind=SpanKind.INTERNAL,
                parent_id=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status_code=SpanStatusCode.OK,
                status_message="System evaluation completed successfully",
                attributes=attributes,
                events=events,
                conversation=None
            )
            
            # Create trace dataset and log to Phoenix
            trace_dataset = TraceDataset.from_spans([span])
            self.phoenix_client.log_traces(trace_dataset, project_name=self.project_name)

            logger.debug("✅ System evaluation traced to Phoenix")

        except Exception as e:
            logger.warning(f"⚠️ Failed to trace system evaluation to Phoenix: {e}")

    def get_phoenix_handler(self):
        """Get the Phoenix callback handler for LlamaIndex integration."""
        return self.phoenix_handler

    def get_phoenix_url(self) -> str:
        """Get the Phoenix dashboard URL."""
        if self.phoenix_client:
            return "http://localhost:6006"
        return "Phoenix not available"

    def is_phoenix_available(self) -> bool:
        """Check if Phoenix is available."""
        return PHOENIX_AVAILABLE and self.phoenix_client is not None

    def is_ragas_available(self) -> bool:
        """Check if RAGAs is available."""
        try:
            from core.rag_engine.ragas_evaluator import ragas_evaluator
            return ragas_evaluator.is_available()
        except ImportError:
            return False

# Global evaluator instance
rag_evaluator = RAGEvaluator()
