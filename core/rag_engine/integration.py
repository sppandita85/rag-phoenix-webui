#!/usr/bin/env python3
"""
RAG System Integration Module
Combines RAGAs evaluation and Phoenix monitoring for comprehensive RAG tracking
"""

import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import uuid

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from core.rag_engine.evaluation import rag_evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGIntegration:
    """Main integration class for RAG system monitoring and evaluation."""
    
    def __init__(self, project_name: str = "NorthBayPoC"):
        """Initialize the RAG integration."""
        self.project_name = project_name
        self.evaluator = rag_evaluator
        
        logger.info(f"🚀 RAG Integration initialized for project: {project_name}")
        logger.info(f"✅ RAGAs available: {self.evaluator.is_ragas_available()}")
        logger.info(f"✅ Phoenix available: {self.evaluator.is_phoenix_available()}")
    
    def process_rag_query(
        self, 
        question: str, 
        context: str, 
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a RAG query with evaluation and monitoring."""
        try:
            # Prepare metadata
            query_metadata = {
                "project": self.project_name,
                "query_type": "rag",
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Evaluate the RAG response
            evaluation_result = self.evaluator.evaluate_rag_response(
                question=question,
                context=context,
                answer=answer,
                metadata=query_metadata
            )
            
            # Create comprehensive result
            result = {
                "question": question,
                "answer": answer,
                "context": context,
                "evaluation": evaluation_result,
                "metadata": query_metadata,
                "success": True
            }
            
            logger.info(f"✅ RAG query processed and evaluated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing RAG query: {e}")
            return {
                "question": question,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_evaluate_rag_system(
        self, 
        queries: List[Dict[str, str]],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Batch evaluate multiple RAG queries."""
        try:
            # Extract data for evaluation
            questions = [q["question"] for q in queries]
            contexts = [q["context"] for q in queries]
            answers = [q["answer"] for q in queries]
            
            # Run system evaluation
            system_evaluation = self.evaluator.evaluate_rag_system(
                questions=questions,
                contexts=contexts,
                answers=answers,
                metadata_list=metadata_list
            )
            
            # Create batch result
            batch_result = {
                "total_queries": len(queries),
                "system_evaluation": system_evaluation,
                "individual_results": [],
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # Process individual results
            for i, query in enumerate(queries):
                individual_result = self.process_rag_query(
                    question=query["question"],
                    context=query["context"],
                    answer=query["answer"],
                    metadata=metadata_list[i] if metadata_list else None
                )
                batch_result["individual_results"].append(individual_result)
            
            logger.info(f"✅ Batch evaluation completed for {len(queries)} queries")
            return batch_result
            
        except Exception as e:
            logger.error(f"❌ Error in batch evaluation: {e}")
            return {
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG system."""
        return {
            "project_name": self.project_name,
            "ragas_available": self.evaluator.is_ragas_available(),
            "phoenix_available": self.evaluator.is_phoenix_available(),
            "phoenix_url": self.evaluator.get_phoenix_url(),
            "timestamp": datetime.now().isoformat(),
            "status": "healthy"
        }
    
    def get_phoenix_handler(self):
        """Get the Phoenix callback handler for LlamaIndex integration."""
        return self.evaluator.get_phoenix_handler()
    
    def start_monitoring(self):
        """Start the monitoring systems."""
        logger.info("🚀 Starting RAG system monitoring...")
        
        if self.evaluator.is_phoenix_available():
            logger.info(f"📊 Phoenix monitoring active at: {self.evaluator.get_phoenix_url()}")
        
        if self.evaluator.is_ragas_available():
            logger.info("📈 RAGAs evaluation system active")
        
        logger.info("✅ RAG system monitoring started successfully")
    
    def trace_request(self, trace_id: str, question: str, endpoint: str, user_id: str = None, session_id: str = None):
        """Trace a request to Phoenix for monitoring."""
        try:
            if not self.evaluator.is_phoenix_available():
                return
            
            # Import Phoenix components
            from phoenix.trace.schemas import Span, SpanContext, SpanKind, SpanStatusCode, SpanEvent
            from phoenix.trace import TraceDataset
            from datetime import datetime
            
            # Create span context with proper span_id
            span_id = str(uuid.uuid4())
            context = SpanContext(
                trace_id=trace_id,
                span_id=span_id
            )
            
            # Create span events
            events = [
                SpanEvent(
                    name="request_received",
                    timestamp=datetime.now(),
                    attributes={
                        "trace_id": trace_id,
                        "endpoint": endpoint,
                        "project": self.project_name,
                        "question": question,
                        "event_type": "rag_request_start"
                    }
                )
            ]
            
            # Create span attributes
            attributes = {
                "trace_id": trace_id,
                "question": question,
                "endpoint": endpoint,
                "user_id": user_id,
                "session_id": session_id,
                "project": self.project_name,
                "timestamp": datetime.now().isoformat(),
                "rag_status": "request_received",
                "question_length": len(question),
                "event_type": "rag_request_start"
            }
            
            # Note: Conversation object not available in this Phoenix version
            conversation = None
            
            # Create span
            span = Span(
                name=f"RAG Request - {endpoint}",
                context=context,
                span_kind=SpanKind.UNKNOWN,
                parent_id=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status_code=SpanStatusCode.OK,
                status_message="Request received successfully",
                attributes=attributes,
                events=events,
                conversation=conversation
            )
            
            # Create trace dataset and log to Phoenix
            trace_dataset = TraceDataset.from_spans([span])
            self.evaluator.phoenix_client.log_traces(trace_dataset, project_name=self.project_name)
            
            logger.debug(f"✅ Request traced to Phoenix: {trace_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to trace request to Phoenix: {e}")
    
    def trace_response(self, trace_id: str, question: str, answer: str, context: str, sources: list, 
                      confidence: float, performance_metrics: dict, endpoint: str, user_id: str = None, session_id: str = None):
        """Trace a RAG response to Phoenix for monitoring."""
        try:
            print(f"🔍 trace_response called with trace_id: {trace_id}")
            if not self.evaluator.is_phoenix_available():
                print(f"⚠️ Phoenix not available in evaluator")
                return
            print(f"✅ Phoenix is available, proceeding with response tracing")
            
            # Import Phoenix components
            from phoenix.trace.schemas import Span, SpanContext, SpanKind, SpanStatusCode, SpanEvent
            from phoenix.trace import TraceDataset
            from datetime import datetime
            
            # Create span context with proper span_id
            span_id = str(uuid.uuid4())
            context_obj = SpanContext(
                trace_id=trace_id,
                span_id=span_id
            )
            
            # Create span events
            events = [
                SpanEvent(
                    name="rag_response_generated",
                    timestamp=datetime.now(),
                    attributes={
                        "trace_id": trace_id,
                        "endpoint": endpoint,
                        "project": self.project_name,
                        "question": question,
                        "answer_length": len(answer),
                        "context_length": len(context),
                        "sources_count": len(sources),
                        "confidence": confidence,
                        "response_time_ms": performance_metrics.get("response_time_ms", 0),
                        "event_type": "rag_response_complete"
                    }
                )
            ]
            
            # Create span attributes
            attributes = {
                "trace_id": trace_id,
                "question": question,
                "answer": answer,
                "context": context,
                "sources": str(sources),
                "confidence": confidence,
                "endpoint": endpoint,
                "user_id": user_id,
                "session_id": session_id,
                "project": self.project_name,
                "timestamp": datetime.now().isoformat(),
                "rag_status": "response_generated",
                "question_length": len(question),
                "answer_length": len(answer),
                "context_length": len(context),
                "sources_count": len(sources),
                "response_time_ms": performance_metrics.get("response_time_ms", 0),
                "query_length": performance_metrics.get("query_length", 0),
                "response_length": performance_metrics.get("response_length", 0),
                "total_tokens": performance_metrics.get("total_tokens", 0),
                "event_type": "rag_response_complete"
            }
            
            # Create span
            span = Span(
                name=f"RAG Response - {endpoint}",
                context=context_obj,
                span_kind=SpanKind.UNKNOWN,
                parent_id=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status_code=SpanStatusCode.OK,
                status_message="RAG response generated successfully",
                attributes=attributes,
                events=events,
                conversation=None
            )
            
            # Create trace dataset and log to Phoenix
            trace_dataset = TraceDataset.from_spans([span])
            print(f"🔍 Sending response trace to Phoenix with span: {span.name}")
            self.evaluator.phoenix_client.log_traces(trace_dataset, project_name=self.project_name)
            print(f"✅ Response trace sent to Phoenix successfully")
            
            logger.debug(f"✅ Response traced to Phoenix: {trace_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to trace response to Phoenix: {e}")
            print(f"❌ Error in trace_response: {e}")
            import traceback
            traceback.print_exc()
    
    def trace_error(self, trace_id: str, error: str, error_type: str, endpoint: str):
        """Trace an error to Phoenix for monitoring."""
        try:
            if not self.evaluator.is_phoenix_available():
                return
            
            # Import Phoenix components
            from phoenix.trace.schemas import Span, SpanContext, SpanKind, SpanStatusCode, SpanEvent
            from phoenix.trace import TraceDataset
            from datetime import datetime
            
            # Create span context with proper span_id
            span_id = str(uuid.uuid4())
            context = SpanContext(
                trace_id=trace_id,
                span_id=span_id
            )
            
            # Create span events
            events = [
                SpanEvent(
                    name="error_occurred",
                    timestamp=datetime.now(),
                    attributes={
                        "trace_id": trace_id,
                        "error": error,
                        "error_type": error_type,
                        "endpoint": endpoint,
                        "project": self.project_name
                    }
                )
            ]
            
            # Create span attributes
            attributes = {
                "trace_id": trace_id,
                "endpoint": endpoint,
                "error": error,
                "error_type": error_type,
                "project": self.project_name,
                "timestamp": datetime.now().isoformat()
            }
            
            # Create span
            span = Span(
                name=f"RAG Error - {endpoint}",
                context=context,
                span_kind=SpanKind.UNKNOWN,
                parent_id=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status_code=SpanStatusCode.ERROR,
                status_message=f"Error: {error}",
                attributes={
                    "trace_id": trace_id,
                    "error": error,
                    "error_type": error_type,
                    "endpoint": endpoint,
                    "project": self.project_name,
                    "event": "error_occurred"
                },
                events=events,
                conversation=None
            )
            
            # Create trace dataset and log to Phoenix
            trace_dataset = TraceDataset.from_spans([span])
            self.evaluator.phoenix_client.log_traces(trace_dataset, project_name=self.project_name)
            
            logger.debug(f"✅ Error traced to Phoenix: {trace_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to trace error to Phoenix: {e}")
    
    def stop_monitoring(self):
        """Stop the monitoring systems."""
        logger.info("🛑 Stopping RAG system monitoring...")
        # Cleanup operations if needed
        logger.info("✅ RAG system monitoring stopped")

# Global integration instance
rag_integration = RAGIntegration()
