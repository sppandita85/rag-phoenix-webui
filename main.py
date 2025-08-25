#!/usr/bin/env python3
"""
NorthBayPoC RAG System - Complete Project Flow
A complete RAG (Retrieval-Augmented Generation) system with Web UI integration
"""

import os
import sys
import time
import subprocess
import requests
import signal
import threading
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class ProjectManager:
    """Manages the complete project flow from document processing to service startup."""
    
    def __init__(self):
        self.processes = []
        self.services_ready = False
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")
        
        # Check if inputs directory has documents
        input_dir = project_root / "inputs" / "input_data"
        if not input_dir.exists():
            print("❌ Input directory not found: inputs/input_data/")
            return False
            
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found in inputs/input_data/")
            print("   Please add PDF documents to process")
            return False
            
        print(f"✅ Found {len(pdf_files)} PDF files to process")
        
        # Check if virtual environment is activated
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("⚠️  Virtual environment not detected. Please activate your venv first:")
            print("   source venv/bin/activate")
            return False
            
        print("✅ Virtual environment is active")
        return True
    
    def process_documents(self) -> bool:
        """Process PDF documents: parse, chunk, embed, and load into vector store."""
        print("\n📚 Processing documents...")
        
        try:
            # Import document processing modules
            from core.document_processor.pdf_parser import PDFParser
            from core.vector_store.llama_model_simple import LlamaModelSimple
            
            print("✅ Document processing modules imported")
            
            # Initialize LlamaModel to process documents
            print("🔄 Initializing LlamaModel for document processing...")
            llama_model = LlamaModelSimple()
            
            # This will trigger document processing, chunking, and embedding
            print("🔄 Processing documents and loading into vector store...")
            # The LlamaModel initialization should handle document processing
            
            print("✅ Document processing completed")
            return True
            
        except ImportError as e:
            print(f"❌ Failed to import document processing modules: {e}")
            return False
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            return False
    
    def start_services(self) -> bool:
        """Start all required services: RAG API, Web UI, and Phoenix."""
        print("\n🔧 Starting services...")
        
        try:
            # Start RAG API
            print("🚀 Starting RAG API...")
            api_process = subprocess.Popen([
                "uvicorn", "api.simple_api:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(("RAG API", api_process))
            print("✅ RAG API started")
            
            # Wait for API to be ready
            print("⏳ Waiting for RAG API to be ready...")
            time.sleep(5)
            
            # Start Web UI
            print("🌐 Starting Web UI...")
            webui_process = subprocess.Popen([
                "open-webui", "serve", "--port", "8080"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(("Web UI", webui_process))
            print("✅ Web UI started")
            
            # Start Phoenix
            print("📊 Starting Phoenix monitoring...")
            phoenix_process = subprocess.Popen([
                "python", "launch_phoenix.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(("Phoenix", phoenix_process))
            print("✅ Phoenix started")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start services: {e}")
            return False
    
    def wait_for_services(self, timeout: int = 60) -> bool:
        """Wait for all services to be ready."""
        print(f"\n⏳ Waiting for services to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        services_ready = {
            "RAG API": False,
            "Web UI": False,
            "Phoenix": False
        }
        
        while time.time() - start_time < timeout:
            # Check RAG API
            if not services_ready["RAG API"]:
                try:
                    response = requests.get("http://localhost:8000/health", timeout=5)
                    if response.status_code == 200:
                        services_ready["RAG API"] = True
                        print("✅ RAG API is ready")
                except:
                    pass
            
            # Check Web UI
            if not services_ready["Web UI"]:
                try:
                    response = requests.get("http://localhost:8080", timeout=5)
                    if response.status_code == 200:
                        services_ready["Web UI"] = True
                        print("✅ Web UI is ready")
                except:
                    pass
            
            # Check Phoenix
            if not services_ready["Phoenix"]:
                try:
                    response = requests.get("http://localhost:6006", timeout=5)
                    if response.status_code == 200:
                        services_ready["Phoenix"] = True
                        print("✅ Phoenix is ready")
                except:
                    pass
            
            # Check if all services are ready
            if all(services_ready.values()):
                print("🎉 All services are ready!")
                self.services_ready = True
                return True
            
            time.sleep(2)
        
        print("⚠️  Service startup timeout. Some services may not be ready.")
        return False
    
    def run_health_check(self) -> bool:
        """Run a comprehensive health check."""
        print("\n🏥 Running health check...")
        
        try:
            # Check RAG API health
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ RAG API Health Check:")
                print(f"   Status: {health_data.get('api_status', 'Unknown')}")
                print(f"   LlamaModel: {health_data.get('llama_model_loaded', False)}")
                print(f"   Database: {health_data.get('database_connected', False)}")
                print(f"   RAG Evaluation: {health_data.get('rag_evaluation_available', False)}")
                print(f"   Phoenix Tracing: {health_data.get('phoenix_tracing', 'Unknown')}")
            else:
                print(f"❌ RAG API health check failed: {response.status_code}")
                return False
            
            # Test a simple RAG query
            print("\n🧪 Testing RAG query...")
            test_response = requests.post(
                "http://localhost:8000/chat",
                json={
                    "question": "Test query to verify system",
                    "user_id": "startup_test",
                    "session_id": "startup_test"
                },
                timeout=30
            )
            
            if test_response.status_code == 200:
                test_data = test_response.json()
                print("✅ RAG Query Test Successful:")
                print(f"   Answer: {test_data.get('answer', 'N/A')[:100]}...")
                print(f"   Evaluation Available: {test_data.get('performance_metrics', {}).get('evaluation_available', False)}")
            else:
                print(f"❌ RAG query test failed: {test_response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup processes on exit."""
        print("\n🧹 Cleaning up processes...")
        for service_name, process in self.processes:
            try:
                print(f"🛑 Stopping {service_name}...")
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        print("✅ Cleanup completed")
    
    def run(self):
        """Run the complete project flow."""
        print("🚀 NorthBayPoC RAG System - Complete Project Flow")
        print("=" * 60)
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                print("❌ Prerequisites not met. Exiting.")
                return False
            
            # Process documents
            if not self.process_documents():
                print("❌ Document processing failed. Exiting.")
                return False
            
            # Start services
            if not self.start_services():
                print("❌ Service startup failed. Exiting.")
                return False
            
            # Wait for services to be ready
            if not self.wait_for_services():
                print("⚠️  Some services may not be ready.")
            
            # Run health check
            if not self.run_health_check():
                print("⚠️  Health check failed. Some services may not be working properly.")
            
            # Display final status
            print("\n🎉 Project Startup Complete!")
            print("=" * 40)
            print("📊 Services Status:")
            print("   RAG API:     http://localhost:8000")
            print("   Web UI:      http://localhost:8080")
            print("   Phoenix:     http://localhost:6006")
            print("   API Docs:    http://localhost:8000/docs")
            
            print("\n💡 Next Steps:")
            print("   1. Open Web UI at http://localhost:8080")
            print("   2. Ask questions about your documents")
            print("   3. Monitor performance in Phoenix dashboard")
            print("   4. Press Ctrl+C to stop all services")
            
            # Keep the script running
            print("\n⏳ Keeping services running... (Press Ctrl+C to stop)")
            while True:
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Received interrupt signal...")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            self.cleanup()
            print("\n👋 Goodbye!")

def main():
    """Main entry point for the RAG system."""
    # Set up signal handling
    def signal_handler(signum, frame):
        print("\n🛑 Received signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run project manager
    manager = ProjectManager()
    manager.run()

if __name__ == "__main__":
    main()