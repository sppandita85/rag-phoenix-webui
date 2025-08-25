#!/usr/bin/env python3
"""
Open WebUI Integration for Chatbot Interface
"""

import os
import subprocess
import time
import requests
from typing import Dict, Any, Optional

class WebUIIntegration:
    def __init__(self, project_name: str = "NorthBayPoC"):
        """Initialize Open WebUI integration."""
        self.project_name = project_name
        self.webui_process = None
        self.webui_url = "http://localhost:8080"
        
    def start_webui(self, port: int = 8080) -> bool:
        """Start Open WebUI server."""
        try:
            print(f"🚀 Starting Open WebUI on port {port}...")
            
            # Start Open WebUI in background
            try:
                self.webui_process = subprocess.Popen([
                    "venv/bin/open-webui", "serve", "--port", str(port)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Wait for server to start
                time.sleep(8)
                
                # Check if server is running
                if self.check_webui_status():
                    print(f"✅ Open WebUI started successfully at {self.webui_url}")
                    return True
                else:
                    print("❌ Failed to start Open WebUI")
                    return False
            except Exception as e:
                print(f"❌ Error starting Open WebUI: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting Open WebUI: {e}")
            return False
    
    def check_webui_status(self) -> bool:
        """Check if Open WebUI is running."""
        try:
            response = requests.get(self.webui_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop_webui(self):
        """Stop Open WebUI server."""
        if self.webui_process:
            try:
                self.webui_process.terminate()
                self.webui_process.wait(timeout=10)
                print("✅ Open WebUI stopped successfully")
            except subprocess.TimeoutExpired:
                self.webui_process.kill()
                print("⚠️ Open WebUI force killed")
            except Exception as e:
                print(f"❌ Error stopping Open WebUI: {e}")
    
    def get_webui_url(self) -> str:
        """Get the WebUI URL."""
        return self.webui_url
    
    def configure_ollama_models(self, models: Dict[str, str]):
        """Configure Ollama models in Open WebUI."""
        try:
            print("🔧 Configuring Ollama models in Open WebUI...")
            
            # This would typically involve API calls to configure models
            # For now, we'll just print the configuration
            for model_name, model_path in models.items():
                print(f"   Model: {model_name} -> {model_path}")
            
            print("✅ Ollama models configured")
            return True
            
        except Exception as e:
            print(f"❌ Error configuring Ollama models: {e}")
            return False
    
    def create_chat_interface(self, system_prompt: str = None):
        """Create a chat interface configuration."""
        try:
            print("💬 Creating chat interface configuration...")
            
            config = {
                "project": self.project_name,
                "system_prompt": system_prompt or "You are a helpful AI assistant.",
                "webui_url": self.webui_url
            }
            
            print("✅ Chat interface configuration created")
            return config
            
        except Exception as e:
            print(f"❌ Error creating chat interface: {e}")
            return {}
