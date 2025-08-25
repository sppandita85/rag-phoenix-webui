#!/usr/bin/env python3
"""
Launch Phoenix for RAG System Monitoring
"""

import phoenix as px
import time
import signal
import sys

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Phoenix stopped by user")
    sys.exit(0)

def main():
    """Launch Phoenix monitoring dashboard."""
    print("🚀 Launching Phoenix for RAG System Monitoring...")
    print("📊 Dashboard will be available at: http://localhost:6006")
    print("🔄 Phoenix will run in background mode")
    print("📝 Check phoenix.log for detailed logs")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Launch Phoenix app
        print("🌍 Starting Phoenix server...")
        px.launch_app(port=6006)
        
        # Keep the process running
        print("✅ Phoenix is now running in background mode")
        print("📊 Access dashboard at: http://localhost:6006")
        print("🔄 Press Ctrl+C to stop Phoenix")
        
        # Keep the process alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Phoenix stopped by user")
    except Exception as e:
        print(f"❌ Error launching Phoenix: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
