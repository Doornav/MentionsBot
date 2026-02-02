#!/usr/bin/env python3
"""
Startup script for the ML service
"""

import os
import sys
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description='Run the Market Insights Engine ML service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with auto-reload')
    
    args = parser.parse_args()
    
    # Add the project root to Python path to ensure imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    print(f"Starting ML service on {args.host}:{args.port}")
    uvicorn.run(
        "ml.api.app:app", 
        host=args.host, 
        port=args.port, 
        reload=args.debug,
        log_level="debug" if args.debug else "info"
    )

if __name__ == "__main__":
    main()