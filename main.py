"""
FRESH ML API Main Entry Point
=============================

Main entry point for the FRESH ML API server.
Run this file to start the API server.
"""

import uvicorn
import argparse
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to start the API server"""
    
    parser = argparse.ArgumentParser(
        description="FRESH ML API Server - Fruit Detection and Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Start with default settings
  python main.py --host 0.0.0.0 --port 8080  # Custom host and port
  python main.py --reload                  # Enable auto-reload for development
        """
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        default="info", 
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Print startup information
    print("🚀 FRESH ML API Server")
    print("=" * 50)
    print(f"📍 Server URL: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Interactive API: http://{args.host}:{args.port}/redoc")
    print(f"🖥️  Log Level: {args.log_level}")
    print(f"👥 Workers: {args.workers}")
    print(f"🔄 Reload: {'Enabled' if args.reload else 'Disabled'}")
    print("=" * 50)
    print()
    
    # Key endpoints information
    print("🛠️  Available Endpoints:")
    print(f"   GET  /api/health                     - Health check")
    print(f"   GET  /api/models/info                - Model information")
    print(f"   POST /api/detection/fruits           - Single image (file upload)")
    print(f"   POST /api/detection/fruits/base64    - Single image (base64)")
    print(f"   POST /api/detection/fruits/batch     - Batch processing")
    print(f"   GET  /api/detection/batch/status/{{id}} - Batch status")
    print()
    
    try:
        # Start the server
        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1  # Workers don't work with reload
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()