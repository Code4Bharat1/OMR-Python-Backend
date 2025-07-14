import os
import sys
import platform
from app import app  # Make sure your Flask app is defined as `app` in app.py or __init__.py

def main():
    """Main function to run the Flask application"""
    
    print("=" * 60)
    print("Flask OMR Application - Windows Compatible")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    
    # Set environment variables for development mode (optional)
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'

    # Start the app
    app.run(
        host='127.0.0.1',  # localhost
        port=5000,         # default port
        debug=True         # auto-restart on code changes
    )

if __name__ == "__main__":
    main()
