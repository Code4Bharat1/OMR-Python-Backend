#!/usr/bin/env python3
"""
Production WSGI server using Waitress
This file serves as the production entry point for the OMR Flask application
"""

import os
import logging
from waitress import serve
from app import app  # ✅ Make sure this points to your Flask instance

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

logger = logging.getLogger(__name__)

def create_production_server():
    """
    Create and configure the production server
    """
    # Get configuration from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8000))
    threads = int(os.environ.get('WAITRESS_THREADS', 4))
    
    # Additional Waitress configuration
    connection_limit = int(os.environ.get('WAITRESS_CONNECTION_LIMIT', 1000))
    cleanup_interval = int(os.environ.get('WAITRESS_CLEANUP_INTERVAL', 30))
    channel_timeout = int(os.environ.get('WAITRESS_CHANNEL_TIMEOUT', 120))
    
    logger.info(f"Starting production server on {host}:{port}")
    logger.info(f"Configuration: threads={threads}, connection_limit={connection_limit}")
    
    # ✅ Correctly indented serve() call
    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        connection_limit=connection_limit,
        cleanup_interval=cleanup_interval,
        channel_timeout=channel_timeout,
        max_request_header_size=8192,
        max_request_body_size=20 * 1024 * 1024  # 20MB
    )

if __name__ == '__main__':
    create_production_server()
 