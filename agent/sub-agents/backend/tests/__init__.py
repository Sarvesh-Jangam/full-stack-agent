"""
Test package for Backend Agent
Contains all unit and integration tests.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_DATABASE_URL = "sqlite:///test_backend_agent.db"
TEST_UPLOAD_DIR = "./test_uploads"

# Create test upload directory
os.makedirs(TEST_UPLOAD_DIR, exist_ok=True)

# Set test environment variables
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["UPLOAD_DIRECTORY"] = TEST_UPLOAD_DIR
os.environ["TESTING"] = "True"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"

def run_async_test(coro):
    """Helper function to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

