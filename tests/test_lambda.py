import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to sys.path so we can import LAMBDA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LAMBDA import LAMBDA

class FakeFile:
    def __init__(self, name):
        self.name = name

def test_lambda_initialization():
    """Test that LAMBDA agent can be initialized without errors."""
    agent = LAMBDA(config_path='config.yaml')
    
    assert agent is not None
    assert agent.conv is not None
    assert agent.session_cache_path is not None
    assert agent.conv.programmer is not None
    assert agent.conv.inspector is not None

def test_lambda_add_invalid_file():
    """Test that adding a file is handled gracefully without crashing."""
    agent = LAMBDA(config_path='config.yaml')
    
    # Create a real physical temp file to simulate a real Gradio temp upload
    temp_file_path = "tests/temp_upload_test.csv"
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write("customer_id,age\nC001,30\nC002,40\n")
    
    try:
        fake_file = FakeFile(temp_file_path)
        status_msg = agent.add_file(fake_file)
        
        # The agent should return an assistant message list, not crash
        assert isinstance(status_msg, list)
        assert len(status_msg) == 1
        assert "role" in status_msg[0]
        assert "content" in status_msg[0]
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
