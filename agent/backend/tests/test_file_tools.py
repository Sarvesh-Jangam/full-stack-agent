"""
Tests for File Processing Tools
"""

import pytest
import asyncio
import os
import tempfile
import json
from pathlib import Path

from backend_agent.tools.file_tools import FileProcessingTool, create_file_tools

class TestFileTools:
    """Test suite for File Processing Tools"""
    
    def setup_method(self):
        """Setup test environment"""
        self.file_tool = FileProcessingTool()
        self.test_files_dir = Path("test_files")
        self.test_files_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup test files"""
        import shutil
        if self.test_files_dir.exists():
            shutil.rmtree(self.test_files_dir)
    
    def test_health_check(self):
        """Test file tool health check"""
        health = self.file_tool.health_check()
        assert "status" in health
        assert "upload_directory" in health
        assert "max_file_size" in health
    
    def test_file_validation(self):
        """Test file validation"""
        # Test valid file
        error = self.file_tool._validate_file("test.txt", 1000)
        assert error is None
        
        # Test invalid extension
        error = self.file_tool._validate_file("test.exe", 1000)
        assert error is not None
        
        # Test file too large
        error = self.file_tool._validate_file("test.txt", self.file_tool.max_file_size + 1)
        assert error is not None
    
    @pytest.mark.asyncio
    async def test_file_upload(self):
        """Test file upload"""
        test_content = b"Hello, World! This is test content."
        filename = "test_upload.txt"
        
        result = await self.file_tool.upload_file(filename, test_content)
        
        assert result.success is True
        assert result.file_info is not None
        assert result.file_info.filename == filename
        assert result.file_info.file_size == len(test_content)
        
        # Verify file exists
        assert os.path.exists(result.file_info.file_path)
    
    @pytest.mark.asyncio
    async def test_csv_processing(self):
        """Test CSV file processing"""
        # Create test CSV
        csv_content = "name,age,city\nJohn,30,New York\nJane,25,San Francisco\n"
        csv_path = self.test_files_dir / "test.csv"
        
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        result = await self.file_tool.process_csv_file(str(csv_path))
        
        assert result.success is True
        assert result.data is not None
        assert "columns" in result.data
        assert "rows" in result.data
        assert len(result.data["rows"]) == 2
    
    @pytest.mark.asyncio
    async def test_json_processing(self):
        """Test JSON file processing"""
        # Create test JSON
        test_data = {
            "users": [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 25}
            ],
            "metadata": {
                "version": "1.0",
                "created_at": "2023-01-01"
            }
        }
        
        json_path = self.test_files_dir / "test.json"
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        result = await self.file_tool.process_json_file(str(json_path))
        
        assert result.success is True
        assert result.data is not None
        assert "content" in result.data
        assert "structure" in result.data
        assert result.data["content"] == test_data
    
    @pytest.mark.asyncio
    async def test_csv_export(self):
        """Test CSV export generation"""
        test_data = [
            {"name": "John", "age": 30, "city": "New York"},
            {"name": "Jane", "age": 25, "city": "San Francisco"},
            {"name": "Bob", "age": 35, "city": "Chicago"}
        ]
        
        result = await self.file_tool.generate_csv_export(test_data, "test_export.csv")
        
        assert result.success is True
        assert result.file_info is not None
        assert result.data["row_count"] == 3
        
        # Verify file exists and has correct content
        assert os.path.exists(result.file_info.file_path)
    
    @pytest.mark.asyncio
    async def test_file_deletion(self):
        """Test file deletion"""
        # Create a test file
        test_file = self.test_files_dir / "delete_me.txt"
        with open(test_file, 'w') as f:
            f.write("Delete this file")
        
        assert test_file.exists()
        
        result = await self.file_tool.delete_file(str(test_file))
        
        assert result.success is True
        assert not test_file.exists()
    
    @pytest.mark.asyncio
    async def test_zip_archive_creation(self):
        """Test ZIP archive creation"""
        # Create test files
        file1 = self.test_files_dir / "file1.txt"
        file2 = self.test_files_dir / "file2.txt"
        
        with open(file1, 'w') as f:
            f.write("Content of file 1")
        with open(file2, 'w') as f:
            f.write("Content of file 2")
        
        file_paths = [str(file1), str(file2)]
        
        result = await self.file_tool.create_zip_archive(file_paths, "test_archive.zip")
        
        assert result.success is True
        assert result.file_info is not None
        assert os.path.exists(result.file_info.file_path)
    
    def test_tool_factory(self):
        """Test file processing tools factory"""
        tools = create_file_tools()
        assert len(tools) > 0
        assert callable(tools[0])

