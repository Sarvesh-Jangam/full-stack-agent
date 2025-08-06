"""
Tests for Database Tools
"""

import pytest
import asyncio
import tempfile
import os

from backend_agent.tools.database_tools import DatabaseTool, DatabaseQuery, create_database_tools
from backend_agent.config.database import DatabaseConfig

class TestDatabaseTools:
    """Test suite for Database Tools"""
    
    def setup_method(self):
        """Setup test database"""
        # Use SQLite for testing
        self.test_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.test_db_file.close()
        
        config = DatabaseConfig()
        config.database_url = f"sqlite:///{self.test_db_file.name}"
        self.db_tool = DatabaseTool(config)
    
    def teardown_method(self):
        """Cleanup test database"""
        if hasattr(self, 'db_tool'):
            asyncio.run(self.db_tool.close())
        if os.path.exists(self.test_db_file.name):
            os.unlink(self.test_db_file.name)
    
    def test_health_check(self):
        """Test database health check"""
        health = self.db_tool.health_check()
        assert "status" in health
        assert "engine" in health
    
    @pytest.mark.asyncio
    async def test_create_table(self):
        """Test table creation"""
        query = """
        CREATE TABLE test_users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
        """
        
        result = await self.db_tool.execute_query(query, {}, "none")
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_insert_record(self):
        """Test record insertion"""
        # Create table first
        await self.db_tool.execute_query("""
            CREATE TABLE test_users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
        """, {}, "none")
        
        # Insert record
        result = await self.db_tool.create_record(
            "test_users",
            {"name": "John Doe", "email": "john@example.com"}
        )
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_read_records(self):
        """Test record reading"""
        # Setup table and data
        await self.db_tool.execute_query("""
            CREATE TABLE test_users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
        """, {}, "none")
        
        await self.db_tool.create_record(
            "test_users",
            {"name": "John Doe", "email": "john@example.com"}
        )
        
        # Read records
        result = await self.db_tool.read_records("test_users")
        assert result.success is True
        assert len(result.data) > 0
    
    @pytest.mark.asyncio
    async def test_transaction(self):
        """Test transaction handling"""
        # Setup table
        await self.db_tool.execute_query("""
            CREATE TABLE test_users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
        """, {}, "none")
        
        # Test transaction
        operations = [
            {
                "query": "INSERT INTO test_users (name, email) VALUES (:name, :email)",
                "params": {"name": "User 1", "email": "user1@example.com"}
            },
            {
                "query": "INSERT INTO test_users (name, email) VALUES (:name, :email)",
                "params": {"name": "User 2", "email": "user2@example.com"}
            }
        ]
        
        result = await self.db_tool.execute_transaction(operations)
        assert result.success is True
    
    def test_tool_factory(self):
        """Test database tools factory"""
        config = DatabaseConfig()
        tools = create_database_tools(config)
        assert len(tools) > 0
        assert callable(tools[0])
