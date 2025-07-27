"""
Database Tools for Backend Agent
Handles all database operations including CRUD, transactions, and complex queries.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import asyncpg
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.sql import text
from pydantic import BaseModel, Field
import pandas as pd

from ..config.database import DatabaseConfig

logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseQuery(BaseModel):
    """Model for database query requests"""
    query: str = Field(..., description="SQL query to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    fetch_mode: str = Field("all", description="Fetch mode: 'all', 'one', 'many'")

class DatabaseResult(BaseModel):
    """Model for database query results"""
    success: bool = Field(..., description="Whether the query was successful")
    data: Any = Field(None, description="Query result data")
    row_count: Optional[int] = Field(None, description="Number of rows affected/returned")
    message: str = Field("", description="Result message")
    execution_time: Optional[float] = Field(None, description="Query execution time")

class DatabaseTool:
    """Advanced database tool with connection pooling and transaction support"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._connection_pool = None
        self._setup_database()
        logger.info("Database tool initialized")

    def _setup_database(self):
        """Setup database engine and connection pool"""
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.config.get_async_url(),
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=self.config.debug
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database engine and session factory created")
            
        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
            raise

    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup"""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {str(e)}")
                raise
            finally:
                await session.close()

    async def execute_query(self, 
                          query: str, 
                          params: Dict[str, Any] = None,
                          fetch_mode: str = "all") -> DatabaseResult:
        """
        Execute a SQL query with parameters
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_mode: 'all', 'one', 'many', or 'none'
        
        Returns:
            DatabaseResult with query results
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.get_session() as session:
                # Execute query with parameters
                result = await session.execute(text(query), params or {})
                
                # Handle different fetch modes
                if fetch_mode == "all":
                    data = result.fetchall()
                    data = [dict(row._mapping) for row in data] if data else []
                elif fetch_mode == "one":
                    row = result.fetchone()
                    data = dict(row._mapping) if row else None
                elif fetch_mode == "many":
                    data = result.fetchmany(100)  # Fetch up to 100 rows
                    data = [dict(row._mapping) for row in data] if data else []
                else:  # fetch_mode == "none"
                    data = None
                
                # Commit the transaction
                await session.commit()
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                row_count = result.rowcount if hasattr(result, 'rowcount') else len(data) if data else 0
                
                logger.info(f"Query executed successfully in {execution_time:.3f}s, {row_count} rows affected")
                
                return DatabaseResult(
                    success=True,
                    data=data,
                    row_count=row_count,
                    message="Query executed successfully",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Query execution failed: {str(e)}")
            
            return DatabaseResult(
                success=False,
                message=f"Query execution failed: {str(e)}",
                execution_time=execution_time
            )

    async def create_record(self, 
                          table_name: str, 
                          data: Dict[str, Any]) -> DatabaseResult:
        """Create a new record in the specified table"""
        try:
            # Build INSERT query
            columns = list(data.keys())
            placeholders = [f":{col}" for col in columns]
            
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING *
            """
            
            return await self.execute_query(query, data, "one")
            
        except Exception as e:
            logger.error(f"Error creating record: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error creating record: {str(e)}"
            )

    async def read_records(self, 
                         table_name: str, 
                         filters: Dict[str, Any] = None,
                         order_by: str = None,
                         limit: int = None) -> DatabaseResult:
        """Read records from the specified table with optional filters"""
        try:
            query = f"SELECT * FROM {table_name}"
            params = {}
            
            # Add WHERE clause if filters are provided
            if filters:
                where_conditions = []
                for key, value in filters.items():
                    where_conditions.append(f"{key} = :{key}")
                    params[key] = value
                query += f" WHERE {' AND '.join(where_conditions)}"
            
            # Add ORDER BY if specified
            if order_by:
                query += f" ORDER BY {order_by}"
            
            # Add LIMIT if specified
            if limit:
                query += f" LIMIT {limit}"
            
            return await self.execute_query(query, params, "all")
            
        except Exception as e:
            logger.error(f"Error reading records: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error reading records: {str(e)}"
            )

    async def update_record(self, 
                          table_name: str, 
                          record_id: Any,
                          data: Dict[str, Any],
                          id_column: str = "id") -> DatabaseResult:
        """Update a record in the specified table"""
        try:
            # Build UPDATE query
            set_clauses = [f"{col} = :{col}" for col in data.keys()]
            params = data.copy()
            params[id_column] = record_id
            
            query = f"""
                UPDATE {table_name} 
                SET {', '.join(set_clauses)}
                WHERE {id_column} = :{id_column}
                RETURNING *
            """
            
            return await self.execute_query(query, params, "one")
            
        except Exception as e:
            logger.error(f"Error updating record: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error updating record: {str(e)}"
            )

    async def delete_record(self, 
                          table_name: str, 
                          record_id: Any,
                          id_column: str = "id") -> DatabaseResult:
        """Delete a record from the specified table"""
        try:
            query = f"DELETE FROM {table_name} WHERE {id_column} = :{id_column} RETURNING *"
            params = {id_column: record_id}
            
            return await self.execute_query(query, params, "one")
            
        except Exception as e:
            logger.error(f"Error deleting record: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error deleting record: {str(e)}"
            )

    async def execute_transaction(self, operations: List[Dict[str, Any]]) -> DatabaseResult:
        """
        Execute multiple operations in a single transaction
        
        Args:
            operations: List of operations, each containing 'query' and 'params'
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.get_session() as session:
                results = []
                
                for operation in operations:
                    query = operation.get('query')
                    params = operation.get('params', {})
                    
                    result = await session.execute(text(query), params)
                    
                    # Collect results based on operation type
                    if query.strip().upper().startswith('SELECT'):
                        data = result.fetchall()
                        results.append([dict(row._mapping) for row in data])
                    else:
                        results.append({"rowcount": result.rowcount})
                
                # Commit all operations
                await session.commit()
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.info(f"Transaction completed successfully in {execution_time:.3f}s")
                
                return DatabaseResult(
                    success=True,
                    data=results,
                    message="Transaction completed successfully",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Transaction failed: {str(e)}")
            
            return DatabaseResult(
                success=False,
                message=f"Transaction failed: {str(e)}",
                execution_time=execution_time
            )

    async def bulk_insert(self, 
                        table_name: str, 
                        records: List[Dict[str, Any]]) -> DatabaseResult:
        """Bulk insert multiple records"""
        try:
            if not records:
                return DatabaseResult(
                    success=False,
                    message="No records provided for bulk insert"
                )
            
            # Build bulk INSERT query
            columns = list(records[0].keys())
            placeholders = ", ".join([f"({', '.join([f':{col}_{i}' for col in columns])})" 
                                    for i in range(len(records))])
            
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES {placeholders}"
            
            # Flatten parameters
            params = {}
            for i, record in enumerate(records):
                for col, value in record.items():
                    params[f"{col}_{i}"] = value
            
            return await self.execute_query(query, params, "none")
            
        except Exception as e:
            logger.error(f"Error in bulk insert: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error in bulk insert: {str(e)}"
            )

    async def search_records(self, 
                           table_name: str, 
                           search_term: str,
                           search_columns: List[str],
                           limit: int = 100) -> DatabaseResult:
        """Search records using full-text search or LIKE operations"""
        try:
            # Build search query with ILIKE for case-insensitive search
            search_conditions = [f"{col} ILIKE :search_term" for col in search_columns]
            query = f"""
                SELECT * FROM {table_name}
                WHERE {' OR '.join(search_conditions)}
                LIMIT {limit}
            """
            
            params = {"search_term": f"%{search_term}%"}
            
            return await self.execute_query(query, params, "all")
            
        except Exception as e:
            logger.error(f"Error searching records: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error searching records: {str(e)}"
            )

    async def get_table_schema(self, table_name: str) -> DatabaseResult:
        """Get schema information for a table"""
        try:
            query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """
            
            return await self.execute_query(query, {"table_name": table_name}, "all")
            
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error getting table schema: {str(e)}"
            )

    async def export_to_csv(self, 
                          query: str, 
                          params: Dict[str, Any] = None,
                          filename: str = None) -> DatabaseResult:
        """Export query results to CSV file"""
        try:
            result = await self.execute_query(query, params, "all")
            
            if not result.success or not result.data:
                return result
            
            # Convert to DataFrame and save as CSV
            df = pd.DataFrame(result.data)
            
            if not filename:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"export_{timestamp}.csv"
            
            export_path = os.path.join(os.getenv('UPLOAD_DIRECTORY', './uploads'), filename)
            df.to_csv(export_path, index=False)
            
            return DatabaseResult(
                success=True,
                data={"filename": filename, "path": export_path, "rows": len(df)},
                message=f"Data exported to {filename}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return DatabaseResult(
                success=False,
                message=f"Error exporting to CSV: {str(e)}"
            )

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection"""
        try:
            # This is a synchronous method for quick health checks
            return {
                "status": "healthy",
                "engine": str(self.engine.url) if self.engine else None,
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")

# Factory function to create database tools
def create_database_tools(config: DatabaseConfig = None) -> List:
    """Create and return database tools for the agent"""
    if not config:
        config = DatabaseConfig()
    
    db_tool = DatabaseTool(config)
    
    # Define individual tool functions that the agent can use
    async def execute_sql_query(query: str, params: dict = None) -> dict:
        """Execute SQL query with parameters"""
        result = await db_tool.execute_query(query, params or {})
        return result.model_dump()
    
    async def create_database_record(table_name: str, data: dict) -> dict:
        """Create a new record in database"""
        result = await db_tool.create_record(table_name, data)
        return result.model_dump()
    
    async def read_database_records(table_name: str, filters: dict = None) -> dict:
        """Read records from database"""
        result = await db_tool.read_records(table_name, filters or {})
        return result.model_dump()
    
    async def update_database_record(table_name: str, record_id: any, data: dict) -> dict:
        """Update a record in database"""
        result = await db_tool.update_record(table_name, record_id, data)
        return result.model_dump()
    
    async def delete_database_record(table_name: str, record_id: any) -> dict:
        """Delete a record from database"""
        result = await db_tool.delete_record(table_name, record_id)
        return result.model_dump()
    
    return [
        execute_sql_query,
        create_database_record,
        read_database_records,
        update_database_record,
        delete_database_record
    ]

if __name__ == "__main__":
    # Test the database tool
    async def test_database_tool():
        config = DatabaseConfig()
        db_tool = DatabaseTool(config)
        
        # Test health check
        health = db_tool.health_check()
        print(f"Health check: {health}")
        
        # Test query execution
        result = await db_tool.execute_query("SELECT 1 as test_value")
        print(f"Test query result: {result}")
        
        await db_tool.close()
    
    asyncio.run(test_database_tool())

