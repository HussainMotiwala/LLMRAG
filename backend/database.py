import urllib.parse
import logging
from typing import Dict, List, Optional, Any, Tuple, Union

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from backend.config import Config, logger

class DatabaseManager:
    """Manages database connections and query execution with simplified validation."""
    
    def __init__(self):
        self.connection_string = Config.get_db_connection_string()
        self.conn_url = f'mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(self.connection_string)}'
        self.engine = None
        self.connected = False
        self.max_rows = 100  # Safety limit for row returns
        
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            logger.info(f"Connecting to database: {Config.DB_NAME} on {Config.DB_HOST}")
            # Create engine with fast executemany for performance
            self.engine = create_engine(
                self.conn_url, 
                fast_executemany=True
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT @@VERSION AS version"))
                version = result.scalar()
                logger.info(f"Connected to SQL Server: {version.split('on')[0].strip() if version and 'on' in version else version}")
                
            self.connected = True
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {str(e)}")
            self.connected = False
            return False
    
    def execute_query(self, query: str) -> Tuple[List[Dict], Optional[str]]:
        """
        Execute SQL query with minimal preprocessing.
        
        Args:
            query: SQL query string
            
        Returns:
            Tuple of (results, error_message)
        """
        # Check for connection
        if not self.connected:
            if not self.connect():
                return [], "Database connection error: Could not establish connection"
        
        # Clean up the query (remove any markdown artifacts)
        cleaned_query = query.strip()
        if cleaned_query.startswith("```sql"):
            cleaned_query = cleaned_query.replace("```sql", "").replace("```", "").strip()
        
        # Only allow SELECT queries
        if not (cleaned_query.upper().strip().startswith("SELECT") or 
        cleaned_query.upper().strip().startswith("WITH")):
            return [], "Only SELECT and WITH queries are allowed for security reasons"
        
        try:
            results = []
            with self.engine.connect() as conn:
                # Execute query
                result = conn.execute(text(cleaned_query))
                
                # Process results
                if result.returns_rows:
                    column_names = result.keys()
                    for row in result:
                        results.append(dict(zip(column_names, row)))
                
                logger.info(f"Query executed successfully with {len(results)} rows")
                return results, None
                
        except Exception as e:
            error_message = f"Query execution error: {str(e)}"
            logger.error(error_message)
            return [], error_message
    
    def validate_and_execute_query(self, query: str) -> Dict[str, Any]:
        """
        Simplified validation and execution.
        
        Returns:
            Dict with status, results, and error information
        """
        # Clean up the query
        cleaned_query = query.strip()
        if cleaned_query.startswith("```sql"):
            cleaned_query = cleaned_query.replace("```sql", "").replace("```", "").strip()
        
        # Only allow SELECT queries
        if not (cleaned_query.upper().strip().startswith("SELECT") or 
        cleaned_query.upper().strip().startswith("WITH")):
            return {
                "status": "prohibited_operation",
                "error": "Only SELECT and WITH queries are allowed",
                "results": None
            }
        
        # Check connection
        if not self.connected:
            if not self.connect():
                return {
                    "status": "connection_error",
                    "error": "Could not establish database connection",
                    "results": None
                }
        
        # Try to execute the query
        try:
            with self.engine.connect() as conn:
                # Execute the query
                result = conn.execute(text(cleaned_query))
                
                # Get metadata and sample results
                column_names = result.keys()
                sample_rows = [dict(zip(column_names, row)) for row in result.fetchmany(5)]
                row_count = result.rowcount if result.rowcount >= 0 else "unknown"
                
                return {
                    "status": "success",
                    "results": sample_rows,
                    "columns": list(column_names),
                    "estimated_row_count": row_count,
                    "error": None
                }
        except Exception as e:
            # Execution error
            return {
                "status": "execution_error",
                "error": f"SQL execution error: {str(e)}",
                "results": None
            }
