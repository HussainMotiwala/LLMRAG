from typing import Dict, Any, List, Optional
import os
import json
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel

from backend.config import Config, logger
from backend.database import DatabaseManager
from backend.rag_manager import RAGManager
from backend.llm_chain import SQLLLMChain

# Initialize the main components
db_manager = DatabaseManager()
connection_success = db_manager.connect()
if not connection_success:
    logger.error("Failed to connect to database. LangServe API may not function correctly.")

# Initialize RAG Manager
try:
    # Try to initialize with GPU first
    rag_manager = RAGManager()
except Exception as e:
    logger.error(f"Error initializing RAG Manager with GPU: {str(e)}")
    logger.warning("Falling back to CPU for embeddings")
    try:
        # Fall back to CPU implementation
        from backend.rag_manager_cpu import RAGManager
        rag_manager = RAGManager(use_gpu=False)
    except Exception as e:
        logger.error(f"Error initializing RAG Manager with CPU: {str(e)}")
        rag_manager = None

# Initialize SQL LLM Chain - try real implementation first, fall back to mock if needed
try:
    # Try to initialize the real LLM chain
    sql_chain = SQLLLMChain(db_manager=db_manager, rag_manager=rag_manager)
    
    # Test if LLM was properly initialized
    if not hasattr(sql_chain, 'sql_llm') or sql_chain.sql_llm is None:
        logger.warning("LLM not initialized properly. Will use mock implementation.")
        raise ValueError("LLM not initialized")
        
except Exception as e:
    logger.warning(f"Falling back to mock LLM chain: {str(e)}")
    try:
        # Import and use the mock implementation
        from backend.mock_llm_chain import MockSQLLLMChain
        sql_chain = MockSQLLLMChain(db_manager=db_manager, rag_manager=rag_manager)
        logger.info("Using Mock LLM Chain for testing")
    except Exception as mock_error:
        logger.error(f"Error initializing Mock LLM Chain: {str(mock_error)}")
        sql_chain = None

# Create FastAPI app
app = FastAPI(
    title="LLM RAG SQL API",
    version="1.0",
    description="API for generating and executing SQL queries with RAG augmentation"
)

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input and output models
class SQLQueryInput(BaseModel):
    query: str
    model: Optional[str] = None
    max_attempts: Optional[int] = 3

class SQLQueryOutput(BaseModel):
    status: str
    sql_query: str
    error: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    attempts: int
    processing_time: float
    model: str
    row_count: Optional[int] = None
    debug_info: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint to check if service is running."""
    return {
        "status": "online",
        "service": "LLM RAG SQL API", 
        "version": "1.0",
        "database_connected": db_manager.connected,
        "models_available": list(Config.AVAILABLE_MODELS.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database_connected": db_manager.connected
    }

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": Config.AVAILABLE_MODELS,
        "default_model": Config.DEFAULT_LLM_MODEL
    }

@app.post("/execute_query", response_model=SQLQueryOutput)
async def execute_query(input_data: SQLQueryInput):
    """Generate and execute an SQL query from natural language."""
    try:
        # Log all component states
        logger.info(f"DEBUG: sql_chain type: {type(sql_chain)}")
        logger.info(f"DEBUG: db_manager type: {type(db_manager)}")
        logger.info(f"DEBUG: rag_manager type: {type(rag_manager)}")
        
        # Check if sql_chain is properly initialized
        if sql_chain is None:
            logger.error("SQL Chain is not initialized")
            return {
                "status": "error",
                "sql_query": "No query generated",
                "error": "SQL Chain not properly initialized - API configuration error",
                "results": None,
                "attempts": 0,
                "processing_time": 0,
                "model": "Unknown",
                "row_count": 0
            }
            
        # Check if sql_chain.execute_chain is callable
        if not hasattr(sql_chain, 'execute_chain') or not callable(getattr(sql_chain, 'execute_chain', None)):
            logger.error("SQL Chain execute_chain method is not callable")
            return {
                "status": "error",
                "sql_query": "No query generated",
                "error": "SQL Chain execute_chain method is not callable - API configuration error",
                "results": None,
                "attempts": 0,
                "processing_time": 0,
                "model": "Unknown",
                "row_count": 0
            }
            
        # Validate model selection
        model_name = None
        if input_data.model:
            if input_data.model in Config.AVAILABLE_MODELS:
                model_name = Config.AVAILABLE_MODELS[input_data.model]
            else:
                model_name = input_data.model  # Use directly if full model name provided
        
        logger.info(f"Executing query: {input_data.query}")
        
        # Execute the chain
        result = sql_chain.execute_chain(
            user_query=input_data.query,
            model_name=model_name,
            max_attempts=input_data.max_attempts
        )
        
        # Format result for API response
        response = {
            "status": result.get("status", "error"),
            "sql_query": result.get("sql_query", ""),
            "error": result.get("error"),
            "results": result.get("raw_data"),
            "attempts": result.get("attempts", 0),
            "processing_time": result.get("processing_time", 0),
            "model": result.get("model", Config.DEFAULT_LLM_MODEL),
            "row_count": len(result.get("raw_data", [])) if result.get("raw_data") else 0,
            "debug_info": result.get("debug_info", {})
        }
        
        return response
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "sql_query": "Error during execution",
            "error": f"Server error: {str(e)}",
            "results": None,
            "attempts": 0,
            "processing_time": 0,
            "model": "Unknown",
            "row_count": 0
        }

def start():
    """Start the LangServe API server."""
    import uvicorn
    uvicorn.run(
        "backend.langserve_app:app",
        host=Config.LANGSERVE_HOST,
        port=Config.LANGSERVE_PORT,
        reload=True
    )

if __name__ == "__main__":
    start()
