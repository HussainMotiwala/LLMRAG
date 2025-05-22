import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the application."""
    # LLM Models - Using Together API
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    
    # Available models
    AVAILABLE_MODELS = {
        "llama3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "qwen-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "deepseek-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    }
    
    # LLM Settings
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
    
    # API Keys
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "LLM_RAG")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    
    # Enable LangSmith tracing if API key is available
    if LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Database settings
    DB_TYPE = os.getenv("DB_TYPE", "mssql")
    DB_HOST = os.getenv("DB_HOST", "localhost\\SQLEXPRESS")
    DB_PORT = os.getenv("DB_PORT", "1433")
    DB_NAME = os.getenv("DB_NAME", "TestLLM")
    DB_USER = os.getenv("DB_USER", "sa")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "TestLLM@123")
    DB_TRUSTED_CONNECTION = os.getenv("DB_TRUSTED_CONNECTION", "0")
    
    # Vector DB settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_db")
    
    # RAG document paths
    SCHEMA_PATH = os.getenv("SCHEMA_PATH", "data/Schema.json")
    METADATA_PATH = os.getenv("METADATA_PATH", "data/Business_Metadata.pdf")
    QUERY_PATH = os.getenv("QUERY_PATH", "data/Prompt_Query_Examples.json")
    COMPLEX_QUERY_PATH = os.getenv("COMPLEX_QUERY_PATH", "data/Complex_Query_Examples.json")
    
    # Processing settings
    MAX_SQL_RETRIES = int(os.getenv("MAX_SQL_RETRIES", "3"))
    
    # LangServe settings
    LANGSERVE_HOST = os.getenv("LANGSERVE_HOST", "0.0.0.0")
    LANGSERVE_PORT = int(os.getenv("LANGSERVE_PORT", "8010"))
    
    @classmethod
    def get_db_connection_string(cls) -> str:
        """Generate database connection string."""
        if cls.DB_TYPE == "mssql":
            # Check if we're using Windows Authentication (trusted connection)            
            if cls.DB_TRUSTED_CONNECTION == "1":
                # Windows Authentication (trusted connection)
                return f'DRIVER={{SQL Server}};SERVER={cls.DB_HOST};DATABASE={cls.DB_NAME};Trusted_Connection=yes;TrustServerCertificate=yes;'
            else:
                # SQL Server Authentication with username and password
                return f'DRIVER={{SQL Server}};SERVER={cls.DB_HOST};DATABASE={cls.DB_NAME};UID={cls.DB_USER};PWD={cls.DB_PASSWORD};TrustServerCertificate=yes;'
        else:
            raise ValueError(f"Unsupported database type: {cls.DB_TYPE}")

# Utility functions
def get_timestamp() -> str:
    """Get current timestamp in readable format."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def sanitize_sql_input(input_str: str) -> str:
    """Sanitize input to prevent SQL injection."""
    # Basic sanitization - remove semicolons, comments, etc.
    # Note: This is basic protection, the main protection is using parameterized queries
    dangerous_patterns = [
        "--", ";", "/*", "*/", "@@", "@", "EXECUTE", "EXEC", "xp_", "sp_"
    ]
    result = input_str
    for pattern in dangerous_patterns:
        result = result.replace(pattern, "")
    return result

def timing_decorator(func):
    """Decorator to track execution time of functions."""
    import time
    import functools
    
    @functools.wraps(func)  # This is important!
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_timestamp = get_timestamp()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        end_timestamp = get_timestamp()
        
        # Add timing info to result if it's a dict
        if isinstance(result, dict):
            result["processing_time"] = processing_time
            result["start_timestamp"] = start_timestamp
            result["end_timestamp"] = end_timestamp
        
        return result
    
    return wrapper  # Return the wrapper function
