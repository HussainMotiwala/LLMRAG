import time
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union

from langchain_together import Together
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langsmith import Client

from backend.config import Config, logger, timing_decorator
from backend.database import DatabaseManager
from backend.rag_manager import RAGManager

class SQLLLMChain:
    """Manages the end-to-end process of generating and executing SQL queries using LLMs with RAG."""
    
    def __init__(self, db_manager, rag_manager):
        """Initialize the chain with required components."""
        self.db_manager = db_manager
        self.rag_manager = rag_manager
        
        logger.info("Using Together API for LLM inference")
        
        # Initialize the LLM
        self.initialize_llm()
        
        # Initialize LangSmith tracing
        try:
            if Config.LANGSMITH_API_KEY and Config.LANGSMITH_PROJECT:
                self.tracer = LangChainTracer(
                    project_name=Config.LANGSMITH_PROJECT
                )
                self.callback_manager = CallbackManager([self.tracer])
                
                # Let's handle potential API version mismatches
                try:
                    # Try importing with current API
                    self.client = Client()
                    # Test if the client is working properly
                    self.langsmith_enabled = True
                    logger.info(f"LangSmith tracing enabled for project: {Config.LANGSMITH_PROJECT}")
                except Exception as e:
                    logger.warning(f"LangSmith client initialization error: {str(e)}")
                    self.langsmith_enabled = False
                    self.client = None
            else:
                self.callback_manager = None
                self.client = None
                self.langsmith_enabled = False
                logger.warning("LangSmith tracing disabled - missing API key or project name")
        except Exception as e:
            logger.warning(f"Error initializing LangSmith: {str(e)}")
            self.callback_manager = None
            self.client = None
            self.langsmith_enabled = False
    
    def initialize_llm(self, model_name=None):
        """Initialize the LLM with Together API."""
        try:
            # Validate API key
            if not Config.TOGETHER_API_KEY:
                logger.error("Missing Together API key. Please set TOGETHER_API_KEY in .env")
                self.sql_llm = None
                return False
            
            # Use default model if not specified
            if model_name is None:
                model_name = Config.DEFAULT_LLM_MODEL
            
            logger.info(f"Initializing Together API for model: {model_name}")
            
            # Use Together for large models
            self.sql_llm = Together(
                model=model_name,
                together_api_key=Config.TOGETHER_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
                top_p=0.95,
            )
            logger.info(f"Successfully initialized Together API with model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Together API: {str(e)}")
            self.sql_llm = None
            return False
    
    def _create_sql_generation_prompt(self, user_query: str) -> str:
        """Create the prompt for SQL generation with RAG context."""
        
        # Get context from RAG manager
        schema_info = self.rag_manager.get_schema_context(user_query)
        metadata_info = self.rag_manager.get_metadata_context(user_query) 
        query_examples_info = self.rag_manager.get_query_examples_context(user_query, False)
        complex_query_examples_info = self.rag_manager.get_query_examples_context(user_query, True)
        
        # Clean the RAG content by extracting only the essential SQL schema information
        def clean_rag_content(content):
            # Remove all the similarity scores, page numbers, embeddings, etc.
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                # Skip lines with similarity info, query embeddings, section markers
                if any(skip_term in line.lower() for skip_term in 
                       ['similarity:', 'embedding', 'section', 'query:', 'page:']):
                    continue
                if line.strip() and not line.startswith('===') and not line.endswith('==='):
                    cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        
        # Apply cleaning to all RAG content
        schema_context = clean_rag_content(schema_info)
        metadata_context = clean_rag_content(metadata_info)
        query_context = clean_rag_content(query_examples_info)
        complex_query_context = clean_rag_content(complex_query_examples_info)
        
        # Format for LLM with proper instruction format
        prompt = f"""<s>[INST] You are an expert SQL developer with 20+ years of experience. 
Generate a Microsoft SQL Server query to answer this question: "{user_query}"
You will respond ONLY with the exact SQL query that addresses the user's request, with NO explanations, comments, or additional text

Follow these critical strict rules. If you do not follow these rules, you may risk someone's life:
Rules:
- Only use tables and columns from the schema
- Only use SELECT statements or WITH for CTEs
- Use TOP at the start or OFFSET 0 ROWS FETCH NEXT N ROWS ONLY at the end instead of LIMIT as LIMIT does not work in Microsoft SQL Server
- Use INNER JOIN to join tables
- Avoid LAG() with aggregate functions. Try to use CTE instead.
- Return only the SQL query with no explanation
- Do not wrap your response in ```sql``` or any other formatting
- Use a semicolon at the end of your query
- Do not use </s> and repeat the same query with some minor changes. ; means end of the working query. Follow this rule or you may risk someone's life.

Below is the database schema information, metadata, and example queries to help you:

SCHEMA INFORMATION: This has the table and relationship names. 
{schema_context}

METADATA INFORMATION: This has descriptions of the columns and fields.
{metadata_context}

Note: The particular field values have the following synonyms:

Digital in Channel_attribute_4 Column: Synonyms: Online
Term in Category_mis Column: Synonyms: Protection
Traditional in Category_mis Column: Synonyms: Savings

Only use the tables and fields mentioned in this schema information and metadata information in your SQL queries

EXAMPLE QUERIES: Sample queries that are similar to the user's request. Cases: Simple joins, Aggregations, Filtering, Grouping, Sorting.
{query_context}

COMPLEX EXAMPLE QUERIES: Sample queries that are similar to the user's request. Cases: Comparisons, Percentage of total, Running total, Growth, Year over Year, Month over Month, Quarter over Quarter, CTEs and windows functions
{complex_query_context}

Return ONLY the SQL query without any explanations. The query must be syntactically correct for Microsoft SQL Server.
[/INST]</s>
"""
        return prompt
    
    def _create_sql_correction_prompt(self, user_query: str, sql_query: str, error_message: str) -> str:
        """Create a prompt to correct an SQL query based on error feedback."""
        
        # Get context from RAG manager for correction
        schema_context = self.rag_manager.get_schema_context(user_query)
        metadata_context = self.rag_manager.get_metadata_context(user_query)
        
        correction_prompt = f"""<s>[INST] You are an expert SQL developer with 20+ years of experience in Microsoft SQL Server.
Your task is to fix an SQL query that contains errors.
You will respond ONLY with the exact SQL query that addresses the user's request, with NO explanations, comments, or additional text.

ORIGINAL QUERY:
{sql_query}

ERROR MESSAGE:
{error_message}

USER QUERY:
{user_query}

SCHEMA INFORMATION: This has the table and relationship names. 
{schema_context}

METADATA INFORMATION: This has descriptions of the columns and fields.
{metadata_context}

Note: The particular field values have the following synonyms:

Digital in Channel_attribute_4 Column: Synonyms: Online
Term in Category_mis Column: Synonyms: Protection
Traditional in Category_mis Column: Synonyms: Savings

Only use the tables and fields mentioned in this schema information and metadata information in your SQL queries

Return ONLY the corrected SQL query without any explanations.
Ensure your response is a valid Microsoft SQL Server query that addresses the error.
[/INST]</s>
"""
        return correction_prompt
    
    @timing_decorator
    def execute_chain(self, user_query: str, model_name=None, max_attempts: int = 3) -> Dict:
        """
        Execute the full chain of SQL generation, validation, and execution.
        
        Args:
            user_query: Natural language query from the user
            model_name: Optional model to use (defaults to Config.DEFAULT_LLM_MODEL)
            max_attempts: Maximum number of attempts for query correction
            
        Returns:
            Dict containing results, query, and execution information
        """
        # Change model if requested
        
        # Create a dict to store debug info
        debug_info = {
            "system_prompt": "",
            "rag_context": {
                "schema_context": "",
                "metadata_context": "",
                "query_examples_context": "",
                "complex_query_examples_context": ""
            }
        }
        
        if model_name and (not hasattr(self, 'sql_llm') or self.sql_llm is None or model_name != getattr(self.sql_llm, 'model', None)):
            llm_initialized = self.initialize_llm(model_name)
            if not llm_initialized:
                return {
                    "status": "error",
                    "sql_query": "No query generated",
                    "error": "Failed to initialize LLM - check API key and model settings",
                    "data": None,
                    "attempts": 0,
                    "processing_time": 0,
                    "model": model_name or "Unknown"
                }
        
        # Check if LLM is properly initialized
        if not hasattr(self, 'sql_llm') or self.sql_llm is None:
            return {
                "status": "error",
                "sql_query": "No query generated",
                "error": "LLM not initialized - check API key and model settings",
                "data": None,
                "attempts": 0, 
                "processing_time": 0,
                "model": "Unknown"
            }
            
        # Create run in LangSmith if available
        run_id = None
        if hasattr(self, 'langsmith_enabled') and self.langsmith_enabled and self.client:
            try:
                run = self.client.create_run(
                    project_name=Config.LANGSMITH_PROJECT,
                    run_type="chain",  # Required parameter
                    name="SQL_Chain_Execution",
                    inputs={
                        "user_query": user_query,
                        "model": self.sql_llm.model
                    }
                )
                run_id = run.id if hasattr(run, 'id') else None  # Added null check
                if run_id:
                    logger.info(f"Created LangSmith run: {run_id}")
            except Exception as e:
                logger.warning(f"Failed to create LangSmith run: {str(e)}")
                run_id = None
        
        start_time = time.time()
        attempts = 0
        final_result = None
        sql_query = None  # Initialize this outside the loop
        
        # Log the processing workflow
        logger.info(f"Starting SQL chain execution for query: {user_query}")
        
        while attempts < max_attempts:
            try:
                # Track current attempt
                attempts += 1
                logger.info(f"Attempt {attempts}/{max_attempts}")
                
                # Step 1: Generate SQL - first attempt or correction based on previous error
                if attempts == 1:
                    schema_context = self.rag_manager.get_schema_context(user_query)
                    metadata_context = self.rag_manager.get_metadata_context(user_query)
                    query_examples_context = self.rag_manager.get_query_examples_context(user_query, False)
                    complex_query_examples_context = self.rag_manager.get_query_examples_context(user_query, True)
                    
                    # Store RAG contexts in debug info
                    debug_info["rag_context"]["schema_context"] = schema_context
                    debug_info["rag_context"]["metadata_context"] = metadata_context
                    debug_info["rag_context"]["query_examples_context"] = query_examples_context
                    debug_info["rag_context"]["complex_query_examples_context"] = complex_query_examples_context
                    
                    # Generate prompt with these contexts
                    prompt = self._create_sql_generation_prompt(user_query)
                    # Store the full system prompt
                    debug_info["system_prompt"] = prompt
                    logger.info("Generated initial SQL prompt with RAG context")
                else:
                    # Correction prompt with previous error
                    prompt = self._create_sql_correction_prompt(
                        user_query, 
                        final_result["sql_query"], 
                        final_result["error"]
                    )
                    logger.info(f"Generated correction prompt for attempt {attempts}")
                
                # Get response from LLM
                if self.callback_manager:
                    response = self.sql_llm(prompt, callbacks=self.callback_manager)
                else:
                    response = self.sql_llm(prompt)
                
                # Extract SQL query from response - handle potential formatting
                sql_query = response.strip()
                
                # Remove <code> tags if present
                if sql_query.startswith("<code>") and sql_query.endswith("</code>"):
                    sql_query = sql_query[7:-8].strip()  # 7 is length of "<code>" and 8 is length of "</code>"
                
                if "<code>" in sql_query and "</code>" in sql_query:
                    # If tags are somewhere in the string but not perfectly at start/end
                    sql_query = sql_query.replace("<code>", "").replace("</code>", "").strip()
                    
                if sql_query.startswith("s>") :
                    # If tags are somewhere in the string but not perfectly at start/end
                    sql_query = sql_query.replace("s>", "").strip()    
                
                # First, split by the first </s> token if it exists
                if "</s>" in response:
                    # Take only the content before the first </s> token
                    sql_query = sql_query.split("</s>")[0].strip()
                    
                if "[/s]" in response:
                    # Take only the content before the first </s> token
                    sql_query = sql_query.split("[/s]")[0].strip()
                    
                # Handle potential instruction tokens in response
                if "[/INST]" in sql_query:
                    sql_query = sql_query.split("[/INST]")[1].strip()
                
                # Remove any markdown formatting
                if sql_query.startswith("```sql"):
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                elif sql_query.startswith("```"):
                    sql_query = sql_query.replace("```", "").strip()
                
                logger.info(f"LLM generated SQL query: {sql_query}")
                
                # Step 2: Validate and execute the SQL query
                validation_result = self.db_manager.validate_and_execute_query(sql_query)
                
                # Step 3: Check validation result
                if validation_result["status"] == "success":
                    # Query is valid and executed successfully
                    logger.info("SQL query executed successfully")
                    
                    # Get the full results (not just the sample)
                    results, error = self.db_manager.execute_query(sql_query)
                    
                    if error:
                        final_result = {
                            "status": "execution_error",
                            "sql_query": sql_query,
                            "error": error,
                            "data": None,
                            "results": None,
                            "attempts": attempts,
                            "processing_time": time.time() - start_time,
                            "model": self.sql_llm.model
                        }
                    else:
                        # Success! Return the results
                        df = pd.DataFrame(results)
                        
                        final_result = {
                            "status": "success",
                            "sql_query": sql_query,
                            "error": None,
                            "data": df,
                            "raw_data": results,
                            "attempts": attempts,
                            "processing_time": time.time() - start_time,
                            "model": self.sql_llm.model
                        }
                        
                        # Log to LangSmith
                        if run_id and hasattr(self, 'langsmith_enabled') and self.langsmith_enabled and self.client:
                            try:
                                self.client.update_run(
                                    run_id=run_id,
                                    outputs={
                                        "status": "success",
                                        "sql_query": sql_query,
                                        "attempts": attempts,
                                        "data_sample": str(results[:5] if results else []),
                                        "model": self.sql_llm.model,
                                        "row_count": len(results)
                                    }
                                )
                            except Exception as e:
                                logger.warning(f"Failed to update LangSmith run: {str(e)}")
                        
                        # Break the loop since we have successful results
                        break
                
                else:
                    # Query has errors - store for next iteration
                    logger.info(f"SQL query validation failed: {validation_result['error']}")
                    final_result = {
                        "status": validation_result["status"],
                        "sql_query": sql_query,
                        "error": validation_result["error"],
                        "data": None,
                        "attempts": attempts,
                        "processing_time": time.time() - start_time,
                        "model": self.sql_llm.model
                    }
                    
                    # If this is the last attempt, update LangSmith with error and break the loop
                    if attempts >= max_attempts:  # Changed from == to >= for safety
                        if run_id and hasattr(self, 'langsmith_enabled') and self.langsmith_enabled and self.client:
                            try:
                                self.client.update_run(
                                    run_id=run_id,
                                    outputs={
                                        "status": "error",
                                        "sql_query": sql_query,
                                        "error": validation_result["error"],
                                        "attempts": attempts,
                                        "model": self.sql_llm.model
                                    }
                                )
                            except Exception as e:
                                logger.warning(f"Failed to update LangSmith run with error: {str(e)}")
                        # Explicitly break out of the loop when max attempts reached
                        break
            
            except Exception as e:
                # Log unexpected errors
                error_msg = f"Error in attempt {attempts}: {str(e)}"
                logger.error(error_msg)
                
                final_result = {
                    "status": "error",
                    "sql_query": sql_query if sql_query is not None else "No query generated",
                    "error": error_msg,
                    "data": None,
                    "attempts": attempts,
                    "processing_time": time.time() - start_time,
                    "model": self.sql_llm.model
                }
                
                # If this is the last attempt, update LangSmith with error and break
                if attempts >= max_attempts:  # Changed from == to >= for safety
                    if run_id and hasattr(self, 'langsmith_enabled') and self.langsmith_enabled and self.client:
                        try:
                            self.client.update_run(
                                run_id=run_id,
                                outputs={
                                    "status": "error",
                                    "error": error_msg,
                                    "attempts": attempts,
                                    "model": self.sql_llm.model
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update LangSmith run with error: {str(e)}")
                    # Explicitly break out of the loop when max attempts reached
                    break
        
        # Ensure we have a final result, even if there was a problem
        if final_result is None:
            final_result = {
                "status": "error",
                "sql_query": sql_query if sql_query is not None else "No query generated",
                "error": "Failed to generate or execute a valid SQL query after maximum attempts",
                "data": None,
                "attempts": attempts,
                "processing_time": time.time() - start_time,
                "model": self.sql_llm.model
            }
        
        # Complete the LangSmith run
        if run_id and hasattr(self, 'langsmith_enabled') and self.langsmith_enabled and self.client:
            try:
                self.client.update_run(
                    run_id=run_id,
                    outputs={
                        "status": final_result["status"],
                        "attempts": attempts,
                        "processing_time": final_result["processing_time"],
                        "model": self.sql_llm.model
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update LangSmith run: {str(e)}")
        
        if final_result is not None:
            final_result["debug_info"] = debug_info
        
        return final_result
    
    def display_results(self, result: Dict) -> None:
        """Display the results of the SQL chain execution."""
        if result["status"] == "success":
            print(f"✅ SQL Query successfully executed after {result['attempts']} attempt(s):\n")
            print(f"SQL Query:\n{result['sql_query']}\n")
            print(f"Results ({len(result['raw_data'])} rows):")
            print(result["data"])
        else:
            print(f"❌ SQL Query generation failed after {result['attempts']} attempt(s):\n")
            print(f"SQL Query:\n{result['sql_query']}\n")
            print(f"Error: {result['error']}")