import json
import logging
from typing import Dict, List, Optional, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from backend.config import Config, logger

class RAGManager:
    """Manages Retrieval Augmented Generation components for JSON documents."""
    
    def __init__(self):
        """Initialize the RAG Manager with all required components."""
        logger.info("Initializing RAGManager with JSON documents...")
        
        # Initialize embedding model - optimized for 16GB GPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 8, "normalize_embeddings": True}
        )
        
        # Load JSON files directly first and store as attributes
        try:
            import json
            with open(Config.SCHEMA_PATH, 'r') as file:
                self.schema_json = json.load(file)
                logger.info(f"Loaded schema JSON file: {Config.SCHEMA_PATH}")
                
            with open(Config.QUERY_PATH, 'r') as file:
                self.query_examples = json.load(file)
                logger.info(f"Loaded query examples JSON file: {Config.QUERY_PATH}")

            with open(Config.COMPLEX_QUERY_PATH, 'r') as file:
                self.complex_query_examples = json.load(file)
                logger.info(f"Loaded complex query examples JSON file: {Config.COMPLEX_QUERY_PATH}")
                                
        except Exception as e:
            logger.error(f"Error loading JSON files: {str(e)}")
            self.schema_json = {}
            self.query_examples = {}
        
        # Set up vectorstores for different document types
        logger.info("Setting up vector stores...")
        
        # Schema vectorstore for table-level info
        self.schema_vectorstore = self._setup_schema_vectorstore(
            self.schema_json, 
            "schema_table_info"
        )
        
        # Separate vectorstore for column-level info
        self.column_vectorstore = self._setup_column_vectorstore(
            self.schema_json, 
            "schema_column_info"
        )
        
        # Query examples vectorstore
        self.query_vectorstore = self._setup_query_vectorstore(
            self.query_examples, 
            "query_json"
        )
        
        # Query examples vectorstore
        self.complex_query_vectorstore = self._setup_query_vectorstore(
            self.complex_query_examples, 
            "complex_query_json"
        )
        
        logger.info("RAGManager initialization complete.")
    
    def filter_complex_metadata(self, metadata):
        """Convert complex metadata types to strings to prevent vectorstore errors."""
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value
            elif isinstance(value, list):
                filtered[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                filtered[key] = json.dumps(value)
            else:
                filtered[key] = str(value)
        return filtered
    
    def _setup_schema_vectorstore(self, schema_data: dict, store_name: str) -> Chroma:
        """Set up vector store for table and database level schema information."""
        try:
            # Log the setup process
            logger.info(f"Setting up vectorstore for {store_name} from schema data")
            
            # Create documents from schema structure - ONLY for table-level and database-level info
            documents = []
            
            # Database overview document
            db_doc = Document(
                page_content=f"Database: {schema_data.get('database', 'Insurance')}\n" +
                             f"Tables: {', '.join([t['name'] for t in schema_data.get('tables', [])])}\n" +
                             f"Purpose: {schema_data.get('metadata', {}).get('purpose', '')}\n" +
                             f"Business Domain: {schema_data.get('metadata', {}).get('business_domain', 'Insurance')}\n" +
                             f"Primary Keys: {schema_data.get('metadata', {}).get('primary_keys', '')}",
                metadata=self.filter_complex_metadata({
                    "source": "schema", 
                    "type": "database_overview", 
                    "id": "db_overview"
                })
            )
            documents.append(db_doc)
            
            # Relationship documents
            for i, rel in enumerate(schema_data.get("relationships", [])):
                rel_doc = Document(
                    page_content=f"Relationship Type: {rel.get('type', '')}\n" +
                                 f"Tables: {', '.join(rel.get('tables', []))}\n" +
                                 f"Columns: {', '.join(rel.get('columns', []))}\n" +
                                 f"Description: {rel.get('description', '')}\n" +
                                 f"Join Hint: These tables can be joined using the specified columns.",
                    metadata=self.filter_complex_metadata({
                        "source": "schema", 
                        "type": "relationship", 
                        "id": f"relationship_{i}"
                    })
                )
                documents.append(rel_doc)
            
            # Table documents - but NOT column documents
            for table in schema_data.get("tables", []):
                table_name = table.get("name", "")
                
                # Combine table descriptions
                table_desc = ""
                if isinstance(table.get("description"), list):
                    table_desc = "\n".join(table.get("description", []))
                else:
                    table_desc = str(table.get("description", ""))
                
                # Table synonyms
                table_synonyms = table.get("synonyms", [])
                if isinstance(table_synonyms, list):
                    table_synonyms_str = ", ".join(table_synonyms)
                else:
                    table_synonyms_str = str(table_synonyms)
                
                # Table definition document
                table_doc = Document(
                    page_content=f"Table: {table_name}\n" +
                                 f"Schema: {table.get('schema', 'dbo')}\n" +
                                 f"Description: {table_desc}\n" +
                                 f"Synonyms: {table_synonyms_str}\n" +
                                 f"Columns: {', '.join([c['name'] for c in table.get('columns', [])])}",
                    metadata=self.filter_complex_metadata({
                        "source": "schema", 
                        "type": "table", 
                        "table_name": table_name, 
                        "synonyms": table_synonyms_str,
                        "id": f"table_{table_name}"
                    })
                )
                documents.append(table_doc)
            
            logger.info(f"Created {len(documents)} document chunks for {store_name}")
            
            if not documents:
                raise ValueError(f"No documents created from schema data")
            
            # Create vector store from documents
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=f"{Config.VECTOR_DB_PATH}/{store_name}"
            )
        except Exception as e:
            logger.error(f"Error setting up {store_name} vectorstore: {str(e)}")
            # Return empty vectorstore
            return Chroma.from_texts(
                texts=[f"{store_name} document not available"],
                embedding=self.embeddings,
                persist_directory=f"{Config.VECTOR_DB_PATH}/{store_name}_fallback"
            )
    
    def _setup_column_vectorstore(self, schema_data: dict, store_name: str) -> Chroma:
        """Set up vector store SPECIFICALLY for column-level information."""
        try:
            # Log the setup process
            logger.info(f"Setting up vectorstore for {store_name} specifically for column data")
            
            # Create documents ONLY for column information
            documents = []
            
            # Process tables to extract column information
            for table in schema_data.get("tables", []):
                table_name = table.get("name", "")
                
                # Process each column in the table
                for column in table.get("columns", []):
                    col_name = column.get("name", "")
                    col_type = column.get("type", "")
                    business_meaning = column.get("business_meaning", "")
                    description = column.get("description", "")
                    
                    # Handle synonyms properly
                    synonyms = column.get("synonyms", [])
                    synonyms_str = ", ".join(synonyms) if isinstance(synonyms, list) else str(synonyms)
                    
                    # Handle possible values
                    possible_values = column.get("possible_values", [])
                    possible_values_str = ""
                    if possible_values:
                        if isinstance(possible_values, list):
                            possible_values_str = f"Possible Values: {', '.join(possible_values)}\n"
                        else:
                            possible_values_str = f"Possible Values: {possible_values}\n"
                    
                    # Handle prompt triggers for better time dimension recognition
                    prompt_triggers = column.get("prompt_triggers", [])
                    prompt_triggers_str = ""
                    if prompt_triggers:
                        if isinstance(prompt_triggers, list):
                            prompt_triggers_str = f"Prompt Triggers: {', '.join(prompt_triggers)}\n"
                        else:
                            prompt_triggers_str = f"Prompt Triggers: {prompt_triggers}\n"
                    
                    # Special handling for time-based columns - critical for January 2025 type queries
                    is_time_column = (
                        "date" in col_type.lower() or 
                        "time" in col_type.lower() or 
                        "datetime" in col_name.lower() or 
                        col_name.lower() == "transaction_date"
                    )
                    
                    time_info = ""
                    if is_time_column:
                        time_info = "TIME DIMENSION COLUMN: This column can be used for filtering by specific time periods.\n"
                        time_info += "Can filter for: specific months (January, February, etc.), years (2024, 2025), quarters, or date ranges.\n"
                        time_info += "Use this column for all time-based queries including month-specific and year-specific analysis.\n"
                        time_info += "Common time operations: YEAR(), MONTH(), DAY(), DATEPART(), DATEDIFF()\n"
                    
                    # Enhanced business context for time columns
                    if col_name.lower() == "transaction_date":
                        time_info += "The Transaction_Date is critical for any time-period analysis, especially for:\n"
                        time_info += "- Filtering transactions by specific months like January 2025\n"
                        time_info += "- Calculating growth rates between time periods\n"
                        time_info += "- Analyzing trends over time by month, quarter or year\n"
                        time_info += "- Time-series forecasting and historical performance analysis\n"
                    
                    # Create document with rich column information
                    column_doc = Document(
                        page_content=f"Table: {table_name}\n" +
                                     f"Column: {col_name}\n" +
                                     f"Type: {col_type}\n" +
                                     f"Business Meaning: {business_meaning}\n" +
                                     f"Description: {description}\n" +
                                     f"Synonyms: {synonyms_str}\n" +
                                     possible_values_str +
                                     prompt_triggers_str +
                                     time_info,
                        metadata=self.filter_complex_metadata({
                            "source": "schema", 
                            "type": "column", 
                            "table_name": table_name, 
                            "column_name": col_name,
                            "synonyms": synonyms,
                            "prompt_triggers": prompt_triggers,
                            "is_time_column": is_time_column,
                            "id": f"column_{table_name}_{col_name}"
                        })
                    )
                    documents.append(column_doc)
                    
                    # For transaction_date, create additional dedicated time dimension document
                    if col_name.lower() == "transaction_date":
                        time_dimension_doc = Document(
                            page_content=f"CRITICAL TIME DIMENSION: Transaction_Date in {table_name}\n" +
                                        f"This date column is essential for filtering by time periods like months, years, and quarters.\n" +
                                        f"Type: {col_type}\n" +
                                        f"Business Use: Used to filter transactions by specific time periods such as January 2025.\n" +
                                        f"Common Filters: MONTH(Transaction_Date) = 1 for January, YEAR(Transaction_Date) = 2025 for year 2025.\n" +
                                        f"Query Patterns: Filtering for specific months (January, February, etc.), years (2024, 2025), or date ranges.",
                            metadata=self.filter_complex_metadata({
                                "source": "schema",
                                "type": "time_dimension",
                                "table_name": table_name,
                                "column_name": col_name,
                                "is_time_column": True,
                                "id": f"time_dimension_{table_name}_{col_name}"
                            })
                        )
                        documents.append(time_dimension_doc)
            
            logger.info(f"Created {len(documents)} column document chunks for {store_name}")
            
            if not documents:
                raise ValueError(f"No column documents created from schema data")
            
            # Create vector store from documents
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=f"{Config.VECTOR_DB_PATH}/{store_name}"
            )
        except Exception as e:
            logger.error(f"Error setting up {store_name} vectorstore: {str(e)}")
            # Return empty vectorstore
            return Chroma.from_texts(
                texts=[f"{store_name} column document not available"],
                embedding=self.embeddings,
                persist_directory=f"{Config.VECTOR_DB_PATH}/{store_name}_fallback"
            )
        
    def _setup_query_vectorstore(self, query_data: dict, store_name: str) -> Chroma:
        """Set up vector store for query examples with enhanced metadata."""
        try:
            # Log the setup process
            logger.info(f"Setting up vectorstore for {store_name} from query examples data")
            
            # Create documents from query examples structure
            documents = []
            
            # Process query examples
            examples = query_data.get("query_examples", [])
            categories = query_data.get("metadata", {}).get("categories", [])
            
            # Create category documents to help understand query types
            for i, category in enumerate(categories):
                category_doc = Document(
                    page_content=f"Category: {category}\n" +
                                f"Description: Queries related to {category.lower()}",
                    metadata=self.filter_complex_metadata({
                        "source": "query_examples",
                        "type": "category",
                        "category": category,
                        "id": f"category_{i}"
                    })
                )
                documents.append(category_doc)
            
            # Create a document for each query example
            for example in examples:
                example_id = example.get("id", 0)
                category = example.get("category", "Unknown")
                prompt = example.get("prompt", "")
                sql = example.get("sql", "")
                description = example.get("description", "")
                
                # Create rich document with all example information
                example_doc = Document(
                    page_content=f"Prompt: {prompt}\n" +
                                f"SQL: {sql}\n" +
                                f"Category: {category}\n" +
                                f"Description: {description}",
                    metadata=self.filter_complex_metadata({
                        "source": "query_examples",
                        "type": "example",
                        "category": category,
                        "id": f"example_{example_id}",
                        "example_id": example_id
                    })
                )
                documents.append(example_doc)
            
            logger.info(f"Created {len(documents)} document chunks for {store_name}")
            
            if not documents:
                raise ValueError(f"No documents created from query examples data")
            
            # Create vector store from documents
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=f"{Config.VECTOR_DB_PATH}/{store_name}"
            )
        except Exception as e:
            logger.error(f"Error setting up {store_name} vectorstore: {str(e)}")
            # Return empty vectorstore as fallback
            return Chroma.from_texts(
                texts=[f"{store_name} document not available"],
                embedding=self.embeddings,
                persist_directory=f"{Config.VECTOR_DB_PATH}/{store_name}_fallback"
            )
    
    def get_schema_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve database and table-level schema information based on query.
        Focuses on providing database overview, table definitions, and relationships.
        """
        # Check if this is a time-based query for special handling
        time_keywords = ["january", "february", "march", "april", "may", "june", "july", 
                        "august", "september", "october", "november", "december",
                        "2023", "2024", "2025", "year", "month", "quarter", "day", "date",
                        "period", "weekly", "monthly", "yearly", "daily", "growth"]
        
        is_time_query = any(keyword in query.lower() for keyword in time_keywords)
        
        # Get relevant chunks from schema vector store
        docs_with_scores = self.schema_vectorstore.similarity_search_with_score(query, k=k*3)
        
        # Create a dictionary to categorize docs by type
        doc_by_type = {
            "database_overview": [],
            "table": [],
            "relationship": []
        }
        
        # Categorize documents by type
        for doc, score in docs_with_scores:
            doc_type = doc.metadata.get("type", "unknown")
            if doc_type in doc_by_type:
                doc_by_type[doc_type].append((doc, score))
        
        # Sort each category by score
        for doc_type in doc_by_type:
            doc_by_type[doc_type].sort(key=lambda x: x[1])  # Lower score is better
        
        # Build the final document list with proper prioritization
        final_docs = []
        
        # 1. First add database overview (1 document)
        if doc_by_type["database_overview"]:
            final_docs.append(doc_by_type["database_overview"][0])
        
        # 2. Add relationship documents (prioritize them to ensure joins are included)
        # Include more relationship docs (up to 2 or 3 based on k)
        rel_count = min(len(doc_by_type["relationship"]), max(2, k//3))
        final_docs.extend(doc_by_type["relationship"][:rel_count])
        
        # 3. Add table documents (use remaining slots)
        remaining_slots = k - len(final_docs)
        if remaining_slots > 0 and doc_by_type["table"]:
            # Add top table docs
            final_docs.extend(doc_by_type["table"][:remaining_slots])
        
        # Ensure we don't exceed k documents
        final_docs = final_docs[:k]
        
        # Format the schema context string
        schema_str = "DATABASE SCHEMA INFORMATION:\n\n"
        
        # If we have no documents, provide a default message
        if not final_docs:
            schema_str += "No relevant schema information found for the query.\n"
            return schema_str
        
        # Add formatted content for each document
        for i, (doc, score) in enumerate(final_docs, 1):
            doc_type = doc.metadata.get("type", "unknown")
            
            # Calculate similarity percentage
            similarity_pct = (1.0 - score) * 100
            
            # Format based on document type
            if doc_type == "database_overview":
                schema_str += f"DATABASE OVERVIEW - Similarity: {similarity_pct:.2f}%\n"
                schema_str += f"{doc.page_content}\n\n"
                
            elif doc_type == "table":
                table_name = doc.metadata.get("table_name", "Unknown")
                schema_str += f"TABLE: {table_name} - Similarity: {similarity_pct:.2f}%\n"
                schema_str += f"{doc.page_content}\n\n"
                
            elif doc_type == "relationship":
                schema_str += f"TABLE RELATIONSHIP - Similarity: {similarity_pct:.2f}%\n"
                schema_str += f"{doc.page_content}\n\n"
                
            else:
                schema_str += f"SCHEMA COMPONENT ({doc_type}) - Similarity: {similarity_pct:.2f}%\n"
                schema_str += f"{doc.page_content}\n\n"
        
        # If this is a time query, add a special note about time-based filtering
        if is_time_query:
            schema_str += "NOTE: For time-based filtering and analysis, see the METADATA CONTEXT for detailed information about date columns like Transaction_Date.\n\n"
        
        return schema_str
    
    def get_metadata_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve column-level metadata with detailed information about columns,
        their data types, descriptions, synonyms, and possible values.
        Special handling for time-based columns when time keywords are present.
        """
        # Extract keywords from the query for better matching
        keywords = query.lower().split()
        
        # Check for time-based keywords
        time_keywords = ["january", "february", "march", "april", "may", "june", "july", 
                        "august", "september", "october", "november", "december",
                        "2023", "2024", "2025", "year", "month", "quarter", "day", "date",
                        "period", "weekly", "monthly", "yearly", "daily", "growth"]
        
        has_time_keywords = any(kw in query.lower() for kw in time_keywords)
        

        # This is the critical fix, as column_vectorstore contains only column documents
        docs_with_scores = self.column_vectorstore.similarity_search_with_score(
            query, k=k*3  # Get more than we need for better filtering
        )
        
        # Initialize prioritized document list
        prioritized_docs = []
        
        # STEP 1: First prioritize transaction_date for time-based queries
        if has_time_keywords:
            for doc, score in docs_with_scores:
                column_name = doc.metadata.get("column_name", "").lower()
                doc_type = doc.metadata.get("type", "")
                
                # Add transaction_date with highest priority
                if column_name == "transaction_date" or doc_type == "time_dimension":
                    prioritized_docs.append((doc, score, 0))  # Priority 0 (highest)
        
        # STEP 2: Add all column documents with appropriate priorities
        for doc, score in docs_with_scores:
            doc_type = doc.metadata.get("type", "")
            
            # Skip if we already added this document
            if any(d.metadata.get("id") == doc.metadata.get("id") for d, _, _ in prioritized_docs):
                continue
            
            # Process based on document type
            if doc_type == "column":
                column_name = doc.metadata.get("column_name", "").lower()
                
                # Check for keyword matches in column name
                direct_match = any(kw in column_name for kw in keywords)
                
                # Get synonyms from metadata
                synonyms_str = doc.metadata.get("synonyms", "").lower()
                synonyms = [s.strip() for s in synonyms_str.split(",") if s.strip()]
                synonym_match = any(any(kw in syn for kw in keywords) for syn in synonyms)
                
                # Assign priority based on match type
                if direct_match:
                    prioritized_docs.append((doc, score, 1))  # Direct column match
                elif synonym_match:
                    prioritized_docs.append((doc, score, 2))  # Synonym match
                elif doc.metadata.get("is_time_column", "False") == "True" and has_time_keywords:
                    prioritized_docs.append((doc, score, 3))  # Time column for time query
                else:
                    prioritized_docs.append((doc, score, 4))  # Any other column
            
            elif doc_type == "time_dimension":
                # Add time dimension docs with high priority for time queries
                if has_time_keywords:
                    prioritized_docs.append((doc, score, 0.5))  # Very high priority
                else:
                    prioritized_docs.append((doc, score, 3.5))  # Medium priority
        
        # Sort by priority first, then by score
        prioritized_docs.sort(key=lambda x: (x[2], x[1]))
        
        # If we have too few documents, just use the top scoring ones
        if len(prioritized_docs) < k:
            # Sort by score and take top ones
            docs_by_score = sorted(docs_with_scores, key=lambda x: x[1])
            
            # Add any that we don't already have
            for doc, score in docs_by_score:
                if not any(d.metadata.get("id") == doc.metadata.get("id") for d, _, _ in prioritized_docs):
                    prioritized_docs.append((doc, score, 9))  # Low priority fallback
                
                # Break if we have enough
                if len(prioritized_docs) >= k:
                    break
        
        # Take top k documents for the final result
        top_docs = prioritized_docs[:k]
        
        # Format the metadata context string
        metadata_str = "DATABASE COLUMN METADATA:\n\n"
        
        # If we have no documents, provide a default message with better debugging
        if not top_docs:
            metadata_str += "No relevant column metadata found for the query.\n"
            # Add counts of document types from the column vectorstore
            doc_types_count = {}
            for doc, _ in docs_with_scores:
                doc_type = doc.metadata.get("type", "unknown")
                doc_types_count[doc_type] = doc_types_count.get(doc_type, 0) + 1
            
            metadata_str += f"Debug: Available column document types: {doc_types_count}\n"
            return metadata_str
        
        # Add formatted content for each document
        for i, (doc, score, priority) in enumerate(top_docs, 1):
            doc_type = doc.metadata.get("type", "")
            
            # Calculate similarity percentage
            similarity_pct = (1.0 - score) * 100
            
            # Format based on document type
            if doc_type == "column":
                # Get column details
                table_name = doc.metadata.get("table_name", "Unknown")
                column_name = doc.metadata.get("column_name", "Unknown")
                
                metadata_str += f"COLUMN: {column_name} (Table: {table_name}) - Similarity: {similarity_pct:.2f}%\n"
                metadata_str += f"{doc.page_content}\n"
                
                # If this is a time column and time keywords are present, add extra note
                is_time_column = doc.metadata.get("is_time_column", "False")
                if has_time_keywords and (is_time_column == "True" or is_time_column is True):
                    metadata_str += "NOTE: This time column can be used for filtering by specific time periods.\n"
                
                metadata_str += "\n"
                
            elif doc_type == "time_dimension":
                # Handle time dimension documents
                metadata_str += f"TIME DIMENSION - Similarity: {similarity_pct:.2f}%\n"
                metadata_str += f"{doc.page_content}\n\n"
                
            else:
                # Generic fallback format
                metadata_str += f"METADATA ({doc_type}) - Similarity: {similarity_pct:.2f}%\n"
                metadata_str += f"{doc.page_content}\n\n"
        
        # If this is a time query but we don't have transaction_date, add a hint
        if has_time_keywords and not any("transaction_date" in d.metadata.get("column_name", "").lower() 
                                         for d, _, _ in top_docs):
            metadata_str += "\nTIME ANALYSIS HINT: For time-based analysis, consider using the Transaction_Date column.\n"
        
        return metadata_str
    
    def get_query_examples_context(self, query: str, complex_query: bool = True, k: int = 5) -> str:
        """Retrieve relevant query examples using both vector similarity and keyword matching."""
        # Get relevant chunks from query examples vector store with scores
        if complex_query:
            docs_with_scores = self.complex_query_vectorstore.similarity_search_with_score(query, k=k*2)
        else:
            docs_with_scores = self.query_vectorstore.similarity_search_with_score(query, k=k*2)
        
        # Filter to focus on example documents (not category summaries)
        example_docs = [(doc, score) for doc, score in docs_with_scores 
                       if doc.metadata.get("type") == "example"]
        
        # Deduplicate by example ID to avoid showing the same example twice
        seen_ids = set()
        unique_examples = []
        
        for doc, score in example_docs:
            example_id = doc.metadata.get("id", "")
            if example_id and example_id not in seen_ids:
                seen_ids.add(example_id)
                unique_examples.append((doc, score))
        
        # Check for time-based queries to get specific examples
        time_keywords = ["january", "february", "march", "april", "may", "june", "july", 
                        "august", "september", "october", "november", "december",
                        "2023", "2024", "2025", "year", "month", "quarter", "day", "date", "period"]
        
        has_time_keywords = any(kw in query.lower() for kw in time_keywords)
        
        # Boost time-based examples for time queries
        if has_time_keywords:
            for i, (doc, score) in enumerate(unique_examples):
                if any(kw in doc.page_content.lower() for kw in time_keywords):
                    # Boost score for time-related examples (lower score is better)
                    unique_examples[i] = (doc, score * 0.7)
        
        # Ensure category diversity by grouping by category
        category_examples = {}
        for doc, score in unique_examples:
            category = doc.metadata.get("category", "unknown")
            if category not in category_examples:
                category_examples[category] = []
            category_examples[category].append((doc, score))
        
        # Get top examples from each category
        diverse_examples = []
        
        # Check for time-based analysis category first
        if has_time_keywords and "Year-over-Year Analysis" in category_examples:
            year_examples = category_examples["Year-over-Year Analysis"]
            year_examples.sort(key=lambda x: x[1])  # Sort by score
            diverse_examples.append(year_examples[0])  # Add top example
        
        # Then get examples from each category
        for category, examples in category_examples.items():
            # Skip if we already added from this category
            if any(d.metadata.get("category") == category for d, _ in diverse_examples):
                continue
                
            # Sort by score
            examples.sort(key=lambda x: x[1])
            # Add top example from each category
            diverse_examples.append(examples[0])
            # If we have more than one example in this category, add another
            if len(examples) > 1 and len(diverse_examples) < k:
                diverse_examples.append(examples[1])
        
        # If we still need more examples, add them based on score
        if len(diverse_examples) < k:
            # Get examples we haven't added yet
            remaining_examples = []
            for doc, score in unique_examples:
                example_id = doc.metadata.get("id", "")
                if example_id not in [d.metadata.get("id", "") for d, _ in diverse_examples]:
                    remaining_examples.append((doc, score))
            
            # Sort by score and add remaining
            remaining_examples.sort(key=lambda x: x[1])
            diverse_examples.extend(remaining_examples[:k-len(diverse_examples)])
        
        # Take top k after filtering for diversity
        top_examples = diverse_examples[:k]
        
        # Format the output with enhanced information
        query_str = "DATABASE QUERY EXAMPLES:\n\n"
        query_str += f"Query: '{query}'\n\n"
        
        for i, (doc, score) in enumerate(top_examples, 1):
            # Convert score to percentage similarity
            similarity_pct = (1.0 - score) * 100  # ChromaDB returns distance, not similarity
            
            # Extract components from the document content
            content_lines = doc.page_content.split('\n')
            prompt = next((line.replace("Prompt: ", "") for line in content_lines if line.startswith("Prompt: ")), "")
            sql = next((line.replace("SQL: ", "") for line in content_lines if line.startswith("SQL: ")), "")
            description = next((line.replace("Description: ", "") for line in content_lines if line.startswith("Description: ")), "")
            category = next((line.replace("Category: ", "") for line in content_lines if line.startswith("Category: ")), "")
            
            query_str += f"Example {i} - Similarity: {similarity_pct:.2f}%\n"
            query_str += f"Prompt: {prompt}\n"
            query_str += f"SQL: {sql}\n"
            query_str += f"Category: {category}\n"
            query_str += f"Description: {description}\n\n"
        
        return query_str