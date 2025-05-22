# LLMRAG
A text to SQL bot using open source LLMs and RAG

# LLMRAG - RAG-Powered SQL Chatbot

A sophisticated text-to-SQL bot using open source LLMs and Retrieval-Augmented Generation (RAG) to convert natural language queries into SQL statements and provide intelligent database interactions.

## 🚀 Features

- **Natural Language to SQL**: Convert plain English questions into SQL queries using advanced LLMs
- **RAG Pipeline**: Intelligent document retrieval with context-aware responses using vector databases
- **Multi-Database Support**: Compatible with SQL Server, MySQL, and PostgreSQL
- **Excel Integration**: Import Excel files directly to SQL Server databases with automated schema detection
- **Interactive Web Interface**: Modern Streamlit-based frontend with real-time chat functionality
- **FastAPI Backend**: RESTful API powered by LangServe for scalable deployment
- **Multiple LLM Support**: Uses Together API with support for Llama, Qwen, and DeepSeek models
- **Vector Database**: ChromaDB and FAISS for efficient document retrieval and embedding storage
- **Advanced Monitoring**: Built-in observability with LangSmith integration
- **Smart Query Correction**: Automatic SQL query refinement with retry logic
- **Comprehensive Debugging**: Detailed logging and debug information for troubleshooting

## 🏗️ Architecture

The system follows a modular RAG workflow with six key stages:

1. **Input Processing**: User prompts are processed and enhanced with system context
2. **RAG Pipeline**: 
   - Document retrieval from vector database
   - Context chunking and embedding generation
   - Relevant context passing to LLM
3. **SQL Generation**: Advanced LLMs (Llama, Qwen, DeepSeek) convert natural language to SQL
4. **Data Processing**: Query execution, validation, and result processing
5. **Output Generation**: LLM summarizes results and generates human-readable responses
6. **Visualization**: Interactive charts and data visualization with export capabilities

## 📁 Project Structure

```
LLMRAG/
├── main.py                     # Main application launcher
├── CreateDatabase.py           # Excel to SQL Server import utility
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── backend/                    # API backend (LangServe + FastAPI)
│   ├── __init__.py            
│   ├── config.py              # Configuration management
│   ├── database.py            # Database connection & query execution
│   ├── rag_manager.py         # RAG pipeline and vector store management
│   ├── llm_chain.py           # LLM integration and SQL generation
│   └── langserve_app.py       # FastAPI application with REST endpoints
├── frontend/                   # Web interface (Streamlit)
│   └── streamlit_app.py       # Interactive chat interface
├── data/                       # RAG documents and vector databases
│   ├── Schema.json            # Database schema information
│   ├── Prompt_Query_Examples.json
│   ├── Complex_Query_Examples.json
│   └── vector_db/             # ChromaDB vector stores
└── .env                       # Environment variables (create this)
```

## 📋 Prerequisites

- **Python 3.8+** with pip
- **SQL Server** (with appropriate ODBC drivers)
- **Together API Account** (for LLM inference)
- **Git** for version control
- **16GB+ RAM** recommended for optimal performance
- **CUDA-compatible GPU** (optional, for faster embeddings)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd LLMRAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:

```env
# LLM Configuration
TOGETHER_API_KEY=your_together_api_key_here
DEFAULT_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

# Database Configuration
DB_TYPE=mssql
DB_HOST=localhost\SQLEXPRESS
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_TRUSTED_CONNECTION=0  # Set to 1 for Windows Authentication

# Vector Database
VECTOR_DB_PATH=data/vector_db

# RAG Document Paths
SCHEMA_PATH=data/Schema.json
QUERY_PATH=data/Prompt_Query_Examples.json
COMPLEX_QUERY_PATH=data/Complex_Query_Examples.json

# LangSmith (Optional - for monitoring)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=LLM_RAG
LANGCHAIN_TRACING_V2=true

# API Configuration
LANGSERVE_HOST=0.0.0.0
LANGSERVE_PORT=8010
```

### 5. Prepare RAG Documents
Ensure you have the following JSON files in the `data/` directory:
- `Schema.json` - Database schema and table relationships
- `Prompt_Query_Examples.json` - Simple query examples
- `Complex_Query_Examples.json` - Advanced query patterns

### 6. Set Up Database Connection
Install SQL Server ODBC drivers if not already installed:
- **Windows**: Usually pre-installed
- **Linux**: Install `msodbcsql17` or `msodbcsql18`
- **macOS**: Install via Homebrew

## 🚀 Quick Start

### Option 1: Run Complete Application
```bash
python main.py
```
This starts both the backend API and frontend interface.

### Option 2: Run Components Separately
```bash
# Terminal 1: Start backend API
python main.py --backend-only

# Terminal 2: Start frontend interface
python main.py --frontend-only
```

### Option 3: Import Excel Data First
```bash
python CreateDatabase.py
```
Follow the prompts to import your Excel files into SQL Server.

## 🎯 Usage Examples

### Database Import
1. Run `python CreateDatabase.py`
2. Provide Excel file path and database connection details
3. Choose authentication method (Windows or SQL Server)
4. The script creates tables for each Excel sheet with optimized data types

### Natural Language Queries
Once the application is running (default: http://localhost:8501), try queries like:

- **Simple Queries**:
  - "Show me all policies from January 2025"
  - "What's the total premium amount by segment?"
  - "List customers with claims over $10,000"

- **Complex Analysis**:
  - "Compare premium amounts between traditional and digital channels"
  - "Show month-over-month growth in policy count for 2024"
  - "Generate a report of top 10 customers by premium amount"

- **Time-based Analysis**:
  - "Analyze quarterly trends in claim amounts"
  - "Show year-over-year comparison of policy renewals"
  - "What was the average claim amount in Q4 2024?"

## 🔧 Configuration

### LLM Models
Available models through Together API:
- **Llama 3.3 70B**: Best for complex reasoning and SQL generation
- **Qwen 2.5 Coder 32B**: Optimized for code generation
- **DeepSeek R1 70B**: Alternative high-performance model

### Vector Database Setup
The system uses ChromaDB with separate vector stores for:
- **Schema Store**: Database structure and relationships
- **Column Store**: Detailed column metadata and descriptions
- **Query Examples**: Simple and complex query patterns

### Database Support
- **Primary**: Microsoft SQL Server (2016+)
- **Experimental**: MySQL, PostgreSQL
- **Authentication**: Windows Authentication or SQL Server Authentication

## 📊 API Endpoints

The FastAPI backend provides these REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status and health check |
| `/health` | GET | Detailed health information |
| `/models` | GET | List available LLM models |
| `/execute_query` | POST | Generate and execute SQL from natural language |

### Example API Request
```bash
curl -X POST "http://localhost:8010/execute_query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Show me total premium by segment",
       "model": "llama3-70b",
       "max_attempts": 3
     }'
```

## 🔍 Advanced Features

### Smart Query Correction
- Automatic SQL syntax validation
- Context-aware error correction
- Multi-attempt query refinement
- Detailed error reporting

### RAG Context Retrieval
- **Schema Context**: Table structures and relationships
- **Metadata Context**: Column descriptions and business meanings
- **Query Examples**: Pattern matching for similar queries
- **Semantic Search**: Vector similarity for relevant context

### Debug Information
The interface provides comprehensive debugging:
- Generated SQL queries with syntax highlighting
- RAG context and system prompts
- Processing metrics and timing
- Raw API request/response data
- Token usage estimates

## 🛡️ Security Considerations

- **Input Validation**: All queries are validated before execution
- **Query Restrictions**: Only SELECT and WITH statements allowed
- **Connection Security**: Uses TrustServerCertificate for secure connections
- **API Keys**: Environment variable storage for sensitive data
- **SQL Injection Protection**: Parameterized queries and input sanitization

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   ```bash
   # Check SQL Server service status
   # Verify connection string in .env
   # Ensure ODBC drivers are installed
   ```

2. **Together API Issues**:
   ```bash
   # Verify API key in .env file
   # Check API quota and rate limits
   # Test connectivity to Together API
   ```

3. **Vector Database Issues**:
   ```bash
   # Clear vector database cache
   rm -rf data/vector_db/*
   # Restart application to rebuild indexes
   ```

4. **Import Errors**:
   ```bash
   # Check Excel file permissions
   # Verify column names and data types
   # Ensure database write permissions
   ```

### Debug Mode
Enable verbose logging:
```bash
export LANGCHAIN_VERBOSE=true
export LANGSMITH_TRACING=true
```

### Performance Optimization
- Use GPU acceleration for embeddings when available
- Increase batch size for large data imports
- Optimize vector database chunk sizes
- Monitor memory usage during processing

## 📈 Monitoring and Observability

### LangSmith Integration
- Real-time trace monitoring
- Query performance analytics
- Error tracking and debugging
- Model performance comparison

### Metrics Tracked
- Query execution time
- SQL generation accuracy
- Token usage and costs
- Error rates and types
- User interaction patterns

## 🤝 Contributing

1. **Fork the Repository**
   ```bash
   git fork <repository-url>
   git clone <your-fork-url>
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes and Test**
   ```bash
   # Add your changes
   git add .
   git commit -m "Add: your feature description"
   ```

4. **Submit Pull Request**
   ```bash
   git push origin feature/your-feature-name
   # Create PR via GitHub interface
   ```

### Development Guidelines
- Follow PEP 8 for Python code style
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes
- Test with multiple database configurations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework and LLM orchestration
- **Together AI** for high-performance LLM inference
- **Streamlit** for the intuitive web interface
- **FastAPI** for the robust backend API
- **ChromaDB** for efficient vector storage
- **HuggingFace** for state-of-the-art embeddings

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check inline code documentation
- **Community**: Join discussions in GitHub Discussions
- **API Documentation**: Visit `/docs` endpoint when running

## 🔄 Version History

- **v1.0.0**: Initial release with core RAG functionality
- **v1.1.0**: Added Together API integration and multiple model support
- **v1.2.0**: Enhanced UI with debug information and monitoring
- **v1.3.0**: Improved query correction and Excel import features

---

**Built with ❤️ using cutting-edge open-source technologies**

*Transform your data interactions with the power of AI and RAG*
