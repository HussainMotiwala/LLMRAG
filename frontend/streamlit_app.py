import streamlit as st
import pandas as pd
import requests
import json
import time
import datetime
from typing import Dict, Any, List, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="SQL Query Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LangServe API settings
API_URL = "http://localhost:8010"
DEFAULT_TIMEOUT = 180 

# Session state initialization
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'query_submitted' not in st.session_state:
    st.session_state.query_submitted = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .header {
        padding: 1rem;
        background-color: #f8f9fa;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e2f0fd;
        border-left: 5px solid #0c5460;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 5px solid #6c757d;
    }
    
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
    }
    
    .message-content {
        margin-bottom: 1rem;
    }
    
    .message-timestamp {
        font-size: 0.8rem;
        color: #6c757d;
    }
    
    /* Debug section styling */
    .debug-section {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #dee2e6;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 0.8rem;
        margin: 0.3rem;
        flex: 1;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Results table styling */
    .results-table {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Input box styling */
    .input-container {
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-top: 1px solid #e9ecef;
        position: sticky;
        bottom: 0;
    }
    
    /* Make the chat scrollable */
    .chat-container {
        overflow-y: auto;
        flex-grow: 1;
        padding: 1rem;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: white;
        border-top: 1px solid #e9ecef;
        z-index: 1000;
    }
    
    /* Add space at the bottom to prevent content from being hidden by the fixed footer */
    .footer-spacer {
        height: 150px;
    }
    
    /* Hide the default Streamlit elements like footer */
    footer {
        visibility: hidden;
    }
    
    /* Raw JSON styling */
    .raw-json {
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-all;
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    
    /* Structured JSON keys */
    .json-key {
        font-weight: bold;
        color: #0066cc;
    }
    
    /* Structured JSON values */
    .json-value {
        color: #333;
    }
    
    /* JSON block container */
    .json-block {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
        border-left: 3px solid #6c757d;
    }
    
    /* Prettier JSON formatting */
    pre.json {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
        color: #333;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .json-string { color: #008000; }
    .json-number { color: #0000ff; }
    .json-boolean { color: #b22222; }
    .json-null { color: #808080; }
    .json-property { color: #a52a2a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Function to check API health
def check_api_health(timeout=DEFAULT_TIMEOUT):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True, response.json()
        else:
            return False, response.json() if response.status_code == 200 else {"error": "API not responding correctly"}
    except Exception as e:
        return False, {"error": str(e)}

# Function to get available models
def get_available_models(timeout=DEFAULT_TIMEOUT):
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", {}), response.json().get("default_model", "")
        else:
            return {}, ""
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return {}, ""

# Function to format JSON in a structured way
def format_structured_json(data, is_nested=False):
    html = ""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                # For nested structures, create a collapsible section
                if not is_nested:
                    html += f'<div class="json-block"><span class="json-key">{key}:</span><br>'
                    html += format_structured_json(value, True)
                    html += '</div>'
                else:
                    html += f'<span class="json-key">{key}:</span><br>'
                    html += format_structured_json(value, True)
                    html += '<br>'
            else:
                # Format the value based on type
                if value is None:
                    value_str = "null"
                elif isinstance(value, bool):
                    value_str = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    value_str = str(value)
                else:
                    value_str = f'"{str(value)}"'
                html += f'<span class="json-key">{key}:</span> <span class="json-value">{value_str}</span><br>'
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                html += f'<span class="json-key">[{i}]</span><br>'
                html += format_structured_json(item, True)
                html += '<br>'
            else:
                if item is None:
                    item_str = "null"
                elif isinstance(item, bool):
                    item_str = "true" if item else "false"
                elif isinstance(item, (int, float)):
                    item_str = str(item)
                else:
                    item_str = f'"{str(item)}"'
                html += f'<span class="json-key">[{i}]</span> <span class="json-value">{item_str}</span><br>'
    return html

# Function to prettify JSON with good formatting
def prettify_json(obj, indent=2):
    """Format JSON object with proper indentation and syntax highlighting"""
    if obj is None:
        return ""
    
    # Convert to JSON string with proper indentation
    if isinstance(obj, str):
        try:
            # Try to parse string as JSON
            obj = json.loads(obj)
        except:
            # Not valid JSON, return as is
            return f'<pre class="json">{obj}</pre>'
    
    # Get nicely formatted JSON string
    json_str = json.dumps(obj, indent=indent, sort_keys=False)
    
    # Add HTML syntax highlighting
    json_str = json_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    json_str = json_str.replace('\n', '<br>')
    json_str = json_str.replace('  ', '&nbsp;&nbsp;')
    
    # Highlight different JSON elements
    json_str = json_str.replace('"([^"]*)":', '<span class="json-property">"\\1":</span>')
    json_str = json_str.replace(': "([^"]*)"', ': <span class="json-string">"\\1"</span>')
    json_str = json_str.replace('(\\d+)', '<span class="json-number">\\1</span>')
    json_str = json_str.replace('true|false', '<span class="json-boolean">\\0</span>')
    json_str = json_str.replace('null', '<span class="json-null">null</span>')
    
    return f'<pre class="json">{json_str}</pre>'

# Function to format JSON response content
# Function to format JSON response content
def format_json_response_content(response_data):
    """Format API response content to show:
    - Status on one line
    - SQL query with actual newlines
    - Error on a separate line
    - Each result bounded by {} on separate lines
    """
    if not response_data:
        return response_data
    
    content = response_data.get("content", "")
    if not content or not isinstance(content, str):
        return response_data
    
    try:
        # Try to parse the content string as JSON
        content_json = json.loads(content)
        
        # Format the content according to requirements
        output_lines = []
        
        # Add status on one line
        if "status" in content_json:
            output_lines.append(f"Status: {content_json.get('status', 'unknown')}")
        
        # Add SQL query with actual newlines
        sql_query = content_json.get('sql_query', '')
        if sql_query:
            # Replace escaped newlines with actual newlines
            sql_query = sql_query.replace('\\n', '\n')
            output_lines.append(f"SQL Query:\n{sql_query}")
        
        # Add error on separate line (if it exists)
        error = content_json.get('error')
        if error:
            output_lines.append(f"Error: {error}")
        
        # Format each result bounded by {} on separate lines
        results = content_json.get('results', [])
        if results:
            output_lines.append("Results:")
            for res in results:
                output_lines.append(f"{{{json.dumps(res)}}}")
        
        # Join all parts with double newlines
        formatted_content = '\n\n'.join(output_lines)
        
        # Create a copy of the response data
        formatted_response = response_data.copy()
        
        # Replace the content string with the formatted content
        formatted_response["content"] = formatted_content
        
        return formatted_response
    except json.JSONDecodeError:
        # Not valid JSON, leave as is
        return response_data
    
# Function to execute SQL query via API
def execute_query(query, model=None, max_attempts=3, timeout=DEFAULT_TIMEOUT):
    try:
        payload = {
            "query": query,
            "max_attempts": max_attempts
        }
        
        if model:
            payload["model"] = model
        
        # Log the request for debugging
        request_info = {
            "url": f"{API_URL}/execute_query",
            "payload": payload,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        response = requests.post(
            f"{API_URL}/execute_query",
            json=payload,
            timeout=60  # Longer timeout for query execution
        )
        
        # Store the raw response for debugging
        raw_response = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": response.text[:5000] if len(response.text) > 5000 else response.text  # Limit length for very large responses
        }
        
        if response.status_code == 200:
            result = response.json()
            # Pass through any debug_info from the API response
            if "debug_info" in result:
                result["debug_info"] = result["debug_info"]
            # Add raw request/response for debugging
            result["_debug"] = {
                "request": request_info,
                "response": raw_response
            }
            return result
        else:
            st.error(f"API Error: {response.status_code}")
            error_result = {
                "status": "error", 
                "error": f"API Error: {response.status_code}",
                "_debug": {
                    "request": request_info,
                    "response": raw_response
                }
            }
            try:
                error_result["error_details"] = response.json()
            except:
                error_result["error_details"] = "Failed to parse API response"
            return error_result
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "_debug": {
                "request": {
                    "url": f"{API_URL}/execute_query",
                    "payload": payload if 'payload' in locals() else {},
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "exception": str(e)
            }
        }

# Function to estimate token counts
def estimate_tokens(text):
    # Approximate token count based on words (rough estimate)
    if not text:
        return 0
    words = str(text).split()
    return len(words) * 1.3  # Average of 1.3 tokens per word

# Save query to state
def save_query():
    if st.session_state.query_input and st.session_state.query_input.strip():
        st.session_state.current_query = st.session_state.query_input.strip()
        st.session_state.query_submitted = True
        st.session_state.query_input = ""  # Clear the input

# Sidebar with controls
with st.sidebar:
    st.header("Settings")
    
    # Check API connection status
    api_status, api_info = check_api_health()
    if api_status:
        st.success("✅ API Connected")
        
        # Database status
        if api_info.get("database_connected", False):
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Not Connected")
    else:
        st.error(f"❌ API Not Connected: {api_info.get('error', 'Unknown error')}")
    
    # Get available models and create selector
    models_dict, default_model = get_available_models()
    model_options = list(models_dict.keys())
    
    # Only show model selector if we have models
    if model_options:
        selected_model = st.selectbox(
            "Select Model:",
            options=model_options,
            index=0 if not default_model else model_options.index(default_model) if default_model in model_options else 0,
            key="model_selector"
        )
    else:
        selected_model = None
        st.warning("No models available. Check API connection.")
    
    # Max attempts slider
    max_attempts = st.slider("Max Attempts:", min_value=1, max_value=5, value=3, key="max_attempts_slider")
    
    # Clear history button
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.query_submitted = False
        st.session_state.current_query = ""
        st.success("Conversation cleared!")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This assistant uses LLMs to generate SQL queries from natural language requests. "
                "The system searches through database schema information to provide accurate results.")

# Header 
st.markdown("<h1 style='text-align: center;'>SQL Query Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Ask natural language questions about your data</p>", unsafe_allow_html=True)

# Process query if submitted
if st.session_state.query_submitted:
    query = st.session_state.current_query
    start_time = time.time()
    
    # Add user message to conversation
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.conversation.append({
        "role": "user",
        "content": query,
        "timestamp": timestamp
    })
    
    # Call API
    with st.spinner("Generating SQL query and fetching data..."):
        result = execute_query(
            query=query, 
            model=selected_model,
            max_attempts=max_attempts
        )
    
    execution_time = time.time() - start_time
    
    # Format response message
    if result["status"] == "success":
        response_message = "I've generated an SQL query and executed it successfully. See the results below."
    else:
        response_message = f"I encountered an error: {result.get('error', 'Unknown error')}"
    
    # Add assistant response to conversation
    result["role"] = "assistant"
    result["message"] = response_message
    result["query"] = query
    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["execution_time"] = execution_time
    
    st.session_state.conversation.append(result)
    
    # Reset submission flag
    st.session_state.query_submitted = False
    st.session_state.current_query = ""

# Display conversation history
chat_container = st.container()

with chat_container:
    for i, entry in enumerate(st.session_state.conversation, 1):  # Start index at 1
        if entry["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">
                    <span>You</span>
                    <span class="message-timestamp">{entry["timestamp"]}</span>
                </div>
                <div class="message-content">
                    {entry["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:  # assistant role
            status_emoji = "✅" if entry.get("status") == "success" else "❌"
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">
                    <span>Assistant {status_emoji}</span>
                    <span class="message-timestamp">{entry["timestamp"]}</span>
                </div>
                <div class="message-content">
                    {entry.get("message", "")}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # If successful, display the results
            if entry.get("status") == "success" and entry.get("results"):
                # Convert results to DataFrame
                df = pd.DataFrame(entry["results"])
                # Reset index to start from 1 instead of 0
                df.index = range(1, len(df) + 1)
                
                # Display results table
                st.markdown("<div class='results-table'>", unsafe_allow_html=True)
                st.subheader("Results")
                st.dataframe(df, use_container_width=True)
                
                # Allow downloading as CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download as CSV",
                    csv,
                    "query_results.csv",
                    "text/csv",
                    key=f'download-csv-{i}'
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Collapsible debug information
            with st.expander("Debug Information", expanded=False):
                st.markdown("<div class='debug-section'>", unsafe_allow_html=True)
                
                # SQL Query
                st.subheader("Generated SQL Query")
                st.code(entry.get("sql_query", "No query generated"), language="sql")
                
                # Metrics in cards
                st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                
                # Calculate token estimates
                input_tokens = int(estimate_tokens(entry.get("query", "")))
                output_tokens = int(estimate_tokens(str(entry.get("results", "")) + entry.get("sql_query", "")))
                
                metrics = [
                    {"name": "Status", "value": entry.get("status", "Unknown")},
                    {"name": "Processing Time", "value": f"{entry.get('processing_time', 0):.2f}s"},
                    {"name": "Total Time", "value": f"{entry.get('execution_time', 0):.2f}s"},
                    {"name": "Attempts", "value": entry.get("attempts", 0)},
                    {"name": "Model", "value": entry.get("model", "Unknown")},
                    {"name": "Input Tokens", "value": input_tokens},
                    {"name": "Output Tokens", "value": output_tokens},
                    {"name": "Row Count", "value": entry.get("row_count", 0)},
                ]
                
                for metric in metrics:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>{metric['name']}</h4>
                        <div>{metric['value']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Error details if any
                if entry.get("status") != "success":
                    st.subheader("Error Details")
                    st.error(entry.get("error", "Unknown error"))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                st.subheader("System Prompt and RAG Content")   
                if "debug_info" in entry:  # Check directly in entry, 
                    
                    
                    # System Prompt tab
                    debug_info = entry.get("debug_info", {})  # Get directly from entry
                    prompt_tabs = st.tabs(["System Prompt", "Schema Context", "Metadata Context", 
                                        "Query Examples", "Complex Query Examples"])
                    
                    with prompt_tabs[0]:
                        st.code(debug_info.get("system_prompt", "No system prompt available"), language="text")
                    
                    # RAG Context tabs
                    rag_context = debug_info.get("rag_context", {})
                    
                    with prompt_tabs[1]:
                        st.code(rag_context.get("schema_context", "No schema context available"), language="text")
                    
                    with prompt_tabs[2]:
                        st.code(rag_context.get("metadata_context", "No metadata context available"), language="text")
                    
                    with prompt_tabs[3]:
                        st.code(rag_context.get("query_examples_context", "No query examples available"), language="text")
                    
                    with prompt_tabs[4]:
                        st.code(rag_context.get("complex_query_examples_context", "No complex query examples available"), language="text")
    
                # Raw LangServe API output - Better structured
                st.subheader("Raw LangServe API Data")
                
                # Get the debug info if available
                debug_inform = entry.get("_debug", {})
                
                if debug_inform:
                    # Display request information
                    st.subheader("API Request")
                    
                    # Format the request payload in a structured way
                    request_data = debug_inform.get("request", {})
                    if request_data:
                        st.markdown("<div class='raw-json'>", unsafe_allow_html=True)
                        st.markdown(format_structured_json(request_data), unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No request data available")
                    
                    # Display response information with improved formatting
                    st.subheader("API Response")
                    
                    # Get response data and format it properly
                    response_data = debug_inform.get("response", {})
                    if response_data:
                        # Process and format the content JSON
                        formatted_response_data = format_json_response_content(response_data.copy())
                        
                        # Display the formatted content
                        content = formatted_response_data.get("content", "")
                        
                        # Show status and headers normally
                        st.markdown("#### Status Code")
                        st.code(str(formatted_response_data.get("status_code", "")))
                        
                        st.markdown("#### Headers")
                        st.json(formatted_response_data.get("headers", {}))
                        
                    else:
                        st.info("No response data available")

                else:
                    # Format the entry data
                    entry_data = {k: v for k, v in entry.items() if k != "_debug"}
                    st.json(entry_data)
                
                st.markdown("</div>", unsafe_allow_html=True)

# Add space at the bottom so content isn't hidden by the fixed footer
st.markdown("<div class='footer-spacer'></div>", unsafe_allow_html=True)

# Fixed footer with input box
st.markdown('<div class="footer">', unsafe_allow_html=True)

# Create columns to arrange the input box and button
col1, col2 = st.columns([6, 1])

with col1:
    # Input text box
    st.text_area(
        "Ask a question about your data:",
        key="query_input",
        placeholder="Example: Show me the total premium amount by segment for January 2025",
        height=85
    )

with col2:
    # Place the button at the bottom to align with the text area
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space
    if st.button("Send", key="send_button", on_click=save_query):
        pass  # The on_click handler does the work

st.markdown('</div>', unsafe_allow_html=True)
