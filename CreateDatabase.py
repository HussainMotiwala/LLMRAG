import pandas as pd
import pyodbc
import sqlalchemy
import re
import os
import urllib
import numpy as np
from sqlalchemy import create_engine, text, Integer, String, Float, DateTime, Boolean
from sqlalchemy.types import TypeEngine

def clean_column_name(col_name):
    """Clean column names to be SQL-friendly"""
    # Convert to string in case it's a numeric column name
    col_name = str(col_name)
    
    # Special handling for the 'Policy' field to ensure it stays exactly as 'Policy'
    if col_name.strip() == 'Policy':
        return 'Policy'
    
    # First trim any leading/trailing spaces
    col_name = col_name.strip()
    
    # Replace spaces with underscores
    col_name = col_name.replace(' ', '_')
    
    # Now replace other special characters with underscores
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', col_name)
    
    # Remove leading numbers or underscores
    clean_name = re.sub(r'^[\d_]+', '', clean_name)
    
    # Ensure it's not empty after cleaning
    if not clean_name or clean_name.isdigit():
        clean_name = f"column_{col_name}"
    
    # Truncate if too long for SQL Server (max 128 chars)
    if len(clean_name) > 128:
        clean_name = clean_name[:128]
    
    return clean_name

def infer_sql_type(series):
    """Infer SQL type from pandas Series"""
    dtype = series.dtype
    # Check if series is all null
    if series.isna().all():
        return String(100)  # Default to string for empty columns
    
    # Check for numeric types
    if pd.api.types.is_integer_dtype(dtype):
        return Integer()
    elif pd.api.types.is_float_dtype(dtype):
        return Float()
    # Check for datetime
    elif pd.api.types.is_datetime64_dtype(dtype):
        return DateTime()
    # Check for boolean
    elif pd.api.types.is_bool_dtype(dtype):
        return Boolean()
    # Default to string
    else:
        # Get the max length of strings in the column
        if pd.api.types.is_string_dtype(dtype):
            max_len = series.astype(str).str.len().max()
            if pd.isna(max_len):
                max_len = 100  # Default length
            else:
                max_len = min(max(max_len, 10) * 2, 4000)  # Double length for safety, capped at 4000
            return String(max_len)
        return String(255)  # Default for other types

def excel_to_mssql(excel_path, server, database, username=None, password=None, trusted_connection=False):
    """
    Import all sheets from an Excel file to SQL Server tables.
    
    Parameters:
    - excel_path: Path to the Excel file
    - server: SQL Server name
    - database: Database name
    - username: SQL Server username (if using SQL Server authentication)
    - password: SQL Server password (if using SQL Server authentication)
    - trusted_connection: Use Windows authentication if True
    """
    try:
        # Remove any quotes that might be in the path
        excel_path = excel_path.strip('"\'')
        print(f"Reading Excel file: {excel_path}")
        
        # Read all sheets from Excel file
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        
        if not sheet_names:
            print("No sheets found in the Excel file.")
            return
        
        # Create connection string
        if trusted_connection:
            # Add TrustServerCertificate=yes and fix NVARCHAR issue
            conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;TrustServerCertificate=yes;'
            # Direct ODBC connection for checking tables
            odbc_conn = pyodbc.connect(conn_str)
            conn_url = f'mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}'
        else:
            # Add TrustServerCertificate=yes and fix NVARCHAR issue
            conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;'
            # Direct ODBC connection for checking tables
            odbc_conn = pyodbc.connect(conn_str)
            conn_url = f'mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}'
        
        # Create SQLAlchemy engine with fast_executemany=True for better performance
        engine = create_engine(conn_url, fast_executemany=True)
        
        # Process each sheet
        for sheet_name in sheet_names:
            # Clean sheet name for table name
            table_name = clean_column_name(sheet_name)
            print(f"Processing sheet: {sheet_name} -> Table: {table_name}")
            
            # Read the sheet
            df = pd.read_excel(xls, sheet_name, dtype={'Policy': str})
            
            if df.empty:
                print(f"Sheet '{sheet_name}' is empty. Skipping.")
                continue
            
            # Clean column names
            df.columns = [clean_column_name(col) for col in df.columns]
            
            # Infer data types for columns
            dtypes = {col: infer_sql_type(df[col]) for col in df.columns}
            
            # Check if table exists and drop if requested
            with odbc_conn.cursor() as cursor:
                # Check if table exists using direct SQL query instead of SQLAlchemy reflection
                cursor.execute(f"SELECT OBJECT_ID('{table_name}', 'U')")
                table_exists = cursor.fetchone()[0] is not None
                
                if table_exists:
                    drop_table = input(f"Table '{table_name}' already exists. Drop it? (y/n): ").lower() == 'y'
                    if drop_table:
                        cursor.execute(f"DROP TABLE {table_name}")
                        odbc_conn.commit()
                        print(f"Dropped existing table '{table_name}'")
                    else:
                        print(f"Skipping table '{table_name}'")
                        continue
            
            # Create table and insert data using a more direct approach
            print(f"Creating table '{table_name}' and inserting {len(df)} rows...")
            
            # Create a more robust table creation method with specific SQL types
            create_table_sql = f"CREATE TABLE {table_name} (\n"
            columns = []
            
            # Track if the table has a Policy column for primary key
            has_policy_column = False
            for col in df.columns:
                if clean_column_name(col) == 'Policy':
                    has_policy_column = True
                    break
                    
            for col in df.columns:
                # Get the cleaned column name
                clean_col = clean_column_name(col)
                
                # Special handling for Policy column
                if clean_col == 'Policy':
                    # Always set Policy as VARCHAR(8), but not as primary key
                    sql_type = "VARCHAR(8)"
                else:
                    sql_type = "VARCHAR(255)"  # Default type
                
                    # Infer SQL type from pandas Series
                    dtype = df[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_type = "INT"
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_type = "FLOAT"
                    elif pd.api.types.is_datetime64_dtype(dtype):
                        sql_type = "DATETIME2"
                    elif pd.api.types.is_bool_dtype(dtype):
                        sql_type = "BIT"
                    elif pd.api.types.is_string_dtype(dtype):
                        # Get max length
                        max_len = df[col].astype(str).str.len().max()
                        if pd.isna(max_len):
                            max_len = 100
                        else:
                            max_len = min(max(int(max_len * 1.5), 10), 4000)  # Ensure enough space
                        sql_type = f"VARCHAR({max_len})"
                
                columns.append(f"[{col}] {sql_type}")
            
            create_table_sql += ",\n".join(columns)
            create_table_sql += "\n)"
            
            # Create the table
            with odbc_conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                odbc_conn.commit()
            
            # Insert data using fast_executemany
            # Handle data formatting for Policy fields to preserve leading zeros
            if 'Policy' in df.columns:
                # Check if pandas converted Policy to numeric type (which would lose leading zeros)
                if pd.api.types.is_numeric_dtype(df['Policy'].dtype):
                    # Convert to string and pad with leading zeros to ensure 8 characters
                    df['Policy'] = df['Policy'].astype(str).str.zfill(8)
                else:
                    # If it's already string type, still ensure proper padding
                    df['Policy'] = df['Policy'].astype(str).str.zfill(8)
                
                # Force Policy column to be exactly 8 characters
                df['Policy'] = df['Policy'].str.slice(0, 8)
            
            # Convert DataFrame to list of tuples for insertion
            # Use the DataFrame after string conversion to preserve leading zeros
            data_tuples = [tuple(x) for x in df.replace({np.nan: None}).to_numpy()]
            
            # Generate the INSERT statement
            placeholders = ','.join(['?' for _ in df.columns])
            columns_str = ','.join([f'[{col}]' for col in df.columns])
            insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # Insert the data in batches to avoid memory issues
            batch_size = 1000
            with odbc_conn.cursor() as cursor:
                for i in range(0, len(data_tuples), batch_size):
                    batch = data_tuples[i:i+batch_size]
                    cursor.fast_executemany = True
                    cursor.executemany(insert_sql, batch)
                    odbc_conn.commit()
                    print(f"Inserted {min(i+batch_size, len(data_tuples))} of {len(data_tuples)} rows...")
            
            print(f"Successfully created table '{table_name}' with {len(df)} rows")
            
        # Close the connection
        odbc_conn.close()
        
        print("All sheets have been imported successfully.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    excel_path = input("Enter the path to the Excel file: ")
    # Remove any quotes that might be in the path
    excel_path = excel_path.strip('"\'')
    
    server = input("Enter the SQL Server name: ")
    database = input("Enter the database name: ")
    
    auth_type = input("Use Windows Authentication? (y/n): ").lower()
    
    if auth_type == 'y':
        excel_to_mssql(excel_path, server, database, trusted_connection=True)
    else:
        username = input("Enter SQL Server username: ")
        password = input("Enter SQL Server password: ")
        excel_to_mssql(excel_path, server, database, username, password)