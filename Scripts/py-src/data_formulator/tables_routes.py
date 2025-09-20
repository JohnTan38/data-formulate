# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. from GPT-5

import logging
import sys
import os
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')
import json
import traceback
from flask import request, send_from_directory, session, jsonify, Blueprint, current_app
import pandas as pd
import random
import string
from pathlib import Path
import uuid
import tempfile

from data_formulator.db_manager import db_manager
from data_formulator.data_loader import DATA_LOADERS

import re
from typing import Tuple

# Configure root logger for general application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Get logger for this module
logger = logging.getLogger(__name__)

tables_bp = Blueprint('tables', __name__, url_prefix='/api/tables')

# --- helper: resolve session id safely ------------------------------------
def resolve_session_id() -> str | None:
    """
    Resolve session id from cookie, header, query param or Flask session.
    Return None if not found.
    """
    sid = (
        request.cookies.get("session_id")
        or request.headers.get("X-Session-Id")
        or request.args.get("session_id")
        or session.get("session_id")
    )
    if not sid:
        logger.info("No session_id provided in request. cookies=%s headers_sample=%s",
                    dict(request.cookies), {k: request.headers.get(k) for k in ("X-Session-Id","Authorization")})
    return sid

def session_missing_response():
    return jsonify({"status":"error","message":"session_id not found, please refresh the page"}), 401
# -------------------------------------------------------------------------


@tables_bp.route('/list-tables', methods=['GET'])
def list_tables():
    """List all tables in the current session"""
    try:
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        result = []
        with db_manager.connection(sid) as db:
            table_metadata_list = db.execute("""
                SELECT database_name, schema_name, table_name, schema_name==current_schema() as is_current_schema, 'table' as object_type 
                FROM duckdb_tables() 
                WHERE internal=False AND database_name == current_database()
                UNION ALL 
                SELECT database_name, schema_name, view_name as table_name, schema_name==current_schema() as is_current_schema, 'view' as object_type 
                FROM duckdb_views()
                WHERE view_name NOT LIKE 'duckdb_%' AND view_name NOT LIKE 'sqlite_%' AND view_name NOT LIKE 'pragma_%' AND database_name == current_database()
            """).fetchall()
        
            for table_metadata in table_metadata_list:
                [database_name, schema_name, table_name, is_current_schema, object_type] = table_metadata
                table_name = table_name if is_current_schema else '.'.join([database_name, schema_name, table_name])
                if database_name in ['system', 'temp']:
                    continue
                
                logger.debug(f"table_metadata: {table_metadata}")

                try:
                    # Get column information
                    columns = db.execute(f"DESCRIBE {table_name}").fetchall()
                    # Get row count
                    row_count = db.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    sample_rows = db.execute(f"SELECT * FROM {table_name} LIMIT 1000").fetchdf() if row_count > 0 else pd.DataFrame()
                    
                    # Check if this is a view or a table
                    try:
                        view_info = db.execute(f"SELECT view_name, sql FROM duckdb_views() WHERE view_name = '{table_name}'").fetchone()
                        view_source = view_info[1] if view_info else None
                    except Exception:
                        view_source = None
                    
                    result.append({
                        "name": table_name,
                        "columns": [{"name": col[0], "type": col[1]} for col in columns],
                        "row_count": row_count,
                        "sample_rows": json.loads(sample_rows.to_json(orient='records', date_format='iso')) if not sample_rows.empty else [],
                        "view_source": view_source
                    })
                    
                except Exception as e:
                    logger.error(f"Error getting table metadata for {table_name}: {str(e)}")
                    continue
        
        return jsonify({
            "status": "success",
            "tables": result
        })
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code
        

def assemble_query(aggregate_fields_and_functions, group_fields, columns, table_name):
    """
    Assembles a SELECT query string based on binning, aggregation, and grouping specifications.
    """
    select_parts = []
    output_column_names = []

    # Handle aggregate fields and functions
    for field, function in aggregate_fields_and_functions:
        if field is None:
            if function.lower() == 'count':
                select_parts.append('COUNT(*) as _count')
                output_column_names.append('_count')
        elif field in columns:
            if function.lower() == 'count':
                alias = f'_count'
                select_parts.append(f'COUNT(*) as "{alias}"')
                output_column_names.append(alias)
            else:
                if function in ["avg", "average", "mean"]:
                    aggregate_function = "AVG"
                else:
                    aggregate_function = function.upper()
                alias = f'{field}_{function}'
                select_parts.append(f'{aggregate_function}("{field}") as "{alias}"')
                output_column_names.append(alias)

    # Handle group fields
    for field in group_fields:
        if field in columns:
            select_parts.append(f'"{field}"')
            output_column_names.append(field)
    # If no fields are specified, select all columns
    if not select_parts:
        select_parts = ["*"]
        output_column_names = columns

    from_clause = f"FROM {table_name}"
    group_by_clause = f"GROUP BY {', '.join(group_fields)}" if len(group_fields) > 0 and len(aggregate_fields_and_functions) > 0 else ""

    query = f"SELECT {', '.join(select_parts)} {from_clause} {group_by_clause}"
    return query, output_column_names

@tables_bp.route('/sample-table', methods=['POST'])
def sample_table():
    """Sample a table"""
    try:
        data = request.get_json()
        table_id = data.get('table')
        sample_size = data.get('size', 1000)
        aggregate_fields_and_functions = data.get('aggregate_fields_and_functions', []) 
        select_fields = data.get('select_fields', []) 
        method = data.get('method', 'random') 
        order_by_fields = data.get('order_by_fields', [])

        logger.debug(f"sample_table: {table_id}, {sample_size}, {aggregate_fields_and_functions}, {select_fields}, {method}, {order_by_fields}")
        
        total_row_count = 0

        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        # Validate field names against table columns to prevent SQL injection
        with db_manager.connection(sid) as db:
            columns = [col[0] for col in db.execute(f"DESCRIBE {table_id}").fetchall()]
            logger.debug(f"columns: {columns}")
            
            valid_order_by_fields = [field for field in order_by_fields if field in columns]
            valid_aggregate_fields_and_functions = [
                field_and_function for field_and_function in aggregate_fields_and_functions 
                if field_and_function[0] is None or field_and_function[0] in columns
            ]
            valid_select_fields = [field for field in select_fields if field in columns]

            query, output_column_names = assemble_query(valid_aggregate_fields_and_functions, valid_select_fields, columns, table_id)

            logger.debug(f"query: {query}")
            logger.debug(f"output_column_names: {output_column_names}")

            count_query = f"SELECT *, COUNT(*) OVER () as total_count FROM ({query}) as subq LIMIT 1"
            result = db.execute(count_query).fetchone()
            total_row_count = result[-1] if result else 0

            logger.debug(f"total_row_count: {total_row_count}")

            if method == 'random':
                query += f" ORDER BY RANDOM() LIMIT {sample_size}"
            elif method == 'head':
                if valid_order_by_fields:
                    order_by_clause = ", ".join([f'"{field}"' for field in valid_order_by_fields])
                    query += f" ORDER BY {order_by_clause} LIMIT {sample_size}"
                else:
                    query += f" LIMIT {sample_size}"
            elif method == 'bottom':
                if valid_order_by_fields:
                    order_by_clause = ", ".join([f'"{field}" DESC' for field in valid_order_by_fields])
                    query += f" ORDER BY {order_by_clause} LIMIT {sample_size}"
                else:
                    query += f" ORDER BY ROWID DESC LIMIT {sample_size}"

            logger.debug(f"final query: {query}")

            result_df = db.execute(query).fetchdf()

        return jsonify({
            "status": "success",
            "rows": json.loads(result_df.to_json(orient='records', date_format='iso')),
            "total_row_count": total_row_count
        })
    except Exception as e:
        logger.error(f"Error sampling table: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code

@tables_bp.route('/get-table', methods=['GET'])
def get_table_data():
    """Get data from a specific table"""
    try:
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        with db_manager.connection(sid) as db:
            table_name = request.args.get('table_name')
            page = int(request.args.get('page', 1))
            page_size = int(request.args.get('page_size', 100))
            offset = (page - 1) * page_size
            
            if not table_name:
                return jsonify({
                    "status": "error",
                    "message": "Table name is required"
                }), 400
            
            total_rows = db.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            result = db.execute(
                f"SELECT * FROM {table_name} LIMIT {page_size} OFFSET {offset}"
            ).fetchall()
            columns = [col[0] for col in db.execute(f"DESCRIBE {table_name}").fetchall()]
            rows = [dict(zip(columns, row)) for row in result]
        
            return jsonify({
                "status": "success",
                "table_name": table_name,
                "columns": columns,
                "rows": rows,
                "total_rows": total_rows,
                "page": page,
                "page_size": page_size
            })
    
    except Exception as e:
        logger.error(f"Error getting table data: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code

@tables_bp.route('/create-table', methods=['POST'])
def create_table():
    """Create a new table from uploaded data"""
    try:
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        if 'file' not in request.files and 'raw_data' not in request.form:
            return jsonify({"status": "error", "message": "No file or raw data provided"}), 400
        
        table_name = request.form.get('table_name')
        if not table_name:
            return jsonify({"status": "error", "message": "No table name provided"}), 400
        
        df = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif file.filename.endswith('.json'):
                df = pd.read_json(file)
            else:
                return jsonify({"status": "error", "message": "Unsupported file format"}), 400
        else:
            raw_data = request.form.get('raw_data')
            try:
                df = pd.DataFrame(json.loads(raw_data))
            except Exception as e:
                return jsonify({"status": "error", "message": f"Invalid JSON data: {str(e)}, it must be in the format of a list of dictionaries"}), 400

        if df is None:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        sanitized_table_name = sanitize_table_name(table_name)
            
        with db_manager.connection(sid) as db:
            base_name = sanitized_table_name
            counter = 1
            while True:
                exists = db.execute(f"SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = '{sanitized_table_name}'").fetchone()[0] > 0
                if not exists:
                    break
                sanitized_table_name = f"{base_name}_{counter}"
                counter += 1

            db.register('df_temp', df)
            db.execute(f"CREATE TABLE {sanitized_table_name} AS SELECT * FROM df_temp")
            # some duckdb versions create a view for register, ensure cleanup
            try:
                db.execute("DROP VIEW IF EXISTS df_temp")
            except Exception:
                pass
            
            return jsonify({
                "status": "success",
                "table_name": sanitized_table_name,
                "row_count": len(df),
                "columns": list(df.columns),
                "original_name": base_name,
                "is_renamed": base_name != sanitized_table_name
            })
    
    except Exception as e:
        logger.error(f"Error creating table: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code



@tables_bp.route('/delete-table', methods=['POST'])
def drop_table():
    """Drop a table or view"""
    try:
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        data = request.get_json()
        table_name = data.get('table_name')
        
        if not table_name:
            return jsonify({"status": "error", "message": "No table name provided"}), 400
            
        with db_manager.connection(sid) as db:
            view_exists = db.execute(f"SELECT view_name FROM duckdb_views() WHERE view_name = '{table_name}'").fetchone() is not None
            if view_exists:
                db.execute(f"DROP VIEW IF EXISTS {table_name}")
            table_exists = db.execute(f"SELECT table_name FROM duckdb_tables() WHERE table_name = '{table_name}'").fetchone() is not None
            if table_exists:
                db.execute(f"DROP TABLE IF EXISTS {table_name}")

            if not view_exists and not table_exists:
                return jsonify({
                    "status": "error",
                    "message": f"Table/view '{table_name}' does not exist"
                }), 404
        
            return jsonify({
                "status": "success",
                "message": f"Table/view {table_name} dropped"
            })
    
    except Exception as e:
        logger.error(f"Error dropping table: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code


@tables_bp.route('/upload-db-file', methods=['POST'])
def upload_db_file():
    """Upload a db file"""
    try:
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.db'):
            return jsonify({"status": "error", "message": "Invalid file format. Only .db files are supported"}), 400
        
        temp_dir = os.path.join(tempfile.gettempdir())
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_db_path = os.path.join(temp_dir, f"temp_{sid}.db")
        file.save(temp_db_path)
        
        try:
            import duckdb
            conn = duckdb.connect(temp_db_path, read_only=True)
            conn.execute("SELECT 1").fetchall()
            conn.close()
            
            db_file_path = os.path.join(temp_dir, f"df_{sid}.db")
            os.replace(temp_db_path, db_file_path)
            
            db_manager._db_files[sid] = db_file_path
            
        except Exception as db_error:
            logger.error(f"Error uploading db file: {str(db_error)}")
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
            return jsonify({
                "status": "error",
                "message": f"Invalid DuckDB database file."
            }), 400
        
        return jsonify({
            "status": "success",
            "message": "Database file uploaded successfully",
            "session_id": sid
        })
        
    except Exception as e:
        logger.error(f"Error uploading db file: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error", 
            "message": safe_msg
        }), status_code


@tables_bp.route('/download-db-file', methods=['GET'])
def download_db_file():
    """Download the db file for a session"""
    try:
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        if sid not in db_manager._db_files:
            return jsonify({
                "status": "error",
                "message": "No database file found for this session"
            }), 404
            
        db_file_path = db_manager._db_files[sid]
        
        if not os.path.exists(db_file_path):
            return jsonify({
                "status": "error",
                "message": "Database file not found"
            }), 404
            
        download_name = f"data_formulator_{sid}.db"
        
        return send_from_directory(
            os.path.dirname(db_file_path),
            os.path.basename(db_file_path),
            as_attachment=True,
            download_name=download_name,
            mimetype='application/x-sqlite3'
        )
        
    except Exception as e:
        logger.error(f"Error downloading db file: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code


@tables_bp.route('/reset-db-file', methods=['POST'])
def reset_db_file():
    """Reset the db file for a session"""
    try:
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        logger.info(f"session_id: {sid}")
        
        if sid in db_manager._db_files:
            db_file_path = db_manager._db_files[sid]
            if db_file_path and os.path.exists(db_file_path):
                os.remove(db_file_path)
            db_manager._db_files[sid] = None
            
        temp_db_path = os.path.join(tempfile.gettempdir(), f"temp_{sid}.db")
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
            
        main_db_path = os.path.join(tempfile.gettempdir(), f"df_{sid}.db")
        if os.path.exists(main_db_path):
            os.remove(main_db_path)

        return jsonify({
            "status": "success",
            "message": "Database file reset successfully"
        })

    except Exception as e:
        logger.error(f"Error resetting db file: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code


@tables_bp.route('/query', methods=['POST'])
def query_table():
    """Execute a query on a table"""
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"status": "error", "message": "No query provided"}), 400

        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        with db_manager.connection(sid) as db:
            result = db.execute(query).fetch_df()
        
            return jsonify({
                "status": "success",
                "rows": json.loads(result.to_json(orient='records', date_format='iso')),
                "columns": list(result.columns)
            })
    
    except Exception as e:
        logger.error(f"Error querying table: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code


# Example of a more complex query endpoint
@tables_bp.route('/analyze', methods=['POST'])
def analyze_table():
    """Get basic statistics about a table"""
    try:
        data = request.get_json()
        table_name = data.get('table_name')
        
        if not table_name:
            return jsonify({"status": "error", "message": "No table name provided"}), 400

        sid = resolve_session_id()
        if not sid:
            return session_missing_response()
        
        with db_manager.connection(sid) as db:
        
            columns = db.execute(f"DESCRIBE {table_name}").fetchall()
            
            stats = []
            for col in columns:
                col_name = col[0]
                col_type = col[1]
                
                quoted_col_name = f'"{col_name}"'
                
                stats_query = f"""
                SELECT 
                    COUNT(*) as count,
                    COUNT(DISTINCT {quoted_col_name}) as unique_count,
                    COUNT(*) - COUNT({quoted_col_name}) as null_count
                FROM {table_name}
                """
                
                if col_type in ['INTEGER', 'DOUBLE', 'DECIMAL']:
                    stats_query = f"""
                    SELECT 
                        COUNT(*) as count,
                        COUNT(DISTINCT {quoted_col_name}) as unique_count,
                        COUNT(*) - COUNT({quoted_col_name}) as null_count,
                        MIN({quoted_col_name}) as min_value,
                        MAX({quoted_col_name}) as max_value,
                        AVG({quoted_col_name}) as avg_value
                    FROM {table_name}
                    """
                
                col_stats = db.execute(stats_query).fetchone()
                
                if col_type in ['INTEGER', 'DOUBLE', 'DECIMAL']:
                    stats_dict = dict(zip(
                        ["count", "unique_count", "null_count", "min", "max", "avg"],
                        col_stats
                    ))
                else:
                    stats_dict = dict(zip(
                        ["count", "unique_count", "null_count"],
                        col_stats
                    ))
                
                stats.append({
                    "column": col_name,
                    "type": col_type,
                    "statistics": stats_dict
                })
        
        return jsonify({
            "status": "success",
            "table_name": table_name,
            "statistics": stats
        })
    
    except Exception as e:
        logger.error(f"Error analyzing table: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error",
            "message": safe_msg
        }), status_code


def sanitize_table_name(table_name: str) -> str:
    """
    Sanitize a table name to be a valid DuckDB table name.
    """
    sanitized_table_name = table_name.lower()
    sanitized_table_name = sanitized_table_name.replace('-', '_')
    sanitized_table_name = sanitized_table_name.replace(' ', '_')
    sanitized_table_name = ''.join(c for c in sanitized_table_name if c.isalnum() or c == '_')
    
    if not sanitized_table_name or not sanitized_table_name[0].isalpha():
        sanitized_table_name = 'table_' + sanitized_table_name
        
    if not sanitized_table_name:
        return f'table_{uuid.uuid4()}'
    return sanitized_table_name

def sanitize_db_error_message(error: Exception) -> Tuple[str, int]:
    """
    Sanitize error messages before sending to client.
    Returns a tuple of (sanitized_message, status_code)
    """
    error_msg = str(error)
    
    safe_error_patterns = {
        r"Table.*does not exist": (error_msg, 404),
        r"Table.*already exists": (error_msg, 409),
        r"syntax error": (error_msg, 400),
        r"Catalog Error": (error_msg, 404), 
        r"Binder Error": (error_msg, 400),
        r"Invalid input syntax": (error_msg, 400),
        r"No such file": (error_msg, 404),
        r"Permission denied": ("Access denied", 403),
        r"Entity ID": (error_msg, 500),
        r"session_id": ("session_id not found, please refresh the page", 500),
    }
    
    for pattern, (safe_msg, status_code) in safe_error_patterns.items():
        if re.search(pattern, error_msg, re.IGNORECASE):
            return safe_msg, status_code
            
    logger.error(f"Unexpected error occurred: {error_msg}")
    return f"An unexpected error occurred: {error_msg}", 500


@tables_bp.route('/data-loader/list-data-loaders', methods=['GET'])
def data_loader_list_data_loaders():
    """List all available data loaders"""
    try:
        return jsonify({
            "status": "success",
            "data_loaders": {
                name: {
                    "params": data_loader.list_params(),
                    "auth_instructions": data_loader.auth_instructions()
                }
                for name, data_loader in DATA_LOADERS.items()
            }
        })
    except Exception as e:
        logger.error(f"Error listing data loaders: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error", 
            "message": safe_msg
        }), status_code

@tables_bp.route('/data-loader/list-tables', methods=['POST'])
def data_loader_list_tables():
    """List tables from a data loader"""
    try:
        data = request.get_json()
        data_loader_type = data.get('data_loader_type')
        data_loader_params = data.get('data_loader_params')
        table_filter = data.get('table_filter', None)

        if data_loader_type not in DATA_LOADERS:
            return jsonify({"status": "error", "message": f"Invalid data loader type. Must be one of: {', '.join(DATA_LOADERS.keys())}"}), 400

        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        with db_manager.connection(sid) as duck_db_conn:
            data_loader = DATA_LOADERS[data_loader_type](data_loader_params, duck_db_conn)
            
            if hasattr(data_loader, 'list_tables') and 'table_filter' in data_loader.list_tables.__code__.co_varnames:
                tables = data_loader.list_tables(table_filter=table_filter)
            else:
                tables = data_loader.list_tables()

            return jsonify({
                "status": "success",
                "tables": tables
            })

    except Exception as e:
        logger.error(f"Error listing tables from data loader: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error", 
            "message": safe_msg
        }), status_code


@tables_bp.route('/data-loader/ingest-data', methods=['POST'])
def data_loader_ingest_data():
    """Ingest data from a data loader"""
    try:
        data = request.get_json()
        data_loader_type = data.get('data_loader_type')
        data_loader_params = data.get('data_loader_params')
        table_name = data.get('table_name')

        if data_loader_type not in DATA_LOADERS:
            return jsonify({"status": "error", "message": f"Invalid data loader type. Must be one of: {', '.join(DATA_LOADERS.keys())}"}), 400

        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        with db_manager.connection(sid) as duck_db_conn:
            data_loader = DATA_LOADERS[data_loader_type](data_loader_params, duck_db_conn)
            data_loader.ingest_data(table_name)

            return jsonify({
                "status": "success",
                "message": "Successfully ingested data from data loader"
            })

    except Exception as e:
        logger.error(f"Error ingesting data from data loader: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error", 
            "message": safe_msg
        }), status_code
    

@tables_bp.route('/data-loader/view-query-sample', methods=['POST'])
def data_loader_view_query_sample():
    """View a sample of data from a query"""
    try:
        data = request.get_json()
        data_loader_type = data.get('data_loader_type')
        data_loader_params = data.get('data_loader_params')
        query = data.get('query')

        if data_loader_type not in DATA_LOADERS:
            return jsonify({"status": "error", "message": f"Invalid data loader type. Must be one of: {', '.join(DATA_LOADERS.keys())}"}), 400
        
        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        with db_manager.connection(sid) as duck_db_conn:
            data_loader = DATA_LOADERS[data_loader_type](data_loader_params, duck_db_conn)
            sample = data_loader.view_query_sample(query)

            return jsonify({
                "status": "success",
                "sample": sample,
                "message": "Successfully retrieved query sample"
            })
    except Exception as e:
        logger.error(f"Error viewing query sample: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error", 
            "sample": [],
            "message": safe_msg
        }), status_code
    

@tables_bp.route('/data-loader/ingest-data-from-query', methods=['POST'])
def data_loader_ingest_data_from_query():
    """Ingest data from a data loader"""
    try:
        data = request.get_json()
        data_loader_type = data.get('data_loader_type')
        data_loader_params = data.get('data_loader_params')
        query = data.get('query')
        name_as = data.get('name_as')

        if data_loader_type not in DATA_LOADERS:
            return jsonify({"status": "error", "message": f"Invalid data loader type. Must be one of: {', '.join(DATA_LOADERS.keys())}"}), 400

        sid = resolve_session_id()
        if not sid:
            return session_missing_response()

        with db_manager.connection(sid) as duck_db_conn:
            data_loader = DATA_LOADERS[data_loader_type](data_loader_params, duck_db_conn)
            data_loader.ingest_data_from_query(query, name_as)

            return jsonify({
                "status": "success",
                "message": "Successfully ingested data from data loader"
            })

    except Exception as e:
        logger.error(f"Error ingesting data from data loader: {str(e)}")
        safe_msg, status_code = sanitize_db_error_message(e)
        return jsonify({
            "status": "error", 
            "message": safe_msg
        }), status_code