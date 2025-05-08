# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for Schema RAG (Retrieval Augmented Generation) functionality.

This module provides functionality to:
1. Extract schema information from BigQuery datasets
2. Create embeddings for the schema information
3. Store the embeddings in a BigQuery vector store
4. Retrieve relevant schema information during chat time
"""

import os
from pathlib import Path
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv, set_key
import vertexai
from vertexai import rag
from google.cloud import bigquery

# Define the path to the .env file
env_file_path = Path(__file__).parent.parent.parent / ".env"

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=env_file_path)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
schema_corpus_name = os.getenv("SCHEMA_RAG_CORPUS_NAME", "")

SCHEMA_CORPUS_DISPLAY_NAME = "schema_rag_corpus"
SCHEMA_DATASET_ID = "schema_rag_dataset"
SCHEMA_TABLE_ID = "schema_embeddings"

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")


def create_schema_rag_corpus():
    """Creates a RAG corpus for schema information.
    
    Returns:
        str: The name of the created corpus.
    """
    # Configure embedding model (text-embedding-005)
    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )

    # Configure the vector database backend
    backend_config = rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    )

    # Create the corpus
    schema_corpus = rag.create_corpus(
        display_name=SCHEMA_CORPUS_DISPLAY_NAME,
        backend_config=backend_config,
    )

    # Save the corpus name in the environment file
    write_corpus_to_env(schema_corpus.name)

    return schema_corpus.name


def extract_schema_metadata(
    project_id: str, dataset_id: str, table_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extracts schema metadata from BigQuery tables.
    
    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: Optional. A specific table ID to extract schema from.
                  If None, extracts schema from all tables in the dataset.
    
    Returns:
        List[Dict[str, Any]]: A list of schema metadata dictionaries.
    """
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    metadata_list = []
    
    # If a specific table is specified, only extract schema from that table
    if table_id:
        tables = [client.get_table(f"{project_id}.{dataset_id}.{table_id}")]
    else:
        tables = list(client.list_tables(dataset_ref))
        tables = [client.get_table(table.reference) for table in tables]
    
    for table in tables:
        # Skip views and other non-table objects
        if table.table_type != "TABLE":
            continue
            
        table_metadata = {
            "table_id": table.table_id,
            "dataset_id": dataset_id,
            "project_id": project_id,
            "full_table_name": f"{project_id}.{dataset_id}.{table.table_id}",
            "description": table.description or "",
            "num_rows": table.num_rows,
            "columns": []
        }
        
        for field in table.schema:
            column_metadata = {
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description or "",
                "is_nullable": field.mode == "NULLABLE"
            }
            
            # Create a rich text representation of the column
            column_text = (
                f"Table: {table.table_id}\n"
                f"Column: {field.name}\n"
                f"Type: {field.field_type}\n"
                f"Mode: {field.mode}\n"
            )
            
            if field.description:
                column_text += f"Description: {field.description}\n"
                
            column_metadata["text_representation"] = column_text
            table_metadata["columns"].append(column_metadata)
        
        metadata_list.append(table_metadata)
    
    return metadata_list


def create_schema_files_for_rag(schema_metadata: List[Dict[str, Any]]) -> List[str]:
    """Creates temporary files from schema metadata for RAG ingestion.
    
    Args:
        schema_metadata: A list of schema metadata dictionaries.
        
    Returns:
        List[str]: A list of temporary file paths.
    """
    import tempfile
    import os
    
    temp_files = []
    
    for table_metadata in schema_metadata:
        # Create a temporary file for the table schema
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write table information
            f.write(f"Table: {table_metadata['table_id']}\n")
            f.write(f"Dataset: {table_metadata['dataset_id']}\n")
            f.write(f"Project: {table_metadata['project_id']}\n")
            f.write(f"Full Name: {table_metadata['full_table_name']}\n")
            
            if table_metadata['description']:
                f.write(f"Description: {table_metadata['description']}\n")
                
            f.write(f"Number of Rows: {table_metadata['num_rows']}\n\n")
            f.write("Columns:\n")
            
            # Write column information
            for column in table_metadata['columns']:
                f.write(f"  - Name: {column['name']}\n")
                f.write(f"    Type: {column['type']}\n")
                f.write(f"    Mode: {column['mode']}\n")
                
                if column['description']:
                    f.write(f"    Description: {column['description']}\n")
                
                f.write("\n")
            
            temp_files.append(f.name)
    
    return temp_files


def ingest_schema_to_rag(corpus_name: str, schema_files: List[str]) -> None:
    """Ingests schema files into the RAG corpus.
    
    Args:
        corpus_name: The name of the RAG corpus.
        schema_files: A list of file paths containing schema information.
    """
    # Configure chunking for better retrieval
    transformation_config = rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=256,  # Smaller chunks for schema information
            chunk_overlap=50,
        ),
    )
    
    # Import the schema files into the RAG corpus
    rag.import_files(
        corpus_name,
        schema_files,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=1000,
    )
    
    # Clean up temporary files
    for file_path in schema_files:
        if os.path.exists(file_path):
            os.remove(file_path)


def retrieve_schema_context(query: str) -> str:
    """Retrieves relevant schema context for a query.
    
    Args:
        query: The user's query string.
        
    Returns:
        str: Schema information relevant to the query.
    """
    corpus_name = os.getenv("SCHEMA_RAG_CORPUS_NAME")
    if not corpus_name:
        return "Schema RAG corpus not configured. Please run setup_schema_rag() first."
    
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=5,  # Return top 5 relevant schema chunks
        filter=rag.Filter(vector_distance_threshold=0.5),
    )
    
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=query,
        rag_retrieval_config=rag_retrieval_config,
    )
    
    # Process and format the response
    formatted_response = "Relevant Schema Information:\n\n"
    
    for idx, chunk in enumerate(response.chunks, 1):
        formatted_response += f"--- Schema Chunk {idx} ---\n"
        formatted_response += chunk.content + "\n\n"
    
    return formatted_response


def write_corpus_to_env(corpus_name: str) -> None:
    """Writes the schema corpus name to the .env file.
    
    Args:
        corpus_name: The name of the corpus.
    """
    load_dotenv(env_file_path)  # Load existing variables
    
    # Update the SCHEMA_RAG_CORPUS_NAME in the .env file
    set_key(env_file_path, "SCHEMA_RAG_CORPUS_NAME", corpus_name)
    print(f"SCHEMA_RAG_CORPUS_NAME '{corpus_name}' written to {env_file_path}")


def setup_schema_rag(project_id: str, dataset_id: str, table_id: Optional[str] = None) -> str:
    """Sets up the entire Schema RAG pipeline.
    
    This function:
    1. Creates a RAG corpus for schema information
    2. Extracts schema metadata from BigQuery
    3. Creates temporary files for RAG ingestion
    4. Ingests schema into the RAG corpus
    
    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: Optional. A specific table ID to extract schema from.
        
    Returns:
        str: A status message indicating the result of the setup.
    """
    try:
        # Step 1: Create the RAG corpus
        print("Creating schema RAG corpus...")
        corpus_name = create_schema_rag_corpus()
        
        # Step 2: Extract schema metadata
        print(f"Extracting schema metadata from {project_id}.{dataset_id}...")
        schema_metadata = extract_schema_metadata(project_id, dataset_id, table_id)
        
        if not schema_metadata:
            return f"No tables found in {project_id}.{dataset_id}"
        
        # Step 3: Create temporary files for RAG ingestion
        print("Creating schema files for RAG ingestion...")
        schema_files = create_schema_files_for_rag(schema_metadata)
        
        # Step 4: Ingest schema into the RAG corpus
        print(f"Ingesting schema into corpus: {corpus_name}")
        ingest_schema_to_rag(corpus_name, schema_files)
        
        return (f"Schema RAG setup complete. Ingested schema for "
                f"{len(schema_metadata)} tables into corpus: {corpus_name}")
    
    except Exception as e:
        return f"Error setting up Schema RAG: {str(e)}"


if __name__ == "__main__":
    project_id = os.getenv("BQ_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET_ID")
    
    if not project_id or not dataset_id:
        print("Error: BQ_PROJECT_ID or BQ_DATASET_ID environment variables not set.")
        exit(1)
    
    print(f"Setting up Schema RAG for {project_id}.{dataset_id}...")
    result = setup_schema_rag(project_id, dataset_id)
    print(result)