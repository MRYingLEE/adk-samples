# filepath: c:\Users\SkyTi\teq\adk-samples\agents\data-science\data_science\utils\schema_rag.py
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

"""Utility functions for performing RAG over database schemas."""

import os
from vertexai.language_models import TextEmbeddingModel
from google.cloud import bigquery

SCHEMA_EMBEDDING_MODEL_NAME = os.getenv("SCHEMA_EMBEDDING_MODEL_NAME", "text-embedding-004")

def construct_text_for_column_embedding(dataset_name: str, dataset_description: str,
                                        table_name: str, table_description: str,
                                        column_name: str, column_description: str,
                                        column_data_type: str) -> str:
    """
    Constructs a single string from column metadata for embedding generation.
    This function would be used during the RAG corpus population phase.
    Example: "Dataset: sales_data. Dataset Description: Contains all sales transactions. Table: orders. Table Description: Information about customer orders. Column: order_date. Column Description: The date when the order was placed. Column Data Type: DATE"
    """
    parts = [
        f"Dataset: {dataset_name}",
        f"Dataset Description: {dataset_description}" if dataset_description else None,
        f"Table: {table_name}",
        f"Table Description: {table_description}" if table_description else None,
        f"Column: {column_name}",
        f"Column Description: {column_description}" if column_description else None,
        f"Column Data Type: {column_data_type}"
    ]
    return ". ".join(filter(None, parts))

def get_column_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of texts."""
    model = TextEmbeddingModel.from_pretrained(SCHEMA_EMBEDDING_MODEL_NAME)
    embeddings = model.get_embeddings(texts)
    return [embedding.values for embedding in embeddings]

def get_relevant_schema_from_embeddings(
    question: str,
    project_id: str,
    rag_corpus_id: str,
    bq_client: bigquery.Client,
    top_k_columns: int = 10,
    location: str = None
) -> str:
    """
    Retrieves relevant schema details (top K columns and their table DDLs) based on vector similarity to the question,
    querying a centralized RAG corpus in BigQuery using the provided BigQuery client.

    The RAG corpus table is expected to have at least:
    - 'embedding' (ARRAY<FLOAT64>): Embedding of the column's descriptive text (e.g., generated from construct_text_for_column_embedding).
    - 'dataset_name' (STRING)
    - 'table_name' (STRING)
    - 'column_name' (STRING)
    - 'column_data_type' (STRING)
    - 'column_description' (STRING, optional)
    - 'table_ddl' (STRING): The DDL of the table the column belongs to.
    
    Args:
        question: User's question to match against column embeddings.
        project_id: GCP project ID where the RAG corpus is located.
        rag_corpus_id: ID of the RAG corpus in format 'dataset_id.table_id'.
        bq_client: BigQuery client to use for querying.
        top_k_columns: Number of most relevant columns to retrieve.
        location: BigQuery dataset location (e.g., 'US', 'europe-west1').
        
    Returns:
        A string containing relevant schemas based on the query.
    """
    question_embedding = get_column_embeddings([question])[0]

    if not rag_corpus_id:
        print("Error: BQ_METADATA_RAG_CORPUS_ID is not set. Cannot query schema embeddings.")
        return "-- ERROR: RAG Corpus ID not configured. --"

    if not project_id:
        print("Error: project_id is not set. Cannot query schema embeddings.")
        return "-- ERROR: project_id not configured. --"

    # If rag_corpus_id already contains project_id in format 'project_id.dataset_id.table_id', extract parts
    corpus_parts = rag_corpus_id.split('.')
    if len(corpus_parts) == 3:  # Format is project_id.dataset_id.table_id
        full_table_id = rag_corpus_id
    elif len(corpus_parts) == 2:  # Format is dataset_id.table_id
        full_table_id = f"{project_id}.{rag_corpus_id}"
    else:  # Unexpected format
        print(f"Error: Invalid RAG corpus ID format: {rag_corpus_id}")
        return f"-- ERROR: Invalid RAG corpus ID format: {rag_corpus_id} --"

    print(f"Querying RAG Corpus: {full_table_id} for question: '{question}' to find top {top_k_columns} columns.")

    query = f"""
    WITH RankedColumns AS (
        SELECT
            dataset_name,
            table_name,
            column_name,
            data_type as column_data_type,
            COALESCE(column_description, 'N/A') as column_description,
            ML.DISTANCE(embedding, @question_embedding, 'COSINE') AS distance
        FROM
            `{full_table_id}`
        WHERE embedding IS NOT NULL AND column_name IS NOT NULL
        ORDER BY distance ASC
        LIMIT {top_k_columns}
    )
    SELECT
        rc.dataset_name,
        rc.table_name,
        rc.column_name,
        rc.column_data_type,
        rc.column_description,
        -- Include all columns from the same table to construct a complete table definition
        STRING_AGG(DISTINCT CONCAT(other.column_name, ' ', other.data_type), ',\n    ') AS all_columns
    FROM RankedColumns rc
    JOIN `{full_table_id}` other
    ON rc.dataset_name = other.dataset_name AND rc.table_name = other.table_name
    GROUP BY rc.dataset_name, rc.table_name, rc.column_name, rc.column_data_type, rc.column_description, rc.distance
    ORDER BY rc.distance ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("question_embedding", "FLOAT64", question_embedding),
        ]
    )

    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()

        if not results.total_rows:
            return f"-- No relevant columns found in '{full_table_id}' for the question: '{question}'. Ensure the RAG corpus is populated correctly with column-level embeddings and metadata, and that embeddings match the model '{SCHEMA_EMBEDDING_MODEL_NAME}'. --"

        # Accumulate column definitions per table instead of using full table_ddl
        table_column_defs = {}
        column_details_parts = []

        for row in results:
            table_key = f"{row.dataset_name}.{row.table_name}"
            # collect only relevant column definitions
            table_column_defs.setdefault(table_key, []).append(f"{row.column_name} {row.column_data_type}")

            column_info = (
                f"- Dataset: {row.dataset_name}, Table: {row.table_name}, Column: {row.column_name}, "
                f"Type: {row.column_data_type}, Description: {row.column_description}"
            )
            column_details_parts.append(column_info)

        output_parts = ["-- Relevant Columns based on your question:"]
        output_parts.extend(column_details_parts)
        output_parts.append("\n-- Simplified Table Schemas for selected columns:")

        for table_key, cols in table_column_defs.items():
            # build a partial CREATE TABLE statement with only selected columns
            ddl = (
                f"CREATE TABLE {table_key} (\n    " \
                + ",\n    ".join(cols) \
                + "\n);"
            )
            output_parts.append(ddl)

        return "\n".join(output_parts)

    except Exception as e:
        print(f"Error querying schema embeddings: {e}")
        return f"-- ERROR querying RAG corpus '{full_table_id}': {e} --"

# Future enhancements could include:
# - A function to build/update the RAG index from BigQuery schema if not using an external corpus.
# - Integration with a specific vector database client.
