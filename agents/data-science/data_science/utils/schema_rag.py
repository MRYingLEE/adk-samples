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
# To avoid circular dependency, get_bq_client could be moved to a more general utility
# or passed as an argument if schema_rag.py is imported by bigquery/tools.py.
# For now, attempting a direct import, but this might need refactoring.
from ..sub_agents.bigquery.tools import get_bq_client
from vertexai.language_models import TextEmbeddingModel
from google.cloud import bigquery # Added import

SCHEMA_EMBEDDING_MODEL_NAME = os.getenv("SCHEMA_EMBEDDING_MODEL_NAME", "text-embedding-004")

def get_column_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of texts."""
    model = TextEmbeddingModel.from_pretrained(SCHEMA_EMBEDDING_MODEL_NAME)
    embeddings = model.get_embeddings(texts)
    return [embedding.values for embedding in embeddings]

def get_relevant_schema_from_embeddings(question: str, project_id: str, rag_corpus_id: str) -> str:
    """
    Retrieves relevant schema details (primarily table DDLs) based on vector similarity to the question,
    querying a centralized RAG corpus in BigQuery that contains metadata and embeddings for schema elements.

    The RAG corpus table is expected to have at least 'embedding' (ARRAY<FLOAT64>) and 'table_ddl' (STRING) columns.
    'rag_corpus_id' should be in the format 'dataset_id.table_id'.
    """
    client = get_bq_client() # Assumes get_bq_client() is correctly implemented.
    question_embedding = get_column_embeddings([question])[0]

    if not rag_corpus_id:
        print("Error: BQ_METADATA_RAG_CORPUS_ID is not set. Cannot query schema embeddings.")
        return "-- ERROR: RAG Corpus ID not configured. --"

    if not project_id:
        print("Error: project_id is not set. Cannot query schema embeddings.")
        return "-- ERROR: project_id not configured. --"

    # rag_corpus_id is expected to be in the format "dataset_id.table_id"
    full_table_id = f"{project_id}.{rag_corpus_id}"

    print(f"Querying RAG Corpus: {full_table_id} for question: '{question}' using embeddings.")

    # Construct the BigQuery SQL query using ML.DISTANCE for cosine similarity.
    # This retrieves distinct DDLs for tables associated with the top N most relevant schema elements.
    query = f"""
    WITH RelevantElements AS (
        SELECT
            table_ddl,
            ML.DISTANCE(embedding, @question_embedding, 'COSINE') AS distance
        FROM
            `{full_table_id}`
        WHERE embedding IS NOT NULL AND table_ddl IS NOT NULL
    )
    SELECT DISTINCT table_ddl
    FROM RelevantElements
    ORDER BY distance ASC
    LIMIT 5  -- Retrieve DDLs for tables associated with the top 5 relevant elements
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("question_embedding", "FLOAT64", question_embedding),
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()  # Waits for the job to complete.

        ddls = [row.table_ddl for row in results if row.table_ddl]

        if not ddls:
            return f"-- No relevant DDLs found in '{full_table_id}' for the question: '{question}'. Ensure the table contains 'embedding' and 'table_ddl' columns and that embeddings match the model '{SCHEMA_EMBEDDING_MODEL_NAME}'. --"

        return "\n\n".join(ddls)

    except Exception as e:
        error_message = f"Error querying BigQuery RAG corpus '{full_table_id}': {e}. Check table structure, permissions, and if BigQuery ML API is enabled."
        print(error_message)
        return f"-- ERROR: {error_message} --"

# Future enhancements could include:
# - A function to build/update the RAG index from BigQuery schema if not using an external corpus.
# - Integration with a specific vector database client.
