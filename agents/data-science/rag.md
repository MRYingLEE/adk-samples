# RAG on Columns for Enhanced Database Schema Understanding

This document outlines the implementation of Retrieval Augmented Generation (RAG) focused on individual database column metadata within the Data Science agent. This approach allows the agent to dynamically retrieve and utilize the most relevant parts of a database schema (specifically, details of individual columns and their parent tables) when attempting to understand or generate queries based on a user's natural language question.

## Core Idea

Instead of relying on a static, potentially very large, full database schema, or even just table-level DDLs, the RAG on columns feature enables:
1.  **Detailed Column Embedding:** Generating vector embeddings for each database column. The text used for embedding a column is a rich combination of its dataset's name and description, its parent table's name and description, and the column's own name, data type, and description. This creates a comprehensive representation of each column in the semantic space.
2.  **Dynamic Column-Level Retrieval:** When a user asks a question, the question is embedded. A similarity search is then performed against the corpus of these detailed column embeddings.
3.  **Contextual Schema (Column-Focused):** The top-k most relevant columns are identified. The information retrieved includes not only the details of these columns (name, type, description) but also the DDL of their parent tables. This highly specific and contextual schema information is then provided to the language model.

This significantly improves the agent's ability to handle large and complex databases by focusing on the most pertinent column-level details and their associated table structures, leading to more accurate SQL generation and schema interpretation.

## Implementation Details

The core logic for RAG on columns is primarily located in `agents/data-science/data_science/utils/schema_rag.py` and integrated into the BigQuery tools in `agents/data-science/data_science/sub_agents/bigquery/tools.py`.

### 1. Generating Column Embeddings

The `construct_text_for_column_embedding` function in `agents/data-science/data_science/utils/schema_rag.py` is used to create a detailed textual description for each column. This description includes metadata from the dataset, table, and the column itself.

```python
# File: agents/data-science/data_science/utils/schema_rag.py
# ...
def construct_text_for_column_embedding(dataset_name: str, dataset_description: str,
                                        table_name: str, table_description: str,
                                        column_name: str, column_description: str,
                                        column_data_type: str) -> str:
    """
    Constructs a single string from column metadata for embedding generation.
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
# ...
```

### 2. Retrieving Relevant Schema via RAG

The `get_relevant_schema_from_embeddings` function in `agents/data-science/data_science/utils/schema_rag.py` orchestrates the RAG process. It takes the user's question, generates an embedding for it, and then queries a BigQuery table (acting as a RAG corpus) that stores the pre-computed column embeddings and their associated metadata.

```python
# File: agents/data-science/data_science/utils/schema_rag.py
# ...
def get_relevant_schema_from_embeddings(
    question: str,
    project_id: str,
    rag_corpus_id: str,
    bq_client: bigquery.Client, # Added bq_client
    top_k_columns: int = 10    # Added top_k_columns
) -> str:
    """
    Retrieves relevant schema details (top K columns and their table DDLs) based on vector similarity to the question,
    querying a centralized RAG corpus in BigQuery using the provided BigQuery client.

    The RAG corpus table is expected to have at least:
    - 'embedding' (ARRAY<FLOAT64>): Embedding of the column's descriptive text.
    - 'dataset_name' (STRING)
    - 'table_name' (STRING)
    - 'column_name' (STRING)
    - 'column_data_type' (STRING)
    - 'column_description' (STRING, optional)
    - 'table_ddl' (STRING): DDL of the table the column belongs to.
    'rag_corpus_id' should be in the format 'dataset_id.table_id'.
    """
    question_embedding = get_column_embeddings([question])[0]

    if not rag_corpus_id:
        print("Error: BQ_METADATA_RAG_CORPUS_ID is not set. Cannot query schema embeddings.")
        return "-- ERROR: RAG Corpus ID not configured. --"

    if not project_id:
        print("Error: project_id is not set. Cannot query schema embeddings.")
        return "-- ERROR: project_id not configured. --"

    full_table_id = f"{project_id}.{rag_corpus_id}"

    print(f"Querying RAG Corpus: {full_table_id} for question: '{question}' to find top {top_k_columns} columns.")

    query = f"""
    WITH RankedColumns AS (
        SELECT
            dataset_name,
            table_name,
            column_name,
            column_data_type,
            COALESCE(column_description, 'N/A') as column_description,
            table_ddl,
            ML.DISTANCE(embedding, @question_embedding, 'COSINE') AS distance
        FROM
            `{full_table_id}`
        WHERE embedding IS NOT NULL AND table_ddl IS NOT NULL AND column_name IS NOT NULL
        ORDER BY distance ASC
        LIMIT {top_k_columns}
    )
    SELECT
        rc.dataset_name,
        rc.table_name,
        rc.column_name,
        rc.column_data_type,
        rc.column_description,
        rc.table_ddl
    FROM RankedColumns rc
    ORDER BY rc.distance ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("question_embedding", "FLOAT64", question_embedding),
        ]
    )
    # ... (rest of the function for querying BigQuery and formatting results)
# ...
```
The `BQ_METADATA_RAG_CORPUS_ID` environment variable is crucial here, as it points to the BigQuery table (e.g., `my_dataset.my_column_embeddings_table`) that serves as the RAG corpus.

### 3. Integration into BigQuery Tools

The RAG functionality is integrated into the `get_bigquery_schema` function within `agents/data-science/data_science/sub_agents/bigquery/tools.py`. This function conditionally uses RAG if a question and RAG corpus ID are provided.

```python
# File: agents/data-science/data_science/sub_agents/bigquery/tools.py
# ...
from data_science.utils.schema_rag import get_relevant_schema_from_embeddings as get_relevant_schema_via_rag
# ...
# BQ_METADATA_RAG_CORPUS_ID is fetched within update_database_settings
# ...
def get_bigquery_schema(
    client=None,
    project_id=None,
    question: str = None,
    # rag_corpus_id is now fetched from database_settings
    target_dataset_ids: list[str] = None,
    top_k_columns: int = 10 # Added for RAG
    ):
    """
    Retrieves schema. If a question is provided, it uses RAG to get column-level details.
    Otherwise, if target_dataset_ids are provided, it gets all tables for those.
    """
    rag_corpus_id = database_settings.get("bq_metadata_rag_corpus_id") if database_settings else None

    if question and project_id and rag_corpus_id and client: # Ensure client is available
        print(f"Retrieving schema relevant to the question using RAG corpus: {rag_corpus_id}...")
        return get_relevant_schema_via_rag(
            question=question,
            project_id=project_id,
            rag_corpus_id=rag_corpus_id,
            bq_client=client, # Pass the client
            top_k_columns=top_k_columns # Pass top_k
        )

    if not target_dataset_ids:
        return "-- No specific datasets provided for full schema dump and no question for RAG-based retrieval --\n"
    # ... (rest of the function for full schema dump) ...
# ...
```

### 4. Configuration

The RAG system relies on environment variables for configuration, particularly `BQ_PROJECT_ID`, `BQ_DATASET_IDS`, and `BQ_METADATA_RAG_CORPUS_ID`. These are managed in `update_database_settings` within `agents/data-science/data_science/sub_agents/bigquery/tools.py`.

```python
# File: agents/data-science/data_science/sub_agents/bigquery/tools.py
# ...
def update_database_settings():
    """Update database settings."""
    global database_settings
    
    project_id = get_env_var("BQ_PROJECT_ID")
    dataset_ids_str = get_env_var("BQ_DATASET_IDS")
    metadata_rag_corpus_id = get_env_var("BQ_METADATA_RAG_CORPUS_ID")

    if not dataset_ids_str:
        raise ValueError("BQ_DATASET_IDS environment variable is not set.")
    if not metadata_rag_corpus_id:
        print("Warning: BQ_METADATA_RAG_CORPUS_ID is not set. RAG-based schema retrieval will be limited.")
    
    dataset_ids = [ds_id.strip() for ds_id in dataset_ids_str.split(',')]

    # The ddl_overview can be simplified as dynamic RAG will be the primary source for question-specific schema
    ddl_overview = f"-- Schema for datasets ({', '.join(dataset_ids)}) is primarily retrieved dynamically via RAG from corpus: {metadata_rag_corpus_id} when a question is provided. Otherwise, full schema for targeted datasets is fetched. --\n"

    database_settings = {
        "bq_project_id": project_id,
        "bq_dataset_ids": dataset_ids, # List of dataset IDs
        "bq_metadata_rag_corpus_id": metadata_rag_corpus_id, # Central RAG corpus for schema
        "bq_ddl_schema": ddl_overview, # Overview or placeholder
        **chase_constants.chase_sql_constants_dict,
    }
    return database_settings
# ...
```

### RAG for Schema Retrieval

The Data Science agent can perform Retrieval Augmented Generation (RAG) over database schemas to ground its SQL generation capabilities. This involves:

1.  **Embedding Generation**: When a database is configured, its schema is processed column by column. For each column, a comprehensive textual description is created using `construct_text_for_column_embedding`. This text, which includes dataset name/description, table name/description, and column name/description/data type, is converted into a numerical vector (embedding) using a text embedding model (e.g., `text-embedding-004` from Vertex AI).

2.  **Storing Embeddings**: These column-level embeddings, along with their corresponding metadata (dataset name, table name, column name, column data type, column description, and the DDL statement for the table the column belongs to), are stored in a centralized RAG corpus. In the reference implementation, this corpus is a BigQuery table. This table must contain columns for `embedding` (ARRAY<FLOAT64>), `dataset_name` (STRING), `table_name` (STRING), `column_name` (STRING), `column_data_type` (STRING), `column_description` (STRING, optional), and `table_ddl` (STRING).
    *   The `rag_corpus_id` parameter in `get_relevant_schema_from_embeddings` refers to this BigQuery table in the format `dataset_id.table_id`.

3.  **Querying with RAG**: When the agent needs to generate SQL for a user's question:
    *   The user's natural language question is converted into an embedding using the same text embedding model.
    *   This question embedding is used to query the RAG corpus (the BigQuery table containing column schema embeddings).
    *   The query uses `ML.DISTANCE` (specifically cosine similarity) to find columns whose descriptive embeddings are most similar to the question embedding.
    *   The `get_relevant_schema_from_embeddings` function retrieves details for the top K most relevant columns (dataset, table, column name, type, description) and the distinct `table_ddl` for their parent tables.

4.  **Informing SQL Generation**: The retrieved column details and the DDL statements of their parent tables are then provided as context to the LLM when it generates the SQL query. This helps the LLM understand the structure and meaning of the most relevant columns and tables, leading to more accurate and efficient SQL generation.

**Current Implementation (`get_relevant_schema_from_embeddings`):**

*   Takes the user's `question`, `project_id`, `rag_corpus_id` (e.g., `my_dataset.my_schema_embeddings_table`), a `bigquery.Client` instance, and an optional `top_k_columns` (defaulting to 10) as input.
*   Generates an embedding for the input `question`.
*   Constructs a BigQuery SQL query that:
    *   Selects from the specified RAG corpus table (`project_id.dataset_id.table_id`).
    *   Calculates the cosine distance between the `question_embedding` and the stored `embedding` for each column in the corpus.
    *   Orders the results by this distance (ascending, so most similar comes first) and limits to `top_k_columns`.
    *   Returns the dataset name, table name, column name, column data type, column description, and the table DDL for these top columns.
*   If successful, it returns a formatted string containing:
    *   A list of the "Relevant Columns based on your question," with details for each.
    *   The "Corresponding Table DDLs (schema)" for all unique tables to which the relevant columns belong.
*   Includes error handling for missing configuration or issues during the BigQuery query execution.

This RAG-based approach allows the agent to dynamically fetch only the most pertinent column-level schema information for a given question, rather than overwhelming the LLM with the entire database schema, especially for large and complex databases.

## Future Enhancements
- Functions to build and update the RAG index from BigQuery schema (e.g., a script to iterate through `INFORMATION_SCHEMA.COLUMNS` and `INFORMATION_SCHEMA.TABLES` to populate the RAG corpus table).
- Integration with a specific vector database client as an alternative or supplement to BigQuery `ML.DISTANCE`.

This RAG on columns feature provides a more intelligent and scalable way for the Data Science agent to interact with database schemas.
