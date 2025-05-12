# RAG on Columns for Enhanced Database Schema Understanding

This document outlines the implementation of Retrieval Augmented Generation (RAG) focused on database column metadata within the Data Science agent. This approach allows the agent to dynamically retrieve and utilize the most relevant parts of a database schema (tables and columns) when attempting to understand or generate queries based on a user's natural language question.

## Core Idea

Instead of relying on a static, potentially very large, full database schema, the RAG on columns feature enables:
1.  **Embedding Schema Metadata:** Generating vector embeddings for database column names, descriptions, and other relevant metadata.
2.  **Dynamic Retrieval:** When a user asks a question, the question is embedded, and a similarity search is performed against the corpus of schema embeddings.
3.  **Contextual Schema:** The top-k most relevant schema parts (e.g., DDL for relevant tables, specific column details) are retrieved and provided to the language model as context for query generation or analysis.

This significantly improves the agent's ability to handle large and complex databases by focusing only on the pertinent schema information.

## Implementation Details

The core logic for RAG on columns is primarily located in `agents/data-science/data_science/utils/schema_rag.py` and integrated into the BigQuery tools in `agents/data-science/data_science/sub_agents/bigquery/tools.py`.

### 1. Generating Column Embeddings

The `get_column_embeddings` function in `agents/data-science/data_science/utils/schema_rag.py` is responsible for converting textual descriptions of schema elements (like column names or user questions) into vector embeddings using a pre-trained text embedding model.

```python
# File: agents/data-science/data_science/utils/schema_rag.py
# ...
def get_column_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of texts."""
    model = TextEmbeddingModel.from_pretrained(SCHEMA_EMBEDDING_MODEL_NAME)
    embeddings = model.get_embeddings(texts)
    return [embedding.values for embedding in embeddings]
# ...
```

### 2. Retrieving Relevant Schema via RAG

The `get_relevant_schema_from_embeddings` function in `agents/data-science/data_science/utils/schema_rag.py` orchestrates the RAG process. It takes the user's question, generates an embedding for it, and then (currently a placeholder) queries a vector store containing the pre-computed schema embeddings.

```python
# File: agents/data-science/data_science/utils/schema_rag.py
# ...
def get_relevant_schema_from_embeddings(question: str, project_id: str, rag_corpus_id: str) -> str:
    """
    Retrieves relevant schema details (tables and columns) based on vector similarity to the question,
    querying a centralized RAG corpus that contains metadata for all configured datasets.

    Note: This is a placeholder implementation. Actual RAG querying logic 
    (e.g., connecting to a vector DB, performing similarity search) needs to be implemented.
    """
    # client = get_bq_client() # BQ client might be needed for further schema details post-RAG.
    question_embedding = get_column_embeddings([question])[0]

    if not rag_corpus_id:
        print("Error: BQ_METADATA_RAG_CORPUS_ID is not set. Cannot query schema embeddings.")
        return "-- ERROR: RAG Corpus ID not configured. --"

    print(f"Querying RAG Corpus: {rag_corpus_id} for question: {question} using embeddings (Placeholder).")
    # TODO: Implement actual RAG querying logic here.
    # This would involve:
    # 1. Connecting to the vector store where schema embeddings are stored.
    # 2. Performing a similarity search with the question_embedding.
    # 3. Retrieving the top-k relevant schema parts (e.g., DDL for tables, column names/descriptions).
    # 4. Formatting the retrieved schema information into a string.
    
    # For now, returning a placeholder string indicating what would be done.
    return f"-- Placeholder: DDLs for tables relevant to '{question}' from RAG corpus '{rag_corpus_id}' would be listed here based on embedding search.\n"
# ...
```
The `BQ_METADATA_RAG_CORPUS_ID` environment variable is crucial here, as it points to the identifier for the RAG corpus containing the schema embeddings.

### 3. Integration into BigQuery Tools

The RAG functionality is integrated into the `get_bigquery_schema` function within `agents/data-science/data_science/sub_agents/bigquery/tools.py`. This function conditionally uses RAG if a question and RAG corpus ID are provided.

```python
# File: agents/data-science/data_science/sub_agents/bigquery/tools.py
# ...
from data_science.utils.schema_rag import get_relevant_schema_from_embeddings as get_relevant_schema_via_rag
# ...
BQ_METADATA_RAG_CORPUS_ID = os.getenv("BQ_METADATA_RAG_CORPUS_ID")
# ...
def get_bigquery_schema(
    client=None, 
    project_id=None, 
    question: str = None, 
    rag_corpus_id: str = None,
    target_dataset_ids: list[str] = None 
    ):
    """
    Retrieves schema. If a question and rag_corpus_id are provided, it uses RAG.
    Otherwise, if target_dataset_ids are provided, it gets all tables for those.
    """
    if question and project_id and rag_corpus_id:
        print(f"Retrieving schema relevant to the question using RAG corpus: {rag_corpus_id}...")
        # Use the new RAG function
        return get_relevant_schema_via_rag(question, project_id, rag_corpus_id)

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

1.  **Embedding Generation**: When a database is configured, its schema (table names, column names, column descriptions, data types, etc.) is processed. Each relevant piece of metadata (e.g., "table: customers", "column: email_address (VARCHAR) - The primary email address of the customer") is converted into a numerical vector (embedding) using a text embedding model (e.g., `text-embedding-004` from Vertex AI).

2.  **Storing Embeddings**: These embeddings, along with their corresponding metadata (like the DDL statement for the table the column belongs to, or the column description itself), are stored in a centralized RAG corpus. In the reference implementation, this corpus is a BigQuery table. This table must contain at least an `embedding` column (ARRAY<FLOAT64>) and a `table_ddl` column (STRING) which stores the DDL for the table associated with the embedded element.
    *   The `rag_corpus_id` parameter in `get_relevant_schema_from_embeddings` refers to this BigQuery table in the format `dataset_id.table_id`.

3.  **Querying with RAG**: When the agent needs to generate SQL for a user's question:
    *   The user's natural language question is converted into an embedding using the same text embedding model.
    *   This question embedding is used to query the RAG corpus (the BigQuery table containing schema embeddings).
    *   The query uses `ML.DISTANCE` (specifically cosine similarity) to find schema elements whose embeddings are most similar to the question embedding.
    *   The `get_relevant_schema_from_embeddings` function retrieves the `table_ddl` for the tables associated with the top N most relevant schema elements.

4.  **Informing SQL Generation**: The retrieved DDL statements (representing the most relevant parts of the database schema) are then provided as context to the LLM when it generates the SQL query. This helps the LLM understand the structure of the relevant tables and columns, leading to more accurate and efficient SQL generation.

**Current Implementation (`get_relevant_schema_from_embeddings`):**

*   Takes the user's `question`, `project_id`, and `rag_corpus_id` (e.g., `my_dataset.my_schema_embeddings_table`) as input.
*   Generates an embedding for the input `question`.
*   Constructs a BigQuery SQL query that:
    *   Selects from the specified RAG corpus table (`project_id.dataset_id.table_id`).
    *   Calculates the cosine distance between the `question_embedding` and the stored `embedding` for each schema element in the corpus.
    *   Orders the results by this distance (ascending, so most similar comes first).
    *   Returns the distinct `table_ddl` for the top 5 most relevant schema elements.
*   If successful, it returns a string containing the DDLs of the most relevant tables, separated by double newlines.
*   Includes error handling for missing configuration or issues during the BigQuery query execution.

This RAG-based approach allows the agent to dynamically fetch only the most pertinent schema information for a given question, rather than overwhelming the LLM with the entire database schema, especially for large and complex databases.

## Future Enhancements
-   Implementation of the actual RAG querying logic against a vector database.
-   Functions to build and update the RAG index from BigQuery schema.

This RAG on columns feature provides a more intelligent and scalable way for the Data Science agent to interact with database schemas.
