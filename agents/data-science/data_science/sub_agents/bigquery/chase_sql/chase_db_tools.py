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

"""This code contains the implementation of the tools used for the CHASE-SQL agent."""

import enum
import os

from google.adk.tools import ToolContext

# pylint: disable=g-importing-member
from .dc_prompt_template import DC_PROMPT_TEMPLATE
from .llm_utils import GeminiModel
from .qp_prompt_template import QP_PROMPT_TEMPLATE
from .sql_postprocessor import sql_translator
from ..tools import get_bigquery_schema

# pylint: enable=g-importing-member

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")


class GenerateSQLType(enum.Enum):
    """Enum for the different types of SQL generation methods.

    DC: Divide and Conquer ICL prompting
    QP: Query Plan-based prompting
    """

    DC = "dc"
    QP = "qp"


def exception_wrapper(func):
    """A decorator to catch exceptions in a function and return the exception as a string.

    Args:
       func (callable): The function to wrap.

    Returns:
       callable: The wrapped function.
    """

    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Exception occurred in {func.__name__}: {str(e)}"

    return wrapped_function


def parse_response(response: str) -> str:
    """Parses the output to extract SQL content from the response.

    Args:
       response (str): The output string containing SQL query.

    Returns:
       str: The SQL query extracted from the response.
    """
    query = response
    try:
        if "```sql" in response and "```" in response:
            query = response.split("```sql")[1].split("```")[0]
    except ValueError as e:
        print(f"Error in parsing response: {e}")
        query = response
    return query.strip()


def initial_bq_nl2sql(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Generates an initial SQL query from a natural language question.

    Args:
      question: Natural language question.
      tool_context: Function context.

    Returns:
      str: An SQL statement to answer this question.
    """
    print("****** Running agent with ChaseSQL algorithm.")
    project = tool_context.state["database_settings"]["bq_project_id"]
    dataset_ids_list = tool_context.state["database_settings"]["bq_dataset_ids"] # This is the list of data datasets
    
    # Get RAG dataset information - now with separate project support
    rag_corpus_id = tool_context.state["database_settings"].get("BQ_METADATA_RAG_TABLE_ID") 
    rag_project_id = tool_context.state["database_settings"].get("bq_rag_project_id")
    
    if not rag_corpus_id:
        print("Warning: BQ_METADATA_RAG_TABLE_ID not found in tool_context. ChaseSQL RAG features may be limited.")
        # Fallback to using the main data dataset for RAG if not specified, or handle error
        # For CHASE, RAG is preferred. If rag_corpus_id is not set, schema retrieval might be limited
        # or fall back to a general schema dump if implemented that way in get_bigquery_schema.

    # Retrieve schema based on the question using RAG or by listing schemas for all datasets
    ddl_schema = get_bigquery_schema(
        project_id=project,
        question=question,
        rag_corpus_id=rag_corpus_id, # RAG corpus for embeddings lookup
        rag_project_id=rag_project_id, # Project where RAG corpus is located
        target_dataset_ids=dataset_ids_list # Pass the list of dataset IDs
    )
    if not ddl_schema:
        # Fallback or error handling if schema retrieval fails
        print("Warning: RAG schema retrieval failed. Falling back to full schema or predefined DDL.")
        # Optionally, load a default or full schema here if RAG fails
        # For now, we'll rely on a possible pre-loaded full ddl_schema in tool_context if available,
        # or handle the error if no schema can be provided.
        ddl_schema = tool_context.state["database_settings"].get("bq_ddl_schema", "")
        if not ddl_schema:
            raise ValueError("Schema could not be retrieved, and no fallback DDL schema is available.")

    transpile_to_bigquery = tool_context.state["database_settings"][
        "transpile_to_bigquery"
    ]
    process_input_errors = tool_context.state["database_settings"][
        "process_input_errors"
    ]
    process_tool_output_errors = tool_context.state["database_settings"][
        "process_tool_output_errors"
    ]
    number_of_candidates = tool_context.state["database_settings"][
        "number_of_candidates"
    ]
    model = tool_context.state["database_settings"]["model"]
    temperature = tool_context.state["database_settings"]["temperature"]
    generate_sql_type = tool_context.state["database_settings"]["generate_sql_type"]

    if generate_sql_type == GenerateSQLType.DC.value:
        prompt = DC_PROMPT_TEMPLATE.format(
            SCHEMA=ddl_schema, QUESTION=question, BQ_PROJECT_ID=BQ_PROJECT_ID
        )
    elif generate_sql_type == GenerateSQLType.QP.value:
        prompt = QP_PROMPT_TEMPLATE.format(
            SCHEMA=ddl_schema, QUESTION=question, BQ_PROJECT_ID=BQ_PROJECT_ID
        )
    else:
        raise ValueError(f"Unsupported generate_sql_type: {generate_sql_type}")

    model = GeminiModel(model_name=model, temperature=temperature)
    requests = [prompt for _ in range(number_of_candidates)]
    responses = model.call_parallel(requests, parser_func=parse_response)
    # Take just the first response.
    responses = responses[0]

    # If postprocessing of the SQL to transpile it to BigQuery is required,
    # then do it here.
    if transpile_to_bigquery:
        translator = sql_translator.SqlTranslator(
            model=model,
            temperature=temperature,
            process_input_errors=process_input_errors,
            process_tool_output_errors=process_tool_output_errors,
        )
        # pylint: disable=g-bad-todo
        # pylint: enable=g-bad-todo
        responses: str = translator.translate(
            responses, ddl_schema=ddl_schema, db=dataset_ids_list[0] if dataset_ids_list else None, catalog=project
        )

    return responses
