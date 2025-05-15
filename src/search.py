import multiprocessing
import pickle
from typing import Any, Dict, List, Tuple, Optional, Union
from loguru import logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
import torch
import textdistance
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import json
from dataclasses import dataclass
from enum import Enum
import traceback
import os

# --- Constants ---
# Local search data keys
KEY_NAME = "extracted_core_definition"
KEY_CODE = "code"
KEY_TENSOR = "core_definition_embedding"
FILEPATH_KEY = "filepath"


# --- Data Classes and Enums ---
@dataclass
class SearchResult:
    """Common search result model"""

    name: Optional[str] = None
    code: Optional[str] = None
    doc_string: Optional[str] = None
    doc_url: Optional[str] = None
    kind: Optional[str] = None


class SearchResultType(Enum):
    EMPTY = "empty"
    SUCCESS = "success"
    FAILURE = "failure"


class SearchResponse:
    """Common response model"""

    def __init__(
        self,
        type: SearchResultType,
        results: Optional[List[SearchResult]] = None,
        error: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.type = type
        self.results = results or []
        self.error = error
        self.suggestions = suggestions

    @classmethod
    def empty(cls) -> "SearchResponse":
        return cls(type=SearchResultType.EMPTY)

    @classmethod
    def success(cls, results: List[SearchResult]) -> "SearchResponse":
        return cls(type=SearchResultType.SUCCESS, results=results)

    @classmethod
    def failure(
        cls, error: str, suggestions: Optional[List[str]] = None
    ) -> "SearchResponse":
        return cls(type=SearchResultType.FAILURE, error=error, suggestions=suggestions)

    def __str__(self) -> str:
        if self.type == SearchResultType.SUCCESS:
            return f"Success ({len(self.results)} results)"
        if self.type == SearchResultType.EMPTY:
            return "Empty Result"
        if self.type == SearchResultType.FAILURE:
            return f"Failure: {self.error}" + (
                f" Suggestions: {self.suggestions}" if self.suggestions else ""
            )
        return "Unknown SearchResponse Type"


# --- Caching ---
class SearchCache:
    """Generic cache for search queries"""

    def __init__(self):
        self._cache: Dict[Tuple, SearchResponse] = {}

    def get(self, key: Tuple) -> Optional[SearchResponse]:
        return self._cache.get(key)

    def insert(self, key: Tuple, value: SearchResponse) -> None:
        self._cache[key] = value


# --- Helper Functions ---
def _read_encoded_data(
    path: str,
) -> Optional[Tuple[List[Dict[str, Any]], torch.Tensor]]:
    """Reads pickled data containing docs, embeddings, and filepaths."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list) or not data:
            logger.warning(f"Data file {path} empty/invalid.")
            return None
        valid_data = [
            item
            for item in data
            if isinstance(item, dict)
            and all(k in item for k in [KEY_TENSOR, KEY_NAME, FILEPATH_KEY])
        ]
        if len(valid_data) != len(data):
            logger.warning(
                f"Filtered {len(data) - len(valid_data)} items missing keys from {path}."
            )
        if not valid_data:
            logger.warning(f"No valid data entries in {path}.")
            return None
        embeddings = torch.stack(
            [torch.from_numpy(item[KEY_TENSOR]) for item in valid_data], dim=0
        )
        processed_data = [
            {
                "name": item[KEY_NAME],
                "code": item[KEY_CODE],
                "filepath": item[FILEPATH_KEY],
            }
            for item in valid_data
        ]
        logger.info(f"Loaded {len(processed_data)} items from {path}.")
        return processed_data, embeddings
    except FileNotFoundError:
        logger.warning(f"Data file not found: {path}.")
        return None
    except Exception as e:
        logger.error(f"Error reading data {path}: {e}")
        logger.debug(traceback.format_exc())
        return None


def _preprocess_query(query: str) -> str:
    """Basic query preprocessing."""
    if not isinstance(query, str):
        logger.warning(f"Non-string query: {type(query)}.")
        return ""
    parts = query.split("--")
    processed_query = parts[0].strip()
    return processed_query.replace("\n", " ")


# --- Unified Searcher Class ---
class Searcher:
    """Central class for performing local searches."""

    def __init__(
        self,
        encoded_data_path: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
        top_k: int = 10,
    ):
        self.top_k = top_k
        self.local_cache = SearchCache()
        self.model = model
        self.encoded_data: Optional[List[Dict[str, Any]]] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.local_search_enabled = False

        if encoded_data_path and os.path.exists(encoded_data_path):
            load_result = _read_encoded_data(encoded_data_path)
            if load_result:
                self.encoded_data, self.embeddings = load_result
                self.local_search_enabled = True
                logger.info("Local search enabled.")
                if not self.model or self.embeddings is None:
                    logger.warning(
                        "Semantic search disabled (model/embeddings missing)."
                    )
            else:
                logger.warning(
                    f"Failed to load {encoded_data_path}. Local search disabled."
                )
        elif encoded_data_path:
            logger.warning(
                f"Path '{encoded_data_path}' not found. Local search disabled."
            )
        else:
            logger.info("No local data path provided. Local search disabled.")

    def search_local(
        self,
        query: str,
        num_results: Optional[int] = None,
        lean_type: Optional[str] = None,
    ) -> SearchResponse:
        """Searches local data based on semantic similarity."""
        if (
            not self.local_search_enabled
            or self.model is None
            or self.embeddings is None
        ):
            return SearchResponse.failure(
                "Semantic search disabled (no data/model/embeddings)."
            )
        processed_query = _preprocess_query(query)
        k = num_results or self.top_k
        if not processed_query:
            return SearchResponse.empty()
        cache_key = ("semantic", processed_query, k)
        cached_result = self.local_cache.get(cache_key)
        if cached_result:
            return cached_result
        try:
            results = self._perform_semantic_search(
                processed_query, k, self.model, self.embeddings, lean_type
            )
            response = SearchResponse.success(results)
            self.local_cache.insert(cache_key, response)
            return response
        except Exception as e:
            logger.exception(f"Error during semantic search: {e}")
            return SearchResponse.failure(f"Semantic search failed: {e}")

    def _perform_semantic_search(
        self,
        query: str,
        k: int,
        model: SentenceTransformer,
        embeddings: torch.Tensor,
        lean_type: Optional[str] = None,
    ) -> List[SearchResult]:
        if self.encoded_data is None:
            return []
        query_embedding = model.encode(query, convert_to_tensor=True)
        search_results = semantic_search(query_embedding, embeddings, top_k=k * 2)[0]
        if lean_type:
            filtered_search_results = [
                res
                for res in search_results
                if lean_type in self.encoded_data[res["corpus_id"]]["code"]
            ]
            if filtered_search_results:
                search_results = filtered_search_results
        search_results = search_results[:k]
        results = []
        for res in search_results:
            idx = res["corpus_id"]
            if 0 <= idx < len(self.encoded_data):
                item = self.encoded_data[idx]
                results.append(
                    SearchResult(
                        name=item["name"],
                        code=item["code"].split(":=")[0].strip(),
                        doc_url=item["filepath"],
                    )
                )
            else:
                logger.warning(f"Semantic search invalid corpus_id: {idx}")
        return results


# --- Factory Function ---
def create_searcher_instance(
    model_name_or_path,
    encoded_data_path: Optional[str] = None,
    top_k: int = 10,
) -> Searcher:
    """Creates and initializes a Searcher instance."""
    logger.info(
        f"Creating Searcher. Local data: {encoded_data_path}. Model: {model_name_or_path is not None}"
    )
    model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
    return Searcher(encoded_data_path, model, top_k)


# --- ToolResponse Definition ---
class ToolResponse(BaseModel):
    """Model representing a function call requested by the LLM."""

    function: str = Field(
        default="search_local",
        description="The name of the search method to call.",
    )
    queries: List[str] = Field(..., description="The search query strings")
    lean_type: List[str] = Field(..., description="Filter results by Lean type")

    @field_validator("function", mode="after")
    def validate_function(cls, v):
        if v not in ["search_local"]:
            raise ValueError(f"Invalid function: {v}")
        return v

    @model_validator(mode="after")
    def validate_query(self, info: ValidationInfo) -> "ToolResponse":
        if self.lean_type and len(self.lean_type) == 1:
            self.lean_type = self.lean_type * len(self.queries)
        if self.lean_type is not None and len(self.queries) != len(self.lean_type):
            raise ValueError(
                f"Queries and lean_type must have the same length. Queries: {self.queries}, Lean type: {self.lean_type}"
            )
        if not self.lean_type:
            self.lean_type = [None] * len(self.queries)
        return self

    def execute(
        self, searcher: "Searcher", num_results: int = 5
    ) -> List[SearchResponse]:
        """Executes the search method on the Searcher instance."""
        func_name = self.function
        method_to_call = getattr(searcher, func_name, None)
        if not callable(method_to_call):
            logger.error(f"Method '{func_name}' not found on Searcher or not callable.")
            raise AttributeError(f"Method '{func_name}' not found or not callable.")

        results = []
        for query, lean_type in zip(self.queries, self.lean_type):
            args: Dict[str, Any] = {
                "query": query,
                "num_results": num_results,
                "lean_type": lean_type,
            }

            try:
                logger.info(f"Executing tool: {self.function}({args})")
                result = method_to_call(**args)
                if not isinstance(result, SearchResponse):
                    logger.warning(
                        f"{self.function} returned unexpected type {type(result)}"
                    )
                    return SearchResponse.failure(
                        f"Internal error: Tool returned {type(result)}"
                    )
                logger.info(f"Tool {self.function} result: {result}")
                results.append(result)
            except TypeError as e:
                logger.error(f"TypeError executing {self.function}({args}). Error: {e}")
                raise TypeError(
                    f"Argument mismatch for '{self.function}'. Error: {e}"
                ) from e
            except Exception as e:
                logger.exception(f"Exception during {self.function}({args}): {e}")
                raise RuntimeError(
                    f"Tool execution failed for {self.function}: {e}"
                ) from e

        return results
