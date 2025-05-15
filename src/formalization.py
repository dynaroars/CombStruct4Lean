from typing import List
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, model_validator
import re

from .common import (
    DEFAULT_LAKE_PATH,
    DEFAULT_LEAN_WORKSPACE,
    read_system_prompt,
    compile_lean4,
)
from .search import (
    ToolResponse,
    create_searcher_instance,
    SearchResponse,
    SearchResult,
    SearchResultType,
)
from .common import BaseGenerator


class FormalizationResponse(BaseModel):
    header: str = Field(..., description="Header of the formalization")
    structure: str = Field(
        ..., description="Code snippet for the combinatorial structure"
    )
    theorem: str = Field(..., description="Code snippet for the formal statement")

    @model_validator(mode="after")
    def validate_lean4(self, info: ValidationInfo) -> "FormalizationResponse":
        """
        Validate whether the code is in Lean 4 by checking if there's any code segment that started with `begin` and ended with `end` using regex. If such segment is found, raise a ValueError.
        """

        def search_for_begin_end(code: str) -> bool:
            return re.search(r"\bbegin\b[\s\S]*?\bend\b", code)

        if search_for_begin_end(self.theorem) or search_for_begin_end(self.structure):
            raise ValueError(
                "The code contains `begin` and `end` keywords, which are not supported in Lean 4."
            )

        def verify_header(header: str) -> bool:
            # Allow empty or whitespace-only headers
            if not header.strip():
                return True

            # Check for disallowed Lean3 import patterns (e.g., "import data.real.basic")
            if re.search(r"\bimport\s+data\.", header):
                return False

            return True

        if not verify_header(self.header):
            raise ValueError("The header contains disallowed Lean3 import patterns.")

        return self

    def validate_compilation(
        self, lake_path: str, lean_workspace: str
    ) -> "FormalizationResponse":
        code = self.to_code()
        status, err = compile_lean4(
            code,
            lake_path=lake_path,
            lean_workspace=lean_workspace,
        )
        if not status:
            raise ValueError(f"Compilation error! Feedback:\n\n```lean\n{err}\n```\n")
        return self

    def to_code(self) -> str:
        return f"{self.header}\n\n{self.structure}\n\n{self.theorem}"


class FormalizationPipeline:
    """
    Pipeline for formalizing an informal statement into a formal theorem in Lean 4.
    """

    FORMALIZATION_PROMPT = """Translate the following problem into a formal theorem in Lean4:
```
{problem}
```
"""

    GUIDED_FEEDBACK_PROMPT = """Guided feedback from expert:
```
{feedback}
```
"""

    FEEDBACK_GENERATION_PROMPT = """Problem Description:
{problem}

Previous Feedback:
```
{previous_feedback}
```

Code:
```lean
{previous_code}
```

Traceback/Errors:
```
{traceback}
```

Candidate Theorems:
```lean
{candidate_theorem_code}
```
"""

    SEARCH_PROMPT = """Code:
```lean
{previous_code}
```

Traceback/Errors:
```
{traceback}
```
"""

    def __init__(
        self,
        client_config: dict,
        formalize_gen_config: dict,
        search_gen_config: dict,
        feedback_gen_config: dict,
        lake_path: str = DEFAULT_LAKE_PATH,
        lean_workspace: str = DEFAULT_LEAN_WORKSPACE,
        search_config: dict | None = None,
        max_retries: int = 3,
        num_max_iterations: int = 5,
    ):
        self.client_config = client_config
        self.formalize_gen_config = formalize_gen_config
        self.search_gen_config = search_gen_config
        self.feedback_gen_config = feedback_gen_config
        self.lake_path = lake_path
        self.lean_workspace = lean_workspace
        self.max_retries = max_retries
        self.num_max_iterations = num_max_iterations

        # Initialize separate generators for each step
        self.formalize_generator = BaseGenerator(client_config, **formalize_gen_config)
        self.search_generator = BaseGenerator(client_config, **search_gen_config)
        self.feedback_generator = BaseGenerator(client_config, **feedback_gen_config)

        if search_config is None:
            search_config = {
                "encoded_data_path": None,
                "model_name_or_path": None,
                "top_k": 10,
            }
        self.search_config = search_config
        self.search_tools = create_searcher_instance(**self.search_config)

    def log_traceback(self, exception: Exception):
        self.traceback = str(exception)
        logger.error(f"Traceback: {self.traceback}")

    async def formalize(
        self,
        informal_statement: str,
        previous_code: str | None = None,
        guided_feedback: str | None = None,
    ) -> FormalizationResponse:
        messages = [
            {"role": "system", "content": read_system_prompt("formalize")},
            {"role": "user", "content": informal_statement},
        ]
        if previous_code:
            messages.append({"role": "assistant", "content": previous_code})
            if guided_feedback:
                messages.append(
                    {
                        "role": "user",
                        "content": self.GUIDED_FEEDBACK_PROMPT.format(
                            feedback=guided_feedback
                        ),
                    }
                )
        response = await self.formalize_generator.prompt(
            messages=messages,
            response_model=FormalizationResponse,
            context={
                "lake_path": self.lake_path,
                "lean_workspace": self.lean_workspace,
            },
            max_retries=self.max_retries,
        )
        return response

    async def generate_query(self, previous_code: str, traceback: str) -> ToolResponse:
        messages = [
            {"role": "system", "content": read_system_prompt("search")},
            {
                "role": "user",
                "content": self.SEARCH_PROMPT.format(
                    previous_code=previous_code, traceback=traceback
                ),
            },
        ]
        response = await self.search_generator.prompt_tool(
            messages=messages, max_retries=self.max_retries
        )
        return response

    async def generate_feedback(
        self,
        problem: str,
        previous_code: str,
        traceback: str,
        queries: List[str] | None = None,
        search_responses: List[SearchResponse] | None = None,
        previous_feedback: str | None = None,
    ) -> str:
        # Generate candidate theorem code if search results are available
        if queries and search_responses:
            candidate_theorem_code = ""
            for query, response in zip(queries, search_responses):
                candidate_theorem_code += f"Premises related to {query}:\n"
                if response.type == SearchResultType.SUCCESS:
                    candidate_theorem_code += "\n\n".join(
                        [result.code for result in response.results]
                    )
                else:
                    candidate_theorem_code += "N/A"
                candidate_theorem_code += "\n---\n"
        else:
            candidate_theorem_code = "N/A"

        logger.debug(f"Candidate theorem code: {candidate_theorem_code}")
        messages = [
            {"role": "system", "content": read_system_prompt("feedback")},
            {
                "role": "user",
                "content": self.FEEDBACK_GENERATION_PROMPT.format(
                    problem=problem,
                    previous_code=previous_code,
                    traceback=traceback,
                    candidate_theorem_code=candidate_theorem_code,
                    previous_feedback=previous_feedback if previous_feedback else "N/A",
                ),
            },
        ]
        res = await self.feedback_generator.prompt(
            messages=messages, response_model=str, max_retries=self.max_retries
        )
        return res

    async def run(
        self, problem: str, enable_search: bool = True, enable_feedback: bool = True
    ) -> FormalizationResponse:
        guided_feedback = None
        previous_code = None
        formalization_response = None

        for itr in range(self.num_max_iterations):
            logger.info(f"Iteration {itr + 1} of {self.num_max_iterations}")
            logger.debug("Formalizing...")

            formalization_response = await self.formalize(
                problem, previous_code, guided_feedback
            )
            try:
                formalization_response.validate_compilation(
                    lake_path=self.lake_path,
                    lean_workspace=self.lean_workspace,
                )
                return formalization_response

            except (ValidationError, ValueError) as e:
                traceback = str(e)
                previous_code = formalization_response.to_code()
                logger.debug(f"Validation error: {traceback}")
                logger.debug(f"Previous code: {previous_code}")

                if itr == self.num_max_iterations - 1:
                    raise ValueError("Failed to formalize the problem")

                tool_response = None
                search_responses = None

                if enable_search:
                    # Generate search results
                    logger.debug("Generating query...")
                    tool_response = await self.generate_query(previous_code, traceback)
                    logger.debug(f"Tool response: {tool_response}")
                    search_responses = tool_response.execute(
                        self.search_tools, num_results=self.search_config["top_k"]
                    )
                    logger.debug(f"Search responses: {search_responses}")

                if enable_feedback:
                    # Generate feedback with or without search results
                    logger.debug("Generating feedback...")
                    queries = tool_response.queries if tool_response else None
                    guided_feedback = await self.generate_feedback(
                        problem,
                        previous_code,
                        traceback,
                        queries=queries,
                        search_responses=search_responses,
                    )
                    logger.debug(f"Guided feedback: {guided_feedback}")
                else:
                    guided_feedback = (
                        f"Previous attempt failed with error:\n{traceback}"
                    )

            logger.debug(f"Guided feedback: {guided_feedback}")

        raise ValueError("Failed to formalize the problem")
