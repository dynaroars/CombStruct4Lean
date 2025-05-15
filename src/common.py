from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Tuple, TypeVar, Union

import json
import os
import subprocess
import tempfile
import time
import traceback
import instructor
from pydantic import BaseModel
from anthropic import Anthropic, AsyncAnthropic
from loguru import logger
from openai import OpenAI, AsyncOpenAI
from litellm import completion, acompletion

from src.ast_parser import lean4_parser
from .search import ToolResponse

T = TypeVar("T", bound=BaseModel)

# HOME_DIR = os.path.expanduser("~")
DEFAULT_LAKE_PATH = f"{os.path.expanduser('~')}/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "mathlib4/"


class ClientFactory:
    @staticmethod
    def from_anthropic(
        is_async: bool = False,
        mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
        **kwargs,
    ) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
        if is_async:
            client = AsyncAnthropic(**kwargs)
        else:
            client = Anthropic(**kwargs)
        # client = instructor.patch(client)
        return instructor.from_anthropic(client, mode=mode)

    @staticmethod
    def from_openai(
        is_async: bool = False, mode: instructor.Mode = instructor.Mode.TOOLS, **kwargs
    ) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
        if is_async:
            client = AsyncOpenAI(**kwargs)
        else:
            client = OpenAI(**kwargs)
        client = instructor.patch(client)
        return instructor.from_openai(client, mode=mode)

    @staticmethod
    def from_litellm(
        is_async: bool = False, mode: instructor.Mode = instructor.Mode.TOOLS, **kwargs
    ) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
        completion_func = acompletion if is_async else completion
        return instructor.from_litellm(completion_func, mode=mode)

    @staticmethod
    def get_client(
        **kwargs,
    ) -> Union[instructor.Instructor, instructor.AsyncInstructor]:
        engine = kwargs.pop("engine", "openai")
        assert engine in ["openai", "anthropic", "litellm"]

        is_async = kwargs.pop("is_async", False)

        mode = kwargs.pop("mode", "")

        if not mode:
            if engine == "openai":
                mode = instructor.Mode.TOOLS
            elif engine == "anthropic":
                mode = instructor.Mode.ANTHROPIC_TOOLS
            elif engine == "litellm":
                mode = instructor.Mode.TOOLS

        if engine == "anthropic":
            return ClientFactory.from_anthropic(is_async=is_async, mode=mode, **kwargs)
        elif engine == "openai":
            return ClientFactory.from_openai(is_async=is_async, mode=mode, **kwargs)
        elif engine == "litellm":
            return ClientFactory.from_litellm(is_async=is_async, mode=mode, **kwargs)


class BaseGenerator(Generic[T]):
    """
    Base class for generating structured responses using large language models.

    This class provides a foundation for creating model-specific generators that
    return structured data as Pydantic models. It handles both synchronous and
    asynchronous API calls, response parsing, and retries.

    Attributes:
        client: The LLM client used for generating completions
        is_async: Whether the client should operate asynchronously
        generation_config: Configuration parameters for text generation

    Raises:
        ValueError: If client_config or generation parameters are invalid or output generated is invalid
    """

    def __init__(self, client_config: dict, **gen_config):
        """
        Initialize a new BaseGenerator.

        Args:
            client_config: Configuration for the underlying LLM client,
                           including API keys and endpoint information
            **gen_config: Generation parameters like model, temperature, max_tokens
                          that will be used for all prompt calls
        """
        self.is_async = client_config.pop("is_async", False)
        self.client = ClientFactory.get_client(**client_config)
        self.generation_config = gen_config

    def reset(self):
        self.client.clear()

    def add_hook(self, hook: callable, event: str):
        self.client.on(event, hook)

    async def prompt(
        self,
        messages: List[Dict[str, str]],
        response_model: type[T],
        *args,
        return_with_completion: bool = False,
        max_retries: int = 3,
        is_iterable: bool = False,
        **kwargs,
    ) -> Union[
        T,
        Iterable[T],
        Tuple[T, Any],
        Tuple[Iterable[T], Any],
    ]:
        """
        Send a prompt to the language model and get a structured response.

        Args:
            messages: List of message dictionaries to send to the LLM
            response_model: Pydantic model class to structure the response
            *args: Additional positional arguments to pass to the client
            return_with_completion: If True, returns a tuple of (structured_response, raw_completion)
            max_retries: Maximum number of retry attempts for failed requests
            is_iterable: If True, expects and processes an iterable of responses
            **kwargs: Additional keyword arguments to pass to the client

        Returns:
            Based on the parameters, one of:
            - A single structured response (T)
            - An iterable of structured responses (Iterable[T])
            - A tuple with structured response and raw completion (Tuple[T, Any])
            - A tuple with iterable of structured responses and raw completion (Tuple[Iterable[T], Any])

        Raises:
            ValueError: If the output generated is invalid
        """

        def create_fn(*args, **kwargs):
            return self.client.chat.completions.create(*args, **kwargs)

        def create_with_completion_fn(*args, **kwargs):
            return self.client.chat.completions.create_with_completion(*args, **kwargs)

        def selected_fn():
            fn = create_with_completion_fn if return_with_completion else create_fn
            return fn(
                *args,
                messages=messages,
                response_model=(
                    response_model if not is_iterable else Iterable[response_model]
                ),
                max_retries=max_retries,
                model=self.generation_config["model"],
                temperature=self.generation_config["temperature"],
                max_tokens=self.generation_config["max_tokens"],
                **kwargs,
            )

        @wraps(selected_fn)
        async def async_wrapper():
            result = selected_fn()
            if hasattr(result, "__await__"):
                return await result
            return result

        @wraps(selected_fn)
        def sync_wrapper():
            return selected_fn()

        if self.is_async:
            return await async_wrapper()
        else:
            return sync_wrapper()

    async def prompt_tool(
        self,
        messages: List[Dict[str, str]],
        *args,
        return_with_completion: bool = False,
        max_retries: int = 3,
        **kwargs,
    ) -> Union[
        ToolResponse,
        Tuple[ToolResponse, Any],
    ]:
        """
        Send a prompt to the language model expecting a tool use response.

        Args:
            messages: List of message dictionaries to send to the LLM
            *args: Additional positional arguments to pass to the client
            return_with_completion: If True, returns a tuple of (structured_response, raw_completion)
            max_retries: Maximum number of retry attempts for failed requests
            **kwargs: Additional keyword arguments to pass to the client

        Returns:
            Based on the parameters, one of:
            - A tool response (ToolResponse)
            - A tuple with tool response and raw completion (Tuple[ToolResponse, Any])

        Raises:
            ValueError: If the output generated is invalid or does not conform to ToolResponse.
        """
        return await self.prompt(
            messages=messages,
            response_model=ToolResponse,
            *args,
            return_with_completion=return_with_completion,
            max_retries=max_retries,
            is_iterable=False,  # Tool calls are expected to be singular
            **kwargs,
        )


def _verify_lean4_file(
    code,
    lake_path=DEFAULT_LAKE_PATH,
    lean_workspace=DEFAULT_LEAN_WORKSPACE,
    last_env=None,
    verbose=False,
    timeout=300,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
):
    command = dict(
        cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises
    )
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        logger.info(message_str)
    start_time = time.time()
    system_messages = ""
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout,
            )
        # logger.debug(f"Output stdout: {outputs.stdout}")
        # logger.debug(f"Output stderr: {outputs.stderr}")
        result = json.loads(outputs.stdout)
        ast_results = (
            lean4_parser(code, result["ast"])
            if "ast" in result and result["ast"]
            else {}
        )
        result = {
            "sorries": result.get("sorries", []),
            "tactics": result.get("tactics", []),
            "errors": [
                m for m in result.get("messages", []) if m["severity"] == "error"
            ],
            "warnings": [
                m for m in result.get("messages", []) if m["severity"] == "warning"
            ],
            "infos": [m for m in result.get("messages", []) if m["severity"] == "info"],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            "verified_code": code,
        }
        result["pass"] = not result["errors"]
        result["complete"] = (
            result["pass"]
            and not result["sorries"]
            and not any(
                "declaration uses 'sorry'" in warning["data"]
                or "failed" in warning["data"]
                for warning in result["warnings"]
            )
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages,
        }
    result["verify_time"] = time.time() - start_time
    return result


def extract_text_from_string(
    text: str, start_line: int, start_col: int, end_line: int, end_col: int
) -> str:
    # Split the input string into lines
    lines = text.splitlines()

    # Extract the relevant lines from start_line to end_line (1-indexed)
    extracted_lines = lines[start_line - 1 : end_line]

    # Adjust the start and end columns on the first and last lines
    if start_line == end_line:
        # Single line extraction case
        extracted_text = extracted_lines[0][start_col - 1 : end_col]
    else:
        # Multi-line extraction case
        extracted_lines[0] = extracted_lines[0][
            start_col - 1 :
        ]  # First line from start_col
        extracted_lines[-1] = extracted_lines[-1][:end_col]  # Last line up to end_col
        extracted_text = "".join(extracted_lines)

    return extracted_text


def compile_lean4(code: str, lake_path: str, lean_workspace: str) -> Tuple[bool, str]:
    """
    Compile the formalization code using Lean4 compiler

    Args:
        code (str): The formalization code

    Returns:
        Tuple[bool, str]: A tuple of boolean indicating whether the code is valid and a string of error message if the code is invalid
    """
    try:
        res = _verify_lean4_file(
            code,
            lake_path=lake_path,
            lean_workspace=lean_workspace,
            verbose=False,
            timeout=300,
            allTactics=False,
            ast=True,
            premises=False,
            tactics=False,
        )
    except Exception as e:
        logger.error(
            f"Error verifying Lean 4 code with AST: {e}\n\nRetrying without AST..."
        )
        res = _verify_lean4_file(
            code,
            lake_path=lake_path,
            lean_workspace=lean_workspace,
            verbose=False,
            timeout=300,
            allTactics=False,
            ast=False,
            premises=False,
            tactics=False,
        )
    status = res["pass"]
    if status:
        errs = ""
    else:
        errs = []
        logger.debug(res)
        if "errors" in res:
            for err in res["errors"]:
                text = extract_text_from_string(
                    code,
                    start_line=err["pos"]["line"],
                    start_col=err["pos"]["column"],
                    end_line=err["endPos"]["line"],
                    end_col=err["endPos"]["column"],
                )
                error_feedback = err["data"]
                error_msg = f"Text: {text}\nError: {error_feedback}"
                if error_msg not in errs:
                    errs.append(error_msg)
                    logger.debug(error_msg)
            errs = "\n\n".join(errs)
            errs += "\n"
        else:
            err_message = "Code cannot compiled. Unknown error."
            logger.debug(err_message)
            errs = err_message
    return status, errs


def read_system_prompt(filename: str) -> str:
    """Read a system prompt from the markdown file."""
    prompt_path = Path(f"prompts/{filename}.md")
    with open(prompt_path, "r") as f:
        return f.read()
