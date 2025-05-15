import re
from typing import List


def extract_core_definition(code_snippet: str) -> str | None:
    """
    Extracts the core "keyword name" string from a Lean code snippet.
    Handles cases where keyword and name are on the same or separate lines,
    correctly processing decorators and comments.
    Returns "keyword name" if found, otherwise None.
    """
    keywords = {
        "def",
        "theorem",
        "lemma",
        "structure",
        "inductive",
        "class",
        "abbrev",
        "axiom",
        "axioms",
        "example",
    }
    lines = code_snippet.strip().split("\n")
    first_potential_def_line_index = -1
    first_meaningful_line_content = None
    line_content_after_decorator = None

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("--"):
            continue

        potential_content_for_keyword_check = stripped_line
        if stripped_line.startswith("@"):
            # Handles multiple consecutive decorators like @[foo] @[bar]
            potential_content_for_keyword_check = re.sub(
                r"^(\s*@\[.*?\]\s*)+", "", stripped_line
            ).lstrip()
            if (
                not potential_content_for_keyword_check
                or potential_content_for_keyword_check.startswith("--")
            ):
                continue

        has_keyword = False
        for keyword in keywords:
            if re.match(rf"^\s*{keyword}(\s+|$)", potential_content_for_keyword_check):
                has_keyword = True
                break

        if has_keyword:
            first_potential_def_line_index = i
            first_meaningful_line_content = line
            line_content_after_decorator = potential_content_for_keyword_check
            break

    if first_meaningful_line_content is None:
        return None

    found_keyword = None
    definition_name = None

    for keyword in keywords:
        match = re.match(rf"^\s*{keyword}\s+(\S+)", line_content_after_decorator)
        if match:
            found_keyword = keyword
            definition_name = match.group(1)
            # Clean up potential trailing characters like ':', '(', etc.
            definition_name = re.match(r"^[^:(]*", definition_name).group(0)
            return f"{keyword} {definition_name}"

        match_keyword_alone = re.match(
            rf"^\s*{keyword}\s*$", line_content_after_decorator
        )
        if match_keyword_alone:
            found_keyword = keyword
            break

    if found_keyword and definition_name is None:
        for j in range(first_potential_def_line_index + 1, len(lines)):
            next_line = lines[j]
            stripped_next_line = next_line.strip()
            if (
                not stripped_next_line
                or stripped_next_line.startswith("@")
                or stripped_next_line.startswith("--")
            ):
                continue

            # Check if this line starts with another keyword (likely error or nested def)
            is_another_keyword = False
            for kw in keywords:
                if stripped_next_line.startswith(kw):
                    # Check if it's the keyword followed by space or end of line
                    if (
                        len(stripped_next_line) == len(kw)
                        or stripped_next_line[len(kw)].isspace()
                    ):
                        is_another_keyword = True
                        break
            # Stop searching for name if another definition seems to start
            if is_another_keyword:
                return None

            name_match = re.match(r"^\s*(\S+)", stripped_next_line)
            if name_match:
                definition_name = name_match.group(1)
                # Clean up potential trailing characters like ':', '(', etc.
                definition_name = re.match(r"^[^:(]*", definition_name).group(0)
                return f"{found_keyword} {definition_name}"
            else:
                return None

    return None


def extract_signature(code_snippet: str) -> str | None:
    """
    Extracts the signature of a Lean definition or theorem, excluding decorators,
    the final type/proposition, and the definition body.
    The signature includes keywords, name, parameters, and hypotheses.
    It stops before the last ':' that introduces the type/proposition or before ':='.
    """
    lines = code_snippet.strip().split("\n")
    start_line_index = -1
    meaningful_lines_with_indices = []

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        is_comment = stripped_line.startswith("--")
        is_empty = not stripped_line

        if start_line_index == -1 and not is_empty and not is_comment:
            start_line_index = i

        if start_line_index != -1 and not is_comment:
            meaningful_lines_with_indices.append((i, line))

    if start_line_index == -1:
        return None

    # Priority: 1) End before last ':', 2) End before ' := ' if no relevant ':' found before it.
    end_line_idx = -1
    end_pos = -1
    found_colon_sep = False

    for i, line in reversed(meaningful_lines_with_indices):
        content_before_comment = line.split("--", 1)[0]
        pos = content_before_comment.rfind(":")
        # Need to be careful: avoid ':' inside parameters like (a : Nat) if possible
        # Heuristic: Assume the *last* colon overall before a potential ':=' is the type separator.
        # A more robust solution would need deeper parsing.
        if pos != -1:
            pos_assign = content_before_comment.find(":=", pos)
            if pos_assign == -1:
                end_line_idx = i
                end_pos = pos
                found_colon_sep = True
                break

    if not found_colon_sep:
        for i, line in reversed(meaningful_lines_with_indices):
            content_before_comment = line.split("--", 1)[0]
            pos = content_before_comment.rfind(":=")
            if pos != -1:
                end_line_idx = i
                end_pos = pos
                break

    signature_lines = []
    if end_line_idx != -1:
        separator_found_original_index = end_line_idx

        for i, line in meaningful_lines_with_indices:
            original_index = i
            if original_index < separator_found_original_index:
                signature_lines.append(line.split("--", 1)[0].rstrip())
            elif original_index == separator_found_original_index:
                separator_line_content = line.split("--", 1)[0]
                last_line_part = separator_line_content[:end_pos].rstrip()
                if last_line_part:
                    signature_lines.append(last_line_part)
                break
    else:
        for i, line in meaningful_lines_with_indices:
            signature_lines.append(line.split("--", 1)[0].rstrip())

    if not signature_lines:
        return None

    full_signature_with_decorators = "\n".join(
        line for line in signature_lines if line.strip()
    )

    # Regex to find decorators like @[foo], @[bar], potentially with whitespace
    decorator_pattern = r"^(\s*@\[.*?\]\s*)+"
    final_signature = re.sub(
        decorator_pattern, "", full_signature_with_decorators
    ).strip()

    return final_signature if final_signature else None


def remove_comments(code: str) -> str:
    """
    Removes single-line (--) and block (/- ... -/) comments from Lean code.
    """
    code = re.sub(r"/-.*?-\/", "", code, flags=re.DOTALL)
    code = re.sub(r"--.*", "", code)
    return code


def remove_align(code: str) -> str:
    """
    Removes all aligned definitions from Lean code.
    """
    return re.sub(r"#align.*", "", code)
