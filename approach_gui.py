# streamlit_app.py
# ------------------------------------------------------------
# CrewAI-based Metamorphic Testing for REST APIs (Behave version)
# - Multi-agent: MR -> Step-codegen -> Execute (Behave)
# - One feature file per iteration; one scenario per MR
# - AST repair loop for generated step code (syntax safety)
# - Dynamic Operation Coverage via process-wide requests hook (sitecustomize)
# - Static Operation Coverage via code scan
# - Optional: Schemathesis baseline
#
# Quickstart:
#   pip install streamlit crewai litellm requests pyyaml jsonschema behave
#   # optional baselines / validators:
#   pip install schemathesis openapi-core
#   streamlit run streamlit_app.py
#
# Security:
# - Never hardcode API keys; use env or Streamlit secrets.
# ------------------------------------------------------------

import os
import io
import re
import ast
import sys
import json
import time
import yaml
import shutil
import tempfile
import subprocess
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import textwrap
import streamlit as st
from crewai import Agent, Task, Crew, LLM
import requests
from urllib.parse import urlparse
import traceback
from urllib.parse import urlsplit
from openai import OpenAI
from run_logger import append_run_to_csv
from coordinator import TestManager
# ----------------- Utils -----------------

_HTTP_METHODS = {"GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"}
_MR_ID_PATTERN = re.compile(r"\b(MR[A-Za-z0-9_-]+)\b")

os.environ["OPENAI_API_KEY"] = ""


def _s(x) -> str:
    """Always return a string for CrewAI template inputs."""
    return "" if x is None else str(x)

def clean_step_code(step_code: str) -> str:
    """Remove duplicate imports, BASE_URL lines, and duplicate step functions (whole block)."""
    import re
    seen_imports, seen_funcs = set(), set()
    cleaned_lines = []
    func_pattern = re.compile(r"^def\s+([a-zA-Z0-9_]+)\s*\(")

    lines = step_code.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Deduplicate imports + BASE_URL
        if stripped.startswith(("import ", "from behave", "BASE_URL")):
            if stripped in seen_imports:
                i += 1
                continue
            seen_imports.add(stripped)
            cleaned_lines.append(line)
            i += 1
            continue

        # Deduplicate functions (entire block)
        m = func_pattern.match(stripped)
        if m:
            func_name = m.group(1)
            if func_name in seen_funcs:
                # Skip until the next top-level function or EOF
                i += 1
                while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t")):
                    i += 1
                continue
            seen_funcs.add(func_name)

        cleaned_lines.append(line)
        i += 1

    return "\n".join(cleaned_lines).strip() + "\n"

def _escape_braces(text: str) -> str:
    """Escape braces to keep `.format`/Crew templating from treating them as placeholders."""
    if text is None:
        return ""
    return str(text).replace("{", "{{").replace("}", "}}")

def _template_paths_from_spec(spec_obj):
    out = set()
    for p, methods in (spec_obj.get("paths") or {}).items():
        for m in (methods or {}).keys():
            out.add((m.upper(), p))
    return out

# FIX: derive path prefixes from base_url and OAS servers to normalize runtime hits
def _extract_path_prefixes(spec_obj: dict, base_url: str) -> list[str]:
    """Return possible leading path prefixes (e.g., '/api/v3') to strip from runtime URLs."""
    prefixes = set()

    # From BASE_URL
    try:
        base_path = urlparse(base_url).path or ""
        base_path = base_path.rstrip("/")
        if base_path and not base_path.startswith("/"):
            base_path = "/" + base_path
        if base_path:
            prefixes.add(base_path)
        print("Prefixes: ", prefixes)
    except Exception:
        pass

    # From OAS servers
    servers = spec_obj.get("servers") or []
    for s in servers:
        url = s.get("url") if isinstance(s, dict) else s
        if isinstance(url, str):
            try:
                sp = urlparse(url).path or ""
                sp = sp.rstrip("/")
                if sp and not sp.startswith("/"):
                    sp = "/" + sp
                if sp:
                    prefixes.add(sp)
            except Exception:
                pass

    ordered = sorted(prefixes, key=lambda p: len(p), reverse=True)
    return ordered + [""]

def _strip_prefix(path: str, prefixes: list[str]) -> str:
    """Strip the first matching prefix from the path."""
    p = path or ""
    if not p.startswith("/"):
        p = "/" + p
    for pref in prefixes:
        if pref and (p == pref or p.startswith(pref + "/")):
            return p[len(pref):] or "/"
    return p

def _normalize_hit_to_template(
    method: str,
    path: str,
    templates: set[tuple[str, str]],
    prefixes: list[str] | None = None
) -> tuple[str, str]:
    """
    Improved normalization:
    - Keeps exact path matches if they exist in the OpenAPI spec
    - Normalizes parameter naming (snake_case vs camelCase)
    - Applies prefix stripping (/api/v3 etc.)
    - Ignores query strings and fragments when matching (e.g. /pet/findByStatus?status=archived)
    - Never drops valid endpoints that actually exist in the spec
    """
    # Normalize method
    method = (method or "").upper().strip()

    # Defensive: handle None/empty and possible full URLs
    raw = (path or "").strip() or "/"

    # If a full URL is passed, extract just the path
    # (urlsplit works for both full URLs and bare paths)
    parts = urlsplit(raw)
    raw_path = parts.path or "/"

    # Strip query and fragment explicitly in case urlsplit wasn't used above
    # (safe even if there was no scheme)
    raw_path = raw_path.split("?", 1)[0].split("#", 1)[0]

    # Normalize slashes
    path = "/" + raw_path.lstrip("/")
    path = path.rstrip("/") or "/"

    prefixes = prefixes or [""]

    # 1️⃣ Strip known prefixes (e.g. /api/v3)
    for pref in prefixes:
        if not pref:
            continue
        # Normalize prefix format
        p = "/" + pref.lstrip("/")
        if path == p:
            path = "/"
            break
        if path.startswith(p + "/"):
            path = path[len(p):] or "/"
            break

    path = "/" + path.strip("/") if path != "/" else "/"

    # 2️⃣ Extract all templates for this method
    method_templates = [(m, t) for (m, t) in templates if (m or "").upper() == method]
    if not method_templates:
        # If no method-specific templates, fall back to all templates
        method_templates = list(templates)

    norm_path_no_trailing = path.rstrip("/")
    norm_path_lower = norm_path_no_trailing.lower()

    # 3️⃣ Direct exact match (case-sensitive, no trailing slash diff)
    for _, tpl in method_templates:
        if tpl.rstrip("/") == norm_path_no_trailing:
            return (method, tpl)

    # 4️⃣ Case-insensitive literal match
    for _, tpl in method_templates:
        if tpl.rstrip("/").lower() == norm_path_lower:
            return (method, tpl)

    # 5️⃣ Variable segment normalization (e.g. IDs or differently-cased vars)
    segs = [s for s in norm_path_no_trailing.strip("/").split("/") if s]
    for _, tpl in method_templates:
        tsegs = [s for s in tpl.rstrip("/").strip("/").split("/") if s]
        if len(segs) != len(tsegs):
            continue

        all_match = True
        for s, ts in zip(segs, tsegs):
            # Template variable: {id}, {petId}, etc.
            if ts.startswith("{") and ts.endswith("}"):
                # accept any segment here
                continue

            # Non-variable: must match literally
            if s != ts:
                all_match = False
                break

        if all_match:
            return (method, tpl)

    # 6️⃣ Fallback — no match found, return normalized method + path (don't drop)
    return (method, path)

def extract_ops_from_tests_code(code: str, base_url_var: str = "BASE_URL") -> set[tuple[str,str]]:
    """
    Heuristics to find (METHOD, path) pairs inside generated step code.

    It handles:
    - requests.<method>(BASE_URL + "/path" ...)
    - requests.request("<METHOD>", BASE_URL + "/path" ...)
    - f-strings where the static part is visible: f"{BASE_URL}/path"
    """
    ops = set()
    if not code:
        return ops

    meth_pat = r"(get|post|put|delete|patch|options|head)"
    method_union = "|".join(sorted(_HTTP_METHODS))
    p1 = re.compile(
        rf"requests\.{meth_pat}\s*\(\s*(?:{base_url_var}\s*\+\s*|f?['\"]\s*\{{\s*{base_url_var}\s*\}}\s*)?['\"](/[^'\"\\)]*)",
        re.IGNORECASE,
    )
    p2 = re.compile(
        rf"requests\.request\s*\(\s*['\"]({method_union})['\"]\s*,\s*(?:{base_url_var}\s*\+\s*|f?['\"]\s*\{{\s*{base_url_var}\s*\}}\s*)?['\"](/[^'\"\\)]*)",
        re.IGNORECASE,
    )
    p3 = re.compile(
        rf"requests\.{meth_pat}\s*\(\s*f['\"]\{{\s*{base_url_var}\s*\}}(/[^'\"\\)]*)",
        re.IGNORECASE,
    )

    for m in p1.finditer(code):
        method = m.group(1).upper()
        path   = m.group(2)
        ops.add((method, path))
    for m in p2.finditer(code):
        method = m.group(1).upper()
        path   = m.group(2)
        ops.add((method, path))
    for m in p3.finditer(code):
        method = m.group(1).upper()
        path   = m.group(2)
        ops.add((method, path))

    return ops

# ----------------- Request accounting (parent process) -----------------
st.session_state.setdefault("request_count", 0)
st.session_state.setdefault("covered_ops", set())
st.session_state.setdefault("loaded_run", None)
st.session_state.setdefault("loaded_iteration", None)
st.session_state.setdefault("loaded_summary", None)

@contextmanager
def _request_logging():
    """Temporarily wrap requests.request in the Streamlit process for request accounting."""
    original_request = requests.request

    def _logged_request(method, url, *args, **kwargs):
        st.session_state["request_count"] += 1
        try:
            path = urlparse(url).path
            st.session_state["covered_ops"].add((method.upper(), path))
        except Exception:
            pass
        return original_request(method, url, *args, **kwargs)

    requests.request = _logged_request
    try:
        yield
    finally:
        requests.request = original_request

# ----------------- sitecustomize injector for child (Behave) process -----------------
def _write_sitecustomize(tmpdir: str):
    # FIX: add child request counting + continuous flush
    code = r'''
import os, json, atexit, threading
from urllib.parse import urlparse

COV_FILE = os.environ.get("COVERED_OPS_FILE")
REQ_FILE = os.environ.get("REQUEST_COUNT_FILE")
_seen, _lock = set(), threading.Lock()
_count = 0

def _flush_files():
    try:
        if COV_FILE:
            with open(COV_FILE, "w", encoding="utf-8") as f:
                json.dump(list(_seen), f)
        if REQ_FILE:
            with open(REQ_FILE, "w", encoding="utf-8") as f:
                f.write(str(_count))
    except Exception:
        pass

def _record_hit(method, url):
    global _count
    try:
        path = urlparse(url).path
        with _lock:
            _count += 1
            _seen.add((str(method).upper(), path))
        _flush_files()
    except Exception:
        pass

try:
    import requests

    _orig_api_request = getattr(requests, "request", None)
    _orig_session_request = getattr(requests.sessions.Session, "request", None)
    _orig_session_send = getattr(requests.sessions.Session, "send", None)

    def _wrapped_api_request(method, url, *args, **kwargs):
        _record_hit(method, url)
        return _orig_api_request(method, url, *args, **kwargs)

    def _wrapped_session_request(self, method, url, *args, **kwargs):
        _record_hit(method, url)
        return _orig_session_request(self, method, url, *args, **kwargs)

    def _wrapped_session_send(self, request, *args, **kwargs):
        try:
            _record_hit(request.method, request.url)
        except Exception:
            pass
        return _orig_session_send(self, request, *args, **kwargs)

    if _orig_api_request:
        requests.request = _wrapped_api_request
    if _orig_session_request:
        requests.sessions.Session.request = _wrapped_session_request
    if _orig_session_send:
        requests.sessions.Session.send = _wrapped_session_send

except Exception:
    pass

@atexit.register
def _dump():
    try:
        _flush_files()
    except Exception:
        pass
'''
    with open(os.path.join(tmpdir, "sitecustomize.py"), "w", encoding="utf-8") as f:
        f.write(code)

def run_subprocess(args, cwd=None, timeout=120, env=None):
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr

# ------------ Helpers: robust JSON parsing ------------
def _find_json_array(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    if candidate.count("[") >= candidate.count("]"):
        return candidate
    return None

def try_parse_json_list(obj) -> list:
    """Robustly convert GPT output (Python-literal or JSON-like) into a proper Python list."""
    # 1) Already a list
    if isinstance(obj, list):
        return obj

    # 2) Reject non-string, non-list types
    if obj is None or isinstance(obj, (dict, int, float, bool)):
        return []

    # Normalize to string
    s = str(obj).strip()
    if not s:
        return []

    # ---- STEP 1: Try JSON directly ----
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # If dict contains a list value, return the first list found
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except:
        pass

    # ---- STEP 2: Try Python literal syntax ----
    try:
        py = ast.literal_eval(s)
        if isinstance(py, list):
            return py
        if isinstance(py, dict):
            for v in py.values():
                if isinstance(v, list):
                    return v
    except:
        pass

    # ---- STEP 3: Extract any list-like substring ----
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]

        # Try JSON first
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

        # Try Python literal next
        try:
            py = ast.literal_eval(candidate)
            if isinstance(py, list):
                return py
        except:
            pass

        # ---- STEP 4: Try converting single quotes → double quotes safely ----
        cleaned = re.sub(r"'", '"', candidate)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

    # Nothing worked
    return []


def _coerce_mr_id(value) -> str:
    """Best effort to extract a stable MR identifier from mixed inputs."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = _MR_ID_PATTERN.search(text)
    if match:
        return match.group(1)
    return text

def list_case_study_runs(base_dir: str = "case_studies") -> list[Path]:
    """Return sorted list of existing execution folders."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    runs = [p for p in base_path.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs

def load_json_file(path: Path):
    """Safely load JSON file, returning None on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def extract_mr_ids(mr_list: list) -> list[str]:
    ids: list[str] = []
    for item in mr_list:
        candidate = ""
        if isinstance(item, dict):
            candidate = _coerce_mr_id(item.get("id"))
            if not candidate:
                candidate = _coerce_mr_id(item.get("scenario"))
        elif isinstance(item, (str, int, float)):
            candidate = _coerce_mr_id(item)

        if candidate:
            ids.append(candidate)

    seen = set()
    uniq: list[str] = []
    for mid in ids:
        if mid and mid not in seen:
            seen.add(mid)
            uniq.append(mid)
    return uniq

def strip_code_fences(code: str) -> str:
    if not code:
        return code
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z0-9]*\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
    return code

# ----------------- LLM -----------------
def is_reasoning_like(model: str) -> bool:
    m = model.lower()
    # GPT-5-Codex family
    if "gpt-5-codex" in m:
        return True
    # GPT-5 reasoning family (exclude explicit chat-latest variant)
    if "gpt-5" in m and "chat-latest" not in m:
        return True
    # o1-style reasoning models
    if "o1" in m:
        return True
    return False


def build_llm(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    temperature: float,
    seed: int | None
):
    """
    Robust LLM constructor with clean GPT-5 / reasoning model handling.
    """
    normalized = model.lower()

    # Detect reasoning / codex / structured models
    is_reasoning = (
        normalized.startswith("gpt-5")              # GPT-5, GPT-5.1, GPT-5-long, GPT-5-codex
        or "gpt-5-codex" in normalized              # explicit codex
        or normalized.startswith("o1")              # o1-mini, o1-preview
        or normalized.startswith("o3")              # newer reasoning line
    )

    kwargs = {"model": model}

    # ❌ Do NOT apply sampling to reasoning models
    if not is_reasoning:
        kwargs["temperature"] = temperature

    # Seed (only used if supported)
    if seed is not None:
        kwargs["seed"] = seed

    # -----------------------------
    # OpenAI provider
    # -----------------------------
    if provider == "OpenAI":
        kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
        kwargs["max_retries"] = 3

        # ✔ Enable reasoning effort only where valid
        if is_reasoning:
            kwargs["reasoning_effort"] = "high"

    # -----------------------------
    # Local (Ollama)
    # -----------------------------
    else:
        if not model.startswith("ollama/"):
            kwargs["model"] = "ollama/" + model

        kwargs["base_url"] = (base_url or "http://localhost:11434").rstrip("/")
        kwargs["api_key"] = ""   # Ollama does not use keys

    return LLM(**kwargs)


# ----------------- GPT-5 Refiner Agent -----------------

_refiner_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
REFINER_PROMPT = """
Just Review the Step Definitions code and then check the MR. Step code should be SAFE and RISK FREE to prevent TypeError, ValueError.
MUST REVIEW THE API.. Usually endpoints in Given When Then should be same like for filtering or search APIs etc..
Then check the openapi specification. Dont change anything if everything is correct.
Otherwise fix the Endpoints by reviewing the open api specification to make sure that the step definitions code correctly implements the Metamorphic Relation (MR) against the provided OpenAPI specification.
MUST VERIFY: 
I dont want to get any TypeError here or any other error. ALways review the Open API Specication such as summary and description of each API operation to make sure that the step definitions code is using correct endpoints and methods to implement the MR correctly.
Moroever, For literal API paths, always escape braces: e.g. /pet/\{petId\}.
Rule: escaped braces for paths, normal braces only for actual parameters
Important: MUST review the API and change it by checking the keywords in MR that might match with the Open API summary or description if match then use that path and method and not any other similar ones.
MOST IMPORTANT: Always check for 500 errors after every endpoint call and MUST RAISE THAT ERROR in behave Report and mark that failed if any 500 error is possible in the step definitions code.
Expected Output shold be ONLY the REFINED Python code with Behave step definitions.




"""

def refine_with_gpt5(step_code: str, mr_text: str, oas_text: str, base_url_api: str,behave="") -> str:
    print("IN THE REFINE FUNCTION")

    prompt = f"""
            {REFINER_PROMPT}
            Must use this as BASE_URL:
            {base_url_api}

            OpenAPI:
            {oas_text}

            MR:
            {mr_text}

            Generated Code:
            {step_code}

            """


    resp = _refiner_client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": REFINER_PROMPT},
            {"role": "user", "content": prompt}
        ],
        reasoning={"effort": "none"}
    )

    # NEW API — RETURN STRING IS HERE:
    out = resp.output_text

    print("Refine output:\n", out)

    return out.strip()



MR_REFINER_PROMPT = """
Metamorphic Testing is a type of Property-based testing that tests SUT by making relations between inputs and outputs.
You are the metamorphic relations Refinement Agent. Your job is to just review the Open API Specification and see if the list of metamorphic relations (MRs) provided are in line with the API functionality as described in the OpenAPI spec.
If some metamorphic relations (MRs) do not align with the API functionality, you must refine them to ensure they accurately reflect the capabilities and constraints of the API.
Do not change the structure of the MR, only refine the content to better match the API spec.
Resend me the updated without changing the format for the MR but do not mention endpoint names..
--
If an MR can't be refined then DROP it.
Expected Output: Same format of the MR as are receiving below.
"""

_mr_refiner_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def refine_mr_with_gpt5(raw_mr_obj: dict, oas_text: str, pending_ops: list=None) -> dict | None:
    prompt = f"""
Must check summary and description and in sync with OpenAPI Specification:
{oas_text}

Raw Metamorphic relations (MR) list containing objects of MRs:
{json.dumps(raw_mr_obj, indent=2)}
"""

    resp1 = _mr_refiner_client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": MR_REFINER_PROMPT},
            {"role": "user", "content": prompt}
        ],
        reasoning={"effort": "low"}
    )

    print("MR RESP: ", resp1)

    out = resp1.output_text

    print("OUTPUT TEXT: ", out)
    return out





# ----------------- Agent Fabric -----------------
def make_agents(mr_llm: LLM, code_llm: LLM):
    mr_generator = Agent(
       role = "MR Generator",
        goal = (
            "Get the functional or system requirements from the prompt and, generate up to 5 unique, property-based "
            "Metamorphic Relations (MRs) using the Given/When/Then structure. "
            "Each MR must define a clear, testable transformation between a seed input/output pair "
            "and a follow-up input/output pair. "
            "The 'when' clause must explicitly describe how the follow-up input is derived from the seed input, "
            "and the 'then' clause must define the precise relationship between the FOLLOW-UP OUTPUT and the SEED OUTPUT "
            "(e.g., identical, superset, subset, proportional, differs only in updated fields, etc.)."
        ),
        backstory = (
            "Metamorphic testing verifies program consistency under input transformations when no ground truth output is known. "
            "A metamorphic relation defines a predictable link between multiple executions of a program — "
            "the seed input/output pair and the follow-up input/output pair. "
            "These relations act as behavioral properties or invariants that must always hold. "
            "You derive follow-up inputs from seed inputs via transformations and check whether "
            "the corresponding follow-up outputs relate correctly to the seed outputs. "
            "Use only the provided prompt as the information source, "
            "avoid endpoint or implementation details, and express the MRs in a clear, formal, and testable manner."
        ),
        verbose = True,
        llm=mr_llm,
    )

    # ⚡️ IMPROVED prompt: Behave step definitions per MR
    step_generator = Agent(
        role="Behave Step Code Generator",
        goal=(
            "Generate and then Verify EXECUTABLE Risk-Free, Syntax Error-Free Python code with Behave step definitions for a SINGLE MR against a REST API, "
            "using the  provided OpenAPI spec for paths/methods/schemas and the provided BASE_URL."
        ),
        backstory=(
            "You have expertise in generating robust Behave step definitions of REST APIs. "
            "You convert Gherkin-like MR text into real @given/@when/@then functions. "
            "Use requests; store seed/follow-up responses in 'context'; assert metamorphic relation precisely. "
            "Be robust to missing data by creating/cleaning resources when needed."
            " Do not retrieve a specific resource in GIVEN without making sure that it exists first place so RISK-FREE execution is guaranteed."
        ),
        verbose=True,
        llm=code_llm
    )

    
    return mr_generator, step_generator


    

# ----------------- Tasks -----------------
def make_tasks(mr_generator, step_generator):
    # MR generator 
    task_mr_gen = Task(
       description = (
            "You are a Metamorphic Relation (MR) Generator. "
            "Metamorphic testing verifies relationships between multiple inputs and their corresponding outputs (called Metamorphic Relations), "
            "instead of checking correctness for individual input–output pairs. "
            "Each MR expresses a specific property that must hold between a seed input/output and a derived (follow-up) input/output pair. "

            "Examples of common metamorphic properties include: "
            "Idempotence (repeating an operation should not change the result), "
            "Commutativity (the order of independent inputs should not affect the result), "
            "Monotonicity (increasing input values should not decrease output values), "
            "Invariance (adding irrelevant data should not alter the output), "
            "and Proportionality (scaling inputs should cause proportional scaling in outputs). "

            "Get the functional requirements from the specification useful for generating MRs: {oas}. "
            "Avoid duplicates with existing relations: {previous_mrs}. "
            "Generate up to 5 NEW, UNIQUE, and PROPERTY-BASED Metamorphic Relations (MRs) that define clear, measurable, and testable transformations. "

            "Each MR must follow this structure and explicitly reference both seed and follow-up inputs and outputs. "
            "Return output strictly as a JSON array only. "

            "Check if previous api operation list is empty or not: {api_operations}. "
            "If not empty then, use this with along to map with api documentation to generate Metamorphic Relations."
            "AND FOCUS on PROPERTY-BASED relations that define clear, measurable, and testable transformations. "
            "MAKE SURE TO review the summary and description of each of the API operations provided in the OpenAPI spec to better understand their functionality and behavior. "
            "If previous api operation not empty and have ONE API operation (method,path) is left in here: {api_operations}, then create at least one MR that directly maps and can only be done through that operation and not any in other operations in open api specification. "
            "MAKE SURE TO REVIEW THE SUMMARY AND DESCRIPTION OF EACH OF THE API OPERATIONS PROVIDED IN THE OPENAPI SPEC TO BETTER UNDERSTAND THEIR FUNCTIONALITY AND BEHAVIOR FOR GENERATING UNIQUE MRs. "
            "Sometimes, Open API spec, contains similar APIs. Must look at the Summary and Description carefully and generate unique MRs so that later when it is needed to convert into code then only api operations {api_operations} to be used."


           "{ "\
            "'id': 'MR1', "\
            "'scenario': 'Scenario: <short, property-based description of the metamorphic relation>', "\
            "'given': 'a seed input <describe the valid baseline input or current state> producing a seed output <describe what the output represents> for MR1.', "\
            "'when': 'a follow-up input is created by <describe the transformation or change applied to the seed input by changing the state>, yielding a follow-up output for MR1.', "\
            "'then': '<transformation relation → followup output → seed output>: <define the precise relationship, e.g., remain identical, be a superset, differ only in updated fields, scale proportionally, etc.> for MR1.' "\
            "} "

            "Rules: "
            "1. The WHEN clause must explicitly describe how the follow-up input is derived from the SEED INPUT and mention both. "
            "2. The THEN clause must clearly specify the relationship between the FOLLOW-UP OUTPUT and the SEED OUTPUT using comparative, testable language. "
            "3. Each Given clause must be unique in context and wording. "
            "4. Each MR must define a deterministic, property-based behavior (not vague consistency). "
            "5. Use precise verbs like  'returns', 'includes', or 'contains' instead of vague ones like 'shows' or 'displays'. "
            "6. Keep endpoint-agnostic and domain-neutral. "
            "7. If fewer than 5 unique MRs exist, return only the valid ones; if none, return []. "
            "8. Output strictly in JSON array format with no extra text or markdown. "
            "9. Concise Given When Then statements may use property formulas if needed or not too long characters. "
            "10. **MUST INCLUDE Seed input and Seed output keywords in Given, FOLLOW-UP INPUT and FOLLOW-UP OUTPUT in WHEN and THEN must conclude both Follow up and Seed output keywords   respectively.** "
            "11. You are also provided with all API operations (method, path) that have been NOT BEEN TESTED so far: {api_operations}. "
            "Use this information to CREATE NEW MRs that restate or directly map to these same API operations. "
           
            "Example: "
            "[ "
            "{ "
            "'id': 'MR1', "
            "'scenario': 'Scenario: MR1 Updating an entity should modify only the updated field', "
            "'given': 'a seed input that retrieves the list of pets, producing a seed output with a count of pets for MR1.', "
            "'when': 'a follow-up input is derived from the seed input by modifying one attribute of the entity, producing a follow-up output for MR1.', "
            "'then': 'a follow-up output changes only the modified attribute; all other fields match the seed output for MR1.' "
            "} "

            "Once the MRs are generated, verify if a particular MR is property based and have relation between SEED INPUT and FOLLOW-UP INPUT and SEED OUTPUT and FOLLOW-UP OUTPUT"
        ),
        expected_output=(
            "ONLY a JSON array of objects with fields: id, scenario, given, when, then; "
            "Then must correspond to the MR transformation. (transformation relation → followup output → seed output) "
            "unique vs {previous_mrs}; up to 5; [] if none."
        ),
        agent=mr_generator,
    )
    
    task_step_gen = Task(
    description=(
        "You are a **Behave Step Code Generator** for metamorphic testing of a REST API.\n\n"
        "You receive:\n"
        "- A SINGLE MR object (JSON-like) in endpoint-agnostic, Given/When/Then form: {mr}\n"
        "- The full OpenAPI specification (YAML/JSON): {oas}\n"
        "- The BASE_URL of the live API: {base_url}\n\n"
        "- Important: If Left with ONE API operation (method,path) in the api operations list {api_operations} that has NOT BEEN TESTED YET, then MUST use this to create the MR step code. \n"
        "- **If the MRs can be created using two similar APIs from Open API Spec... then MUST choose the one from the api operations list {api_operations} to create the MR step code**.\n\n"
        "Your job is to:\n"
        "1. Read the MR carefully and understand the **property** it specifies:\n"
        "   - What is the SEED INPUT / SEED OUTPUT?\n"
        "   - What is the FOLLOW-UP INPUT / FOLLOW-UP OUTPUT?\n"
        "   - What exact relationship should hold between SEED OUTPUT and FOLLOW-UP OUTPUT?\n"
        "2. Using ONLY the OpenAPI spec, choose appropriate endpoints, methods, and parameters\n"
        "   that can realize this MR.\n"
        "3. Generate Behave step definitions that:\n"
        "   - Bind EXACTLY to the Given/When/Then strings in the MR (full string match in decorators).\n"
        "   - Implement the metamorphic relation precisely and logically.\n"
        "   - Are fully executable Python code with no syntax errors.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Output ONLY Python code. No backticks, no markdown, no prose.\n"
        "- At the top include exactly:\n"
        "    from behave import given, when, then\n"
        "    import os, json, requests, time\n"
        "- Define:\n"
        "    BASE_URL = os.environ.get('BASE_URL', '{base_url}')\n"
        "- For EACH step, use decorators with **exact, full-string** match to the MR text:\n"
        "    @given('<MR.given text here>')\n"
        "    @when('<MR.when text here>')\n"
        "    @then('<MR.then text here>')\n"
        "  (Use the strings from the MR object exactly as they appear in the feature file.)\n"
        "- Every step function name MUST contain the MR id if present (e.g. _mr_MR1_... ).\n"
        "- Use `context` to share state:\n"
        "    context.seed_request, context.seed_response\n"
        "    context.followup_request, context.followup_response\n"
        "    context.misc for extra data\n"
        "- All HTTP calls MUST use `requests` with `timeout=10`.\n"
        "- Use only paths/methods/parameters defined in the OpenAPI spec:\n"
        "    - Infer list, read, create, update, filter operations from the spec.\n"
        "    - If the MR requires behavior that cannot be mapped to any valid endpoint,\n"
        "      implement the step so it raises `AssertionError` explaining the mismatch.\n"
        "- No duplicate imports, no duplicate BASE_URL definition, no duplicate function names.\n"
        "- On any unexpected exception inside a step:\n"
        "    - Print a clear error message (including MR id and step type).\n"
        "    - Re-raise or assert False so the test fails (do NOT silently ignore).\n\n"
        "METAMORPHIC LOGIC (THIS IS CRITICAL):\n"
        "You MUST implement what the MR actually describes, **not** a generic pattern.\n"
        "Use the MR text + OpenAPI spec as the joint source of truth.\n"
        "In particular:\n"
        "1. If the MR talks about filtering (e.g. status, type, category):\n"
        "   - Use the documented query parameters from the spec.\n"
        "   - Assert that EVERY item in FOLLOW-UP OUTPUT satisfies the filter condition.\n"
        "   - DO NOT assume subset/superset relationships between different filter values\n"
        "     unless that relationship is explicitly part of the MR AND consistent with the spec.\n"
        "   - Example of what you MUST NOT generate:\n"
        "       - Treating results for status='pending' as a subset of status='available'.\n"
        "2. If the MR describes idempotence (repeating same request or update):\n"
        "   - Call the same endpoint multiple times and assert responses/state are stable\n"
        "     according to the MR (e.g., same list, same resource fields).\n"
        "3. If the MR describes an update property:\n"
        "   - Create or fetch a valid resource using spec-compliant fields.\n"
        "   - Apply the update via the correct endpoint.\n"
        "   - Assert only the intended fields change; others remain equal.\n"
        "4. If the MR describes adding/removing elements (e.g. create then list):\n"
        "   - Assert the FOLLOW-UP OUTPUT reflects that change exactly as stated.\n"
        "5. When MR is high-level or endpoint-agnostic:\n"
        "   - Intelligently choose the most suitable endpoint(s) from the spec.\n"
        "   - Keep assertions minimal, direct, and faithful to the MR's property.\n\n"
        "EXAMPLE PATTERNS (for the model, not to hardcode):\n"
        "- Filter MR:\n"
        "    Given SEED INPUT gets all items\n"
        "    When FOLLOW-UP INPUT applies filter status='available'\n"
        "    Then assert: for each item in FOLLOW-UP OUTPUT, item['status'] == 'available'.\n"
        "- Idempotence MR:\n"
        "    When same request is repeated\n"
        "    Then assert: FOLLOW-UP OUTPUT == SEED OUTPUT (or matches property defined in MR).\n"
        "- Update MR:\n"
        "    Given SEED OUTPUT is the original resource\n"
        "    When FOLLOW-UP INPUT updates one field\n"
        "    Then assert: only that field differs between FOLLOW-UP OUTPUT and SEED OUTPUT.\n\n"
        "NEVER:\n"
        "- Never invent relationships like 'results of filter A are a subset of filter B'\n"
        "  unless explicitly required by the MR **and** valid per OpenAPI.\n"
        "- Never ignore the MR text: the MR is the oracle. Your code must encode its logic.\n"

        "**ALWAYS**:\n"
        "Use RISK-FREE CODING practices:\n\n"
        "Defensive (safe against missing keys or variable response formats)\n\n"
        "Type-checked (isinstance for list/dict)\n\n"
        "Using .get() instead of unsafe key access\n"
        "Wrapped in try/except for all requests\n"
        "Logging detailed errors (HTTP, JSON, assertion)\n"
        "Logging detailed errors (HTTP, JSON, assertion)\n"


        "**VERIFICATION OF THE STEP CODE**:\n"
        "- After generating the code, re-parse it to ensure no syntax errors.\n"
        "- Fix any JSON Decode errors, syntax errors, logical errors, or runtime issues in the step definitions.\n"
        "- Ensure that the step definitions:\n"
        "- Code must be executable Python code with no syntax errors.\n\n"
        "- All HTTP calls MUST use `requests` with `timeout=10`.\n"
        "- You know about REST APIs,.. so FIX any issue related to params or other that you face.  \n"
        "- Always and ONLY take help from Open API Specification {oas} to verify the **CODE**, attributes, modal, paths, methods, Info (metadata), External Documentation, Servers, Tags, Paths, Operations, Parameters, Request Bodies, Responses, Schemas (data models), Security Schemes, OAuth2 Scopes, Examples, Enums, Error Descriptions, Media Types (content negotiation), Data Serialization Rules (formats, XML wrapping), Custom Extensions (x- fields), Authorization Details, Validation Rules (required fields, types), and Vendor-specific Metadata (x-swagger-router-controller, x-swagger-router-model).etc.\n"
        "- If test code and OpenAPI Spec conflict with each other, then FIX the test code to make it aligned with OpenAPI Spec.\n"
        "- If a test code using a get method of a path that does not exist in OpenAPI Spec, then FIX it to use the correct path and method from OpenAPI Spec.\n"
        "- If the given of the test code is retrieving a resource that doesn't map to given OpenAPI spec example user then replace it with Open API spec resources."
        "- Always make sure that when retrieving a resource in GIVEN step, that resource must exist first through a prior creation step in WHEN step.\n"
    ),
    expected_output=(
        "ONLY valid Risk-Free, Syntax Error-Free, Python code with Behave step definitions implementing the MR's property, "
        "bound to the MR's Given/When/Then strings, using {base_url} and {oas}."
    ),
    agent=step_generator,
    )

 

    return task_mr_gen, task_step_gen

# ----------------- Crew runner with log capture -----------------
def kickoff_with_logs(agents, tasks, inputs):
    f_out, f_err = io.StringIO(), io.StringIO()
    with redirect_stdout(f_out), redirect_stderr(f_err):
        crew = Crew(agents=agents, tasks=tasks, verbose=True)
        result = crew.kickoff(inputs=inputs)
    # If you want logs, stitch f_out/f_err.
    return result, None

# ----------------- AST Repair for step code -----------------
def _ast_ok(py_code: str) -> tuple[bool, str | None]:
    try:
        ast.parse(py_code)
        return True, None
    except SyntaxError as e:
        return False, f"AST SyntaxError: {e}"

def generate_step_code_for_mr_with_repair(step_agent, task_step_code, mr, oas_text, base_url: str, max_rounds: int = 2, api_operations: list[str] = []) -> tuple[str, list[tuple[str,str|None]]]:
    """
    Calls the step generator for ONE MR and repairs syntax via AST loop.
    Returns (code, logs).
    """
    attempt_logs: list[tuple[str,str|None]] = []
    
    mr_id = str(mr.get("id", "MRX"))




    raw, _ = kickoff_with_logs([step_agent], [task_step_code], inputs={"base_url": base_url, "mr": str(mr), "oas": oas_text,"mr_id": mr_id, "api_operations": api_operations})
    code = strip_code_fences(str(raw))
    ok, err = _ast_ok(code)
    attempt_logs.append(("gen", "Initial generation"))

    # Try to fix AST issues (light self-heal)
    rounds = 0
    while not ok and rounds < max_rounds:
        code_for_prompt = _escape_braces(code)
        err_for_prompt = _escape_braces(err or "")
        code_for_prompt = re.sub(r"\{([A-Z_]+)\}", r"{{\1}}", code_for_prompt)

        fix_task = Task(
            description=(
                "Fix syntax errors, TypeError, ValueError in the following Behave step code. "
                "Keep the SAME decorators (@given/@when/@then) EXACTLY as provided; "
                "preserve BASE_URL, imports, and assertions.\n\n"
                f"--- CODE START ---\n{code_for_prompt}\n--- CODE END ---\n\n"
                f"Error: {err_for_prompt}\n\n"
                "Return ONLY corrected Python code."
            ),
            expected_output="ONLY Python code.",
            agent=step_agent,
        )
        fixed, _ = kickoff_with_logs([step_agent], [fix_task], inputs={"base_url": base_url, "mr": str(mr), "oas": oas_text,"mr_id": mr_id})
        code = strip_code_fences(str(fixed))
        ok, err = _ast_ok(code)
        attempt_logs.append(("fix", err or "Applied syntax fix"))
        rounds += 1

    if not ok:
        # As a last resort, wrap in a minimal, valid module to avoid run-time import explosions.
        safe_wrapper = (
            "from behave import given, when, then\n"
            "import os, requests, json, time\n"
            f'BASE_URL = os.environ.get("BASE_URL", "{base_url}")\n'
            f"# MR {mr_id}: fallback no-op to keep Behave collection stable\n"
            "@given(\"seed input produces seed output.\")\n"
            "def _noop_given(context):\n"
            "    context.seed = {}\n"
            "@when(\"follow-up input is derived from seed input by applying a transformation.\")\n"
            "def _noop_when(context):\n"
            "    context.followup = {}\n"
            "@then(\"follow-up output should relate predictably to the seed output.\")\n"
            "def _noop_then(context):\n"
            "    assert True\n"
        )
        code = safe_wrapper
        attempt_logs.append(("fallback", "Inserted no-op steps to keep run stable."))

    return code, attempt_logs

# ----------------- Feature + Steps (per iteration) -----------------
def write_iteration_feature(mrs_list: list, iter_dir: Path, iteration_idx: int) -> tuple[Path, dict]:
    features_dir = iter_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"Feature: Iteration {iteration_idx} MRs", ""]
    scenario_map: dict[str, str] = {}
    for mr in mrs_list:
        scenario = mr.get("scenario", f"Scenario: {mr.get('id','MRX')}")
        given = mr.get("given", "")
        when = mr.get("when", "")
        then = mr.get("then", "")
        lines.append(f"  {scenario}")
        lines.append(f"    Given {given}")
        lines.append(f"    When {when}")
        lines.append(f"    Then {then}")
        lines.append("")

        mr_id = _coerce_mr_id(mr.get("id") or scenario)
        scenario_map[scenario] = mr_id
        scenario_map.setdefault(scenario.replace("Scenario:", "", 1).strip(), mr_id)

    fpath = features_dir / f"mrs_iteration_{iteration_idx}.feature"
    fpath.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    # Persist scenario-to-MR map for later lookups/loaders
    (iter_dir / "scenario_map.json").write_text(json.dumps(scenario_map, indent=2), encoding="utf-8")

    return features_dir, scenario_map

def write_steps_py(step_modules: list[str], features_dir: Path) -> Path:
    steps_dir = features_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    (steps_dir / "__init__.py").write_text("", encoding="utf-8")
    steps_code = "\n\n# ----\n\n".join(step_modules)

    ##Clean Step code
    steps_code= clean_step_code(steps_code)
    steps_path = steps_dir / "steps.py"
    steps_path.write_text(steps_code, encoding="utf-8")
    return steps_path

# ----------------- Behave run + result parsing + coverage -----------------
def run_behave_and_collect(features_dir: Path, base_url: str,  t: dict, iter_dir: Path, scenario_map: dict | None = None):
    """
    Execute behave and return structured results + coverage.
    Always returns a dict, even if execution or parsing fails.
    """
    report_path = iter_dir / "behave_report.json"
    covered_file = iter_dir / "covered_ops.json"
    # child request count file
    req_count_file = iter_dir / "request_count.txt"

    original_cwd = os.getcwd()
    try:
        # Prepare sitecustomize for child process coverage capture
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sitecustomize(tmpdir)
            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")
            env["BASE_URL"] = base_url
            env["COVERED_OPS_FILE"] = str(covered_file)
            env["REQUEST_COUNT_FILE"] = str(req_count_file)

            # Initialize request counter so first read is safe
            try:
                req_count_file.write_text("0", encoding="utf-8")
            except Exception:
                pass

            # Step 1: Change to the behave directory
            os.chdir(iter_dir)

            cmd = [
                "behave",
                "-f", "json",
                "-o", "behave_report.json",
            ]
            rc, out, err = run_subprocess(cmd, timeout=600, env=env)
            os.chdir(original_cwd)
    except Exception as e:
        # Fail-safe: return error payload
        safe_count = 0
        try:
            if req_count_file.exists():
                safe_count = int((req_count_file.read_text(encoding="utf-8").strip() or "0"))
        except Exception:
            safe_count = 0
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Behave execution failed: {e}",
            "passed": [],
            "failed": [],
            "covered_ops": [],
            "coverage_pct": 0.0,
            "total_ops": 0,
            "request_count": safe_count,
            "error": str(e),
        }

    # --- Parse behave JSON report ---
    behave_json = []
    try:
        if report_path.exists():
            behave_json = json.loads(report_path.read_text(encoding="utf-8") or "[]")
    except Exception as e:
        safe_count = 0
        try:
            if req_count_file.exists():
                safe_count = int((req_count_file.read_text(encoding="utf-8").strip() or "0"))
        except Exception:
            safe_count = 0
        return {
            "exit_code": rc,
            "stdout": out,
            "stderr": err,
            "passed": [],
            "failed": [],
            "covered_ops": [],
            "coverage_pct": 0.0,
            "total_ops": 0,
            "request_count": safe_count,
            "error": f"Failed to parse behave JSON: {e}",
        }
    
    # --- Collect pass/fail rows ---
    passed_rows, failed_rows = [], []
    try:
        for feature in behave_json:
            for scenario in feature.get("elements", []):
                scen_name = scenario.get("name", "")
                m = re.search(r"\b(MR[A-Za-z0-9_-]+)\b", scen_name)
                default_id = _coerce_mr_id(m.group(1) if m else scen_name)
                mr_id = default_id
                if scenario_map:
                    mr_id = scenario_map.get(scen_name, mr_id)
                    if mr_id == default_id:
                        mr_id = scenario_map.get(scen_name.strip(), mr_id)
                mr_id = _coerce_mr_id(mr_id)
                status = "passed"
                details = ""
                for step in scenario.get("steps", []):
                    res = step.get("result", {}) or {}
                    st_status = str(res.get("status", ""))
                    if st_status not in ("passed", "skipped"):
                        status = "failed"
                        details = (str(res.get("error_message")) or "").strip()
                        break
                row = {
                    "mr_id": mr_id,
                    "scenario": scen_name,
                    "status": status,
                    "details": details,
                }
                (passed_rows if status == "passed" else failed_rows).append(row)
    except Exception as e:
        safe_count = 0
        try:
            if req_count_file.exists():
                safe_count = int((req_count_file.read_text(encoding="utf-8").strip() or "0"))
        except Exception:
            safe_count = 0
        return {
            "exit_code": rc,
            "stdout": out,
            "stderr": err,
            "passed": [],
            "failed": [],
            "covered_ops": [],
            "coverage_pct": 0.0,
            "total_ops": 0,
            "request_count": safe_count,
            "error": f"Failed to parse scenarios: {e}",
        }

    # --- Coverage (real, spec-mapped ops only) ---
    covered_hits = set()
    try:
        if covered_file.exists():
            covered_hits = {tuple(x) for x in json.loads(covered_file.read_text(encoding="utf-8"))}
    except Exception:
        covered_hits = set()

    templates = _template_paths_from_spec(spec_obj)
    prefixes = _extract_path_prefixes(spec_obj, base_url)

    # Normalize runtime hits and split mapped/unmapped
    norm_hits = {_normalize_hit_to_template(m, p, templates, prefixes) for (m, p) in covered_hits}
    mapped_hits = {h for h in norm_hits if h in templates}
    unmapped_hits = {h for h in norm_hits if h not in templates}

    total_ops = len(templates)
    op_cov = (len(mapped_hits) / max(total_ops, 1)) * 100.0

    # Persist iteration artifacts
    (iter_dir / "covered_ops_mapped.json").write_text(
        json.dumps([[m, p] for (m, p) in sorted(mapped_hits)], indent=2), encoding="utf-8"
    )
    (iter_dir / "covered_ops_unmapped.json").write_text(
        json.dumps([[m, p] for (m, p) in sorted(unmapped_hits)], indent=2), encoding="utf-8"
    )
    (iter_dir / "coverage_metrics.json").write_text(
        json.dumps({"total_ops": total_ops,
                    "covered_unique": len(mapped_hits),
                    "coverage_pct": op_cov}, indent=2),
        encoding="utf-8"
    )

    # Safe request_count read (normal path)
    request_count = 0
    try:
        if req_count_file.exists():
            request_count = int((req_count_file.read_text(encoding="utf-8").strip() or "0"))
    except Exception:
        request_count = 0
    (iter_dir / "request_stats.json").write_text(
        json.dumps({"requests_child": request_count}, indent=2), encoding="utf-8"
    )

    # Return only real, mapped operations
    return {
        "exit_code": rc,
        "stdout": out,
        "stderr": err,
        "passed": passed_rows,
        "failed": failed_rows,
        "covered_ops": sorted(list(mapped_hits)),   # mapped only
        "coverage_pct": op_cov,
        "total_ops": total_ops,
        "request_count": request_count,
    }

# ----------------- Schemathesis baseline (optional) -----------------
def has_module(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False
    


def run_schemathesis(spec_text: str, base_url: str, checks: str = "all"):
    if not has_module("schemathesis"):
        return {"error": "schemathesis not installed. pip install schemathesis[cli]"}
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "openapi.yaml")
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(spec_text)
        args = [
            "schemathesis", "run",
            "--checks", checks,
            "--experimental-stateful=links",
            spec_path,
            "--base-url", base_url,
        ]
        rc, out, err = run_subprocess(args, cwd=tmpdir, timeout=600)
        return {"exit_code": rc, "stdout": out, "stderr": err}

# ----------------- UI -----------------
st.set_page_config(page_title="Metamorphic Agentic MR", layout="wide")
if "steps_code" not in st.session_state:
    st.session_state["steps_code"] = ""
if "agg_summary" not in st.session_state:
    st.session_state["agg_summary"] = {
        "mr_ids_all": [],
        "mr_ids_passed": [],
        "mr_ids_failed": [],
        "server_crash_count": 0
    }

st.title("ARMeta: Agentic Metamorphic Tester for REST APIs ")

with st.sidebar:
    st.header("🔌 LLM Settings")

    st.markdown("### 🧠 MR Generator Model")
    mr_mode = st.radio("Backend (MR)", ["API (OpenAI)", "Local (Ollama)"], horizontal=True, key="mr_mode")

    if mr_mode == "API (OpenAI)":
        mr_provider = "OpenAI"
        mr_model = st.text_input("Model (MR)", value="gpt-4o", key="mr_model")
        mr_api_key = st.text_input("API Key (MR)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        mr_base_url = None
    else:
        mr_provider = "Local"
        mr_base_url = st.text_input("Ollama Base URL (MR)", value="http://localhost:11434", key="mr_base")
        mr_api_key = ""
        try:
            resp = requests.get(f"{mr_base_url}/api/tags", timeout=3)
            if resp.ok:
                mr_models = [m["name"] for m in resp.json().get("models", [])]
                mr_model = st.selectbox("Local Model (MR)", mr_models or [""])
            else:
                mr_model = ""
                st.error("⚠️ No models found for MR generator.")
        except Exception:
            mr_model = ""
            st.error("❌ Could not connect to Ollama for MR generator.")

    mr_temp = st.slider("Temperature (MR)", 0.0, 1.0, 0.0, 0.05, key="mr_temp")

    st.divider()

    st.markdown("### 💻 Step Code Generator Model")
    step_mode = st.radio("Backend (Code)", ["API (OpenAI)", "Local (Ollama)"], horizontal=True, key="step_mode")

    if step_mode == "API (OpenAI)":
        step_provider = "OpenAI"
        step_model = st.text_input("Model (Code)", value="gpt-4o-mini", key="step_model")
        step_api_key = st.text_input("API Key (Code)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        step_base_url = None
    else:
        step_provider = "Local"
        step_base_url = st.text_input("Ollama Base URL (Code)", value="http://localhost:11434", key="step_base")
        step_api_key = ""
        try:
            resp = requests.get(f"{step_base_url}/api/tags", timeout=3)
            if resp.ok:
                step_models = [m["name"] for m in resp.json().get("models", [])]
                step_model = st.selectbox("Local Model (Code)", step_models or [""])
            else:
                step_model = ""
                st.error("⚠️ No models found for Step generator.")
        except Exception:
            step_model = ""
            st.error("❌ Could not connect to Ollama for Step generator.")

    step_temp = st.slider("Temperature (Code)", 0.0, 1.0, 0.0, 0.05, key="step_temp")

    st.divider()
    st.header("🎯 Stop Criteria")
    target_op_cov = st.slider("Target operation coverage %", 0, 100, 100)
    plateau_window = st.number_input("Plateau window (iterations)", 1, 50, 5)
    max_requests = st.number_input("Max requests", 100, 100000, 1000)
    max_minutes = st.number_input("Max minutes", 1, 240, 30)

    st.sidebar.divider()
    st.sidebar.markdown("### 🧩 Active Config")
    st.sidebar.caption(f"🧠 MR Model: {mr_model} ({mr_provider})")
    st.sidebar.caption(f"💻 Step Model: {step_model} ({step_provider})")

# 1) OpenAPI Specification
st.subheader("1) OpenAPI Specification")
oas_input_mode = st.radio("Provide OpenAPI via", ["Upload file", "Paste text"], horizontal=True, key="oas_mode")
oas_text = ""
if oas_input_mode == "Upload file":
    uploaded_oas = st.file_uploader("Upload OpenAPI (JSON/YAML)", type=None, key="oas_file")
    if uploaded_oas:
        oas_text = uploaded_oas.read().decode("utf-8", errors="ignore")
else:
    oas_text = st.text_area("Paste OpenAPI spec", height=240, placeholder="Paste OAS3 JSON or YAML...")

st.subheader("3) API Base URL")
base_url_api = st.text_input("Base URL of live API", value="http://localhost:8080/api/v3")

col_a, col_b, col_c = st.columns(3)
run_btn = col_a.button("▶️ Run MR Pipeline", use_container_width=True)
# baseline_btn = col_c.button("🧪 Schemathesis Baseline", use_container_width=True)
clear_btn = col_c.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state.clear()
    st.rerun()

# Load existing execution folders
st.subheader("Load Existing Execution")
stored_runs = list_case_study_runs()
default_run_option = "— Select stored run —"
run_options = [default_run_option] + [str(p) for p in stored_runs]
selected_run_option = st.selectbox("Saved runs (from case_studies/)", run_options, key="load_run_select")
custom_run_input = st.text_input("Or enter full/relative path to execution folder", key="load_run_input")
load_col1, load_col2 = st.columns(2)
load_btn = load_col1.button("📂 Load Execution", use_container_width=True, key="load_run_button")
clear_loaded_btn = load_col2.button("Clear Loaded Execution", use_container_width=True, key="clear_loaded_button")

if load_btn:
    candidate = (custom_run_input or (selected_run_option if selected_run_option != default_run_option else "")).strip()
    if not candidate:
        st.warning("Select or enter an execution folder to load.")
    else:
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = Path(candidate)
            if not candidate_path.exists():
                candidate_path = Path("case_studies") / candidate
        try:
            resolved_path = candidate_path.resolve()
        except Exception:
            resolved_path = candidate_path
        if resolved_path.exists() and resolved_path.is_dir():
            st.session_state["loaded_run"] = str(resolved_path)
            st.session_state["loaded_iteration"] = None
            st.session_state["loaded_summary"] = load_json_file(resolved_path / "run_summary.json") or {}
            st.success(f"Loaded execution folder: {resolved_path}")
        else:
            st.error(f"Execution folder not found: {resolved_path}")

if clear_loaded_btn:
    st.session_state["loaded_run"] = None
    st.session_state["loaded_iteration"] = None
    st.session_state["loaded_summary"] = None

loaded_run_path = st.session_state.get("loaded_run")
if loaded_run_path:
    run_path = Path(loaded_run_path)
    st.subheader("📂 Loaded Execution Overview")
    st.caption(str(run_path))

    loaded_summary = st.session_state.get("loaded_summary") or load_json_file(run_path / "run_summary.json")
    if loaded_summary:
        summary_rows = [
            {"Metric": "Total MRs generated", "Count": len(loaded_summary.get("mr_ids_all", [])), "MR IDs": ", ".join(loaded_summary.get("mr_ids_all", [])) or "—"},
            {"Metric": "Passed", "Count": len(loaded_summary.get("mr_ids_passed", [])), "MR IDs": ", ".join(loaded_summary.get("mr_ids_passed", [])) or "—"},
            {"Metric": "Failed", "Count": len(loaded_summary.get("mr_ids_failed", [])), "MR IDs": ", ".join(loaded_summary.get("mr_ids_failed", [])) or "—"},
            {"Metric": "Not executed / skipped", "Count": len(loaded_summary.get("mr_ids_unexecuted", [])), "MR IDs": ", ".join(loaded_summary.get("mr_ids_unexecuted", [])) or "—"},
            {"Metric": "Fault Detection Rate", "Count": f"{loaded_summary.get('fault_detection_rate_pct', 0):.2f}%", "MR IDs": "—"},
            {"Metric": "Total API operations", "Count": loaded_summary.get("total_api_ops", '—'), "MR IDs": "—"},
        ]
        st.table(summary_rows)

    cov_data = load_json_file(run_path / "cumulative_coverage.json")
    if cov_data:
        st.caption(f"Static API coverage: {cov_data.get('coverage_pct', 0):.2f}% ({cov_data.get('covered_unique', 0)}/{cov_data.get('total_ops', 0)})")

    iteration_dirs = [p for p in sorted(run_path.glob("iteration_*")) if p.is_dir()]
    if iteration_dirs:
        iter_names = [p.name for p in iteration_dirs]
        prev_iter = st.session_state.get("loaded_iteration")
        default_index = iter_names.index(prev_iter) if prev_iter in iter_names else 0
        selected_iter = st.selectbox("Iterations", iter_names, index=default_index, key="loaded_iteration_select")
        st.session_state["loaded_iteration"] = selected_iter

        iter_path = run_path / selected_iter
        st.markdown(f"**Iteration path:** `{iter_path}`")

        iter_summary = load_json_file(iter_path / "test_summary.json") or {}
        passed_rows = iter_summary.get("passed", [])
        failed_rows = iter_summary.get("failed", [])

        st.markdown("**✅ Passed Scenarios**")
        if passed_rows:
            for row in passed_rows:
                mr_identifier = _coerce_mr_id(row.get("mr_id") or row.get("scenario"))
                st.markdown(f"- {row.get('scenario')} (MR={mr_identifier or '?'})")
        else:
            st.caption("No passed scenarios recorded for this iteration.")

        st.markdown("**❌ Failed Scenarios**")
        if failed_rows:
            for row in failed_rows:
                mr_identifier = _coerce_mr_id(row.get("mr_id") or row.get("scenario"))
                title = f"{row.get('scenario')} (MR={mr_identifier or '?'})"
                with st.expander(title):
                    st.code(row.get("details", "(no details)"), language="text")
        else:
            st.caption("No failed scenarios recorded for this iteration.")

        coverage_metrics = load_json_file(iter_path / "coverage_metrics.json")
        if coverage_metrics:
            st.caption(f"Iteration coverage: {coverage_metrics.get('coverage_pct', 0):.2f}% ({coverage_metrics.get('covered_unique', 0)}/{coverage_metrics.get('total_ops', 0)})")

        # Show refined step code (latest)
        steps_file = iter_path / "features" / "steps" / "steps.py"
        if steps_file.exists():
            with st.expander("Latest Step Code (auto-refined)"):
                st.code(steps_file.read_text(encoding="utf-8"), language="python")


        feature_dir = iter_path / "features"
        feature_files = sorted(feature_dir.glob("*.feature")) if feature_dir.exists() else []
        if feature_files:
            with st.expander("Feature files"):
                for feat in feature_files:
                    st.markdown(f"**{feat.name}**")
                    st.code(feat.read_text(encoding="utf-8"), language="gherkin")

        behave_stdout = iter_path / "behave_stdout.txt"
        if behave_stdout.exists():
            with st.expander("Behave stdout"):
                st.code(behave_stdout.read_text(encoding="utf-8"), language="text")
        behave_stderr = iter_path / "behave_stderr.txt"
        if behave_stderr.exists():
            with st.expander("Behave stderr"):
                st.code(behave_stderr.read_text(encoding="utf-8"), language="text")
else:
    st.caption("Load a previous execution folder to inspect historical runs.")

# # Baseline
# if baseline_btn:
#     if not oas_text.strip():
#         st.error("Provide OpenAPI spec first.")
#     else:
#         with st.status("Running Schemathesis...", state="running"):
#             res = run_schemathesis(oas_text, base_url_api)
#         st.write("**Schemathesis result:**")
#         st.code(json.dumps(res, indent=2), language="json")

# Prep session trackers
st.session_state.setdefault("ops_from_codegen", set())
st.session_state.setdefault("ops_from_codegen_hist", [])
st.session_state["covered_ops"].clear()
st.session_state["request_count"] = 0
st.session_state["ops_from_codegen"].clear()
st.session_state["undiscovered_ops_hist"] = []

# Main pipeline
if run_btn:
    if not oas_text.strip():
        st.error("Please provide OpenAPI spec (for step generation).")
        st.stop()

    # Parse OAS (for coverage + endpoint choice)
    try:
        spec_obj = yaml.safe_load(oas_text)
    except Exception:
        try:
            spec_obj = json.loads(oas_text)
        except Exception as e:
            st.error(f"OpenAPI spec not valid JSON/YAML: {e}")
            st.stop()

    # Build LLM + agents
    seed = 42
    mr_llm = build_llm(mr_provider, mr_model, mr_api_key, mr_base_url, mr_temp, seed)
    code_llm = build_llm(step_provider, step_model, step_api_key, step_base_url, step_temp, seed)
    mr_generator, step_generator = make_agents(mr_llm, code_llm)

    task_mr_gen, task_step_code = make_tasks(mr_generator, step_generator)

    # Exec folder
    import datetime
    def get_ts_folder(base="case_studies",case_study_name="PetStore-CS"):
        ts = case_study_name + "-" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(base, ts)
        os.makedirs(folder, exist_ok=True)
        return folder

    # Utility
    def extract_total_ops(spec_obj):
        return sum(len(methods or {}) for methods in (spec_obj.get("paths") or {}).values())

    total_ops = extract_total_ops(spec_obj)
    st.sidebar.markdown(f"**Total API operations in spec:** {total_ops}")

    with _request_logging():
        exec_folder = get_ts_folder()
        i = 0
        agg_summary = st.session_state["agg_summary"]
        agg_summary["mr_ids_all"].clear()
        agg_summary["mr_ids_passed"].clear()
        agg_summary["mr_ids_failed"].clear()
        agg_summary["server_crash_count"] = 0

        run_all_ids: set[str] = set()
        run_passed_ids: set[str] = set()
        run_failed_ids: set[str] = set()

        # Save run config
        config = {
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,

            # 🧠 MR Generator LLM
            "mr_model": mr_model,
            "mr_provider": mr_provider,
            "mr_api_mode": "API" if mr_mode == "API (OpenAI)" else "Local",
            "mr_temperature": mr_temp,
            "mr_base_url": mr_base_url or "https://api.openai.com",

            # 💻 Step Code Generator LLM
            "code_model": step_model,
            "code_provider": step_provider,
            "code_api_mode": "API" if step_mode == "API (OpenAI)" else "Local",
            "code_temperature": step_temp,
            "code_base_url": step_base_url or "https://api.openai.com",

            # 🧾 Global info
            "target_op_cov": target_op_cov,
            "plateau_window": plateau_window,
            "max_requests": max_requests,
            "max_minutes": max_minutes,
        }

        with open(os.path.join(exec_folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        previous_mrs = []
        st.session_state.setdefault("covered_ops", set())
        st.session_state["covered_ops"].clear()
        st.session_state["request_count"] = 0

        run_start_ts = time.time()
        st.session_state["op_cov_hist"] = []
        st.session_state["op_hits_hist"] = []
        total_request_count = 0
        (Path(exec_folder) / "total_requests.json").write_text(json.dumps({"requests_child_total": 0}, indent=2), encoding="utf-8")

        # FIX: compute prefixes once for the run
        path_prefixes = _extract_path_prefixes(spec_obj, base_url_api)
        (Path(exec_folder) / "path_prefixes.json").write_text(json.dumps(path_prefixes, indent=2), encoding="utf-8")

        # STOP: plateau tolerance in percentage points
        PLATEAU_DELTA = 0.25
        stop_reason = None

        # Test manager (stop-policy + relay storage). Keep UI/output logic unchanged.
        test_manager = TestManager(
            target_cov=float(target_op_cov),
            plateau_window=int(plateau_window),
            max_requests=int(max_requests),
            max_minutes=float(max_minutes),
        )
        test_manager.record_agent_output("previous_mrs", previous_mrs)
        
        # cumulative coverage across all iterations this run
        st.session_state.setdefault("cum_mapped_ops", set())
        st.session_state.setdefault("cum_cov_hist", [])

        test_manager.start_loop(start_iteration=i)
        while test_manager.continue_loop():
            i = test_manager.iteration
            iter_dir = Path(exec_folder) / f"iteration_{i}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # ---- Step 1: MR Generation ----
            try:
                with st.status("MR Generator running...", state="running") as s1:
                    mrs_raw, _ = kickoff_with_logs(
                        [mr_generator],
                        [task_mr_gen],
                        inputs={"oas": oas_text, "previous_mrs": test_manager.get_agent_output("previous_mrs", previous_mrs), "api_operations": str(st.session_state.get("undiscovered_ops_hist") or "")}
                    )
                    mrs_list = try_parse_json_list(mrs_raw)
                    count_empty = 0
                    for retry in range(3):
                        if not mrs_list:
                            mrs_raw, _ = kickoff_with_logs(
                                [mr_generator],
                                [task_mr_gen],
                                inputs={"oas": oas_text, "previous_mrs": test_manager.get_agent_output("previous_mrs", previous_mrs), "api_operations": str(st.session_state.get("undiscovered_ops_hist") or "")}
                            )
                            mrs_list = try_parse_json_list(mrs_raw)
                            if not mrs_list and retry == 2:
                                st.warning("No new unique MRs found after 3 attempts. Stopping.")
                                break
                        else:
                            break

                    if not mrs_list:
                        stop_reason = save_and_stop("no_new_unique_mrs")
                        st.warning("No new unique MRs. Stopping.")
                        test_manager.stop(stop_reason)
                        break

                    test_manager.record_agent_output("mr_generator_raw", mrs_list)
                    mrs_list=refine_mr_with_gpt5(str(test_manager.get_agent_output("mr_generator_raw")),str(oas_text))
                    mrs_list=str(mrs_list)
                    mrs_list=ast.literal_eval(mrs_list)
                    
                    mrs_list=try_parse_json_list(mrs_list)
                    test_manager.record_agent_output("mr_generator", mrs_list)
                    st.caption("Agent: MR Generator")
                    st.code(json.dumps(mrs_list, indent=2), language="json")
                    (iter_dir / "mr_generator.json").write_text(json.dumps({"raw": mrs_list}, indent=2), encoding="utf-8")

                batch_ids = extract_mr_ids(mrs_list)
                agg_summary["mr_ids_all"].extend(batch_ids)
                run_all_ids.update(batch_ids)
                previous_mrs.append(mrs_list)
                test_manager.record_agent_output("previous_mrs", previous_mrs)
                # Persist full history for reproducibility/debugging (no UI/output changes).
                try:
                    (Path(exec_folder) / "previous_mrs.json").write_text(
                        json.dumps(previous_mrs, indent=2),
                        encoding="utf-8",
                    )
                    (iter_dir / "previous_mrs.json").write_text(
                        json.dumps(previous_mrs, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                s1.update(label="MR Generator finished ✅", state="complete")
            except Exception as e:
                stop_reason = {"reason": "mr_generation_error", "iteration": i, "error": str(e)}
                (Path(exec_folder) / "stop_reason.json").write_text(json.dumps(stop_reason, indent=2), encoding="utf-8")
                st.error(f"MR generation failed: {e}")
                st.code(traceback.format_exc(), language="text")
                test_manager.stop(stop_reason)
                break

            # ---- Step 2: Build Feature + Generate Steps ----
            try:
                with st.status("Generating Behave features + steps...", state="running") as s2:
                    mrs_list = test_manager.get_agent_output("mr_generator", mrs_list)
                    features_dir, scenario_map = write_iteration_feature(mrs_list, iter_dir, i)

                    all_step_modules = []
                    generated_steps = []
                    for mr in mrs_list:
                        try:
                            step_code, gen_logs = generate_step_code_for_mr_with_repair(
                                step_agent=step_generator,
                                task_step_code=task_step_code,
                                mr=mr,
                                oas_text=oas_text,
                                base_url=base_url_api,
                                max_rounds=2,
                                api_operations=str(st.session_state.get("undiscovered_ops_hist") or "")
                            )
                            # -----------------------------
                            # 🚀 NEW: GPT-5 API REFINEMENT
                            # -----------------------------
                            print("--")
                            print(step_code)

                            print("Refining step code with GPT-5...")
                            print("--")
                            try:
                                mr_text = json.dumps(mr, indent=2)
                                test_manager.record_agent_output("step_code_raw", step_code)
                                refined = refine_with_gpt5(test_manager.get_agent_output("step_code_raw"), mr_text, oas_text, base_url_api)

                                # keep CrewAI self-healing afterwards (AST repair)
                                ok, err = _ast_ok(refined)
                                if ok:
                                    step_code = refined
                                else:
                                    st.warning(f"GPT-5 refiner produced syntax errors; using unrefined version. Error: {err}")

                            except Exception as e:
                                st.warning(f"GPT-5 Refiner failed: {e}")

                            # Continue the pipeline normally
                            all_step_modules.append(step_code)
                            generated_steps.append((mr.get("id", "MRX"), step_code, gen_logs))
                        except Exception as inner_e:
                            st.error(f"Step generation failed for MR {mr.get('id')}: {inner_e}")
                            st.code(traceback.format_exc(), language="text")

                    steps_path = write_steps_py(all_step_modules, features_dir)
                    (iter_dir / "steps.py.txt").write_text("\n\n".join(all_step_modules), encoding="utf-8")

                    st.caption("Agent: Step Code Generator (per MR)")
                    for mid, step_source, logs in generated_steps:
                        display_mid = _coerce_mr_id(mid) or mid or "MR"
                        st.markdown(f"**Step code for {display_mid}**")
                        st.code(step_source, language="python")
                        formatted_logs = "\n".join(
                            f"{tag}: {msg}"
                            for tag, msg in logs
                            if msg and msg != step_source
                        )
                        if formatted_logs:
                            st.caption("Generation notes")
                            st.code(formatted_logs, language="text")

                    templates = _template_paths_from_spec(spec_obj)
                    code_ops_raw = set()
                    for mcode in all_step_modules:
                        code_ops_raw |= extract_ops_from_tests_code(mcode, base_url_var="BASE_URL")
                    
                    code_ops_raw = sorted(code_ops_raw, key=lambda x: (x[0], x[1]))
                

                    code_ops_norm = {_normalize_hit_to_template(m, p, templates, path_prefixes) for (m, p) in code_ops_raw}
                    code_ops_norm = sorted(code_ops_norm, key=lambda x: (x[0], x[1]))
                
                    
                    new_ops = {op for op in code_ops_norm if op in templates} - st.session_state["ops_from_codegen"]
                    new_ops = sorted(new_ops, key=lambda x: (x[0], x[1]))
            
                    if new_ops:
                        st.session_state["ops_from_codegen"].update(new_ops)
                    total_ops_cnt = len(templates)
                    static_cov = len(st.session_state["ops_from_codegen"]) / max(total_ops_cnt, 1) * 100.0
                    
                    

                    all_ops_in_spec = _template_paths_from_spec(spec_obj)
                    discovered_ops = st.session_state["ops_from_codegen"]
                    undiscovered_ops = sorted(all_ops_in_spec - discovered_ops, key=lambda x: (x[0], x[1]))
                    st.session_state["undiscovered_ops_hist"] = undiscovered_ops
                    
                    st.write(f"**code_ops_raw:** {code_ops_raw} ")
                    st.write(f"**code_ops_norm:** {code_ops_norm} ")
                    st.write(f"**new_ops:** {new_ops} ")
                    st.write(f"**Len match:** {len(new_ops) == len(code_ops_norm)} ")
                    # --- Extract undiscovered (not yet covered) operations ---
                    st.write("All API Operations extracted till now:", sorted(list(st.session_state["ops_from_codegen"])))
                    st.write(f"🚫 **Undiscovered API Operations (not yet covered by step code)\n: {str(undiscovered_ops)}**")
                    
                    

                    
                    # ---- NEW cumulative tracking ----
                    st.session_state.setdefault("cum_static_hist", [])
                    st.session_state["cum_static_hist"].append(static_cov)
                    (Path(exec_folder) / "cumulative_static_coverage.json").write_text(
                        json.dumps({
                            "total_ops": total_ops_cnt,
                            "ops_referenced": sorted(list(st.session_state["ops_from_codegen"])),
                            "coverage_pct": static_cov
                        }, indent=2),
                        encoding="utf-8"
                    )
                    st.session_state["ops_from_codegen_hist"].append(len(st.session_state["ops_from_codegen"]))
                    
                    
                    st.write(f"**Static Operation Coverage (referenced in steps):** {static_cov:.2f}% "
                            f"({len(st.session_state['ops_from_codegen'])}/{total_ops_cnt})")

                    # Save static coverage
                    (iter_dir / "static_coverage.json").write_text(json.dumps({
                        "ops_referenced": sorted(list(st.session_state["ops_from_codegen"])),
                        "coverage_pct": static_cov,
                        "total_ops": total_ops_cnt
                    }, indent=2), encoding="utf-8")


                    s2.update(label="Features + Steps ready ✅", state="complete")
            except Exception as e:
                stop_reason = {"reason": "step_generation_error", "iteration": i, "error": str(e)}
                (Path(exec_folder) / "stop_reason.json").write_text(json.dumps(stop_reason, indent=2), encoding="utf-8")
                st.error(f"Step generation failed in iteration {i}: {e}")
                st.code(traceback.format_exc(), language="text")
                test_manager.stop(stop_reason)
                break

            # ---- Step 3: Execute Behave ----
            try:
                with st.status("Executing Behave...", state="running") as s3:
                    res = run_behave_and_collect(features_dir, base_url_api, spec_obj, iter_dir, scenario_map)

                    (iter_dir / "behave_stdout.txt").write_text(res.get("stdout", ""), encoding="utf-8")
                    (iter_dir / "behave_stderr.txt").write_text(res.get("stderr", ""), encoding="utf-8")

                    
                    (iter_dir / "test_summary.json").write_text(json.dumps({
                        "passed": res.get("passed", []),
                        "failed": res.get("failed", []),
                        "coverage_pct": res.get("coverage_pct", 0),
                        "covered_ops": res.get("covered_ops", []),
                        "total_ops": res.get("total_ops", 0),
                        "request_count": res.get("request_count", 0),
                        "code_version": 1,
                    }, indent=2), encoding="utf-8")
                    s3.update(label="Behave finished ✅", state="complete")

                st.subheader("✅ Passed Scenarios")
                if res.get("passed"):
                    for row in res["passed"]:
                        st.markdown(f"- {row.get('scenario')} (MR={row.get('mr_id', '???')})")
                        st.session_state["agg_summary"]["mr_ids_passed"].append(row.get("mr_id",""))
                else:
                    st.caption("No passed scenarios detected.")

                st.subheader("❌ Failed Scenarios — with full error trace")
                if res.get("failed"):
                    for row in res["failed"]:
                        title = f"{row.get('scenario')} (MR={row.get('mr_id', '???')})"
                        st.session_state["agg_summary"]["mr_ids_failed"].append(row.get("mr_id",""))
                        with st.expander(title):
                            st.code(row.get("details", "(no details)"), language="text")
                else:
                    st.caption("No failed scenarios detected.")
                # 👉 Insert the new block right here
                server_500_count = 0
                for row in res.get("failed", []):
                    details = row.get("details", "") or ""
                    if "500" in details or "Internal Server Error" in details:
                        server_500_count += 1
                st.session_state["agg_summary"]["server_crash_count"] += server_500_count
                (iter_dir / "server_crash_count.json").write_text(
                    json.dumps({"iteration": i, "count_500": server_500_count}, indent=2),
                    encoding="utf-8"
                )

                # ----- Coverage: iteration + cumulative -----
                # Normalize to tuples (already mapped to spec by the function)
                iter_mapped = {tuple(h) if not isinstance(h, tuple) else h
                            for h in res.get("covered_ops", [])}

                # Ensure accumulators exist
                st.session_state.setdefault("cum_mapped_ops", set())
                st.session_state.setdefault("cum_cov_hist", [])

                # Update cumulative
                st.session_state["cum_mapped_ops"] |= iter_mapped
                cum_count = len(st.session_state["cum_mapped_ops"])
                total_ops_val = res.get("total_ops", 0)
                cum_cov = (cum_count / max(total_ops_val or 1, 1)) * 100.0
                st.session_state["cum_cov_hist"].append(cum_cov)

                # Iteration histories (keep)
                st.session_state.setdefault("op_cov_hist", [])
                st.session_state.setdefault("op_hits_hist", [])
                st.session_state["op_cov_hist"].append(res.get("coverage_pct", 0))
                st.session_state["op_hits_hist"].append(len(iter_mapped))

                # Persist histories & cumulative artifacts
                (Path(exec_folder) / "op_cov_history.json").write_text(json.dumps(st.session_state["op_cov_hist"], indent=2), encoding="utf-8")
                (Path(exec_folder) / "op_hits_history.json").write_text(json.dumps(st.session_state["op_hits_hist"], indent=2), encoding="utf-8")
                (Path(exec_folder) / "cumulative_covered_ops.json").write_text(
                    json.dumps([[m, p] for (m, p) in sorted(st.session_state["cum_mapped_ops"])], indent=2),
                    encoding="utf-8"
                )
                (Path(exec_folder) / "cumulative_coverage.json").write_text(
                    json.dumps({"total_ops": total_ops_val,
                                "covered_unique": cum_count,
                                "coverage_pct": cum_cov}, indent=2),
                    encoding="utf-8"
                )

                # UI: show both
                st.subheader("📊 API Operation Coverage")
            
                st.write(f"{static_cov}")

                st.write(f"**RAW from the Code** {code_ops_raw} ")
                st.write(f"**Normalized to Map OpenAPI Spec** {code_ops_norm} ")
                st.write(f"**New APIs Discovered:** {new_ops} ")
                st.write(f"**Length match:** {len(new_ops) == len(code_ops_norm)} ")
                # --- Extract undiscovered (not yet covered) operations ---
                st.write("All API Operations extracted till now:", sorted(list(st.session_state["ops_from_codegen"])))
                st.write(f"🚫 **Undiscovered API Operations (not yet covered by step code)\n: {str(undiscovered_ops)}**")
            

                # Accumulate child request counts + persist (safe)
                try:
                    total_request_count += int(res.get("request_count", 0) or 0)
                except Exception:
                    pass
                (Path(exec_folder) / "total_requests.json").write_text(
                    json.dumps({"requests_child_total": total_request_count}, indent=2), encoding="utf-8"
                )
            except Exception as e:
                stop_reason = {"reason": "behave_execution_error", "iteration": i, "error": str(e)}
                (Path(exec_folder) / "stop_reason.json").write_text(json.dumps(stop_reason, indent=2), encoding="utf-8")
                st.error(f"Behave execution failed in iteration {i}: {e}")
                st.code(traceback.format_exc(), language="text")
                test_manager.stop(stop_reason)
                break

            # ----------------- STOP CRITERIA -----------------
            now = time.time()
            elapsed_min = (now - run_start_ts) / 60.0
            current_cov = st.session_state["ops_from_codegen_hist"][-1] if st.session_state["ops_from_codegen_hist"] else 0.0
            current_cum_cov = st.session_state["cum_static_hist"][-1] if st.session_state["cum_static_hist"] else 0.0

            # Keep the test manager in sync (does not change stop/UI behavior here).
            try:
                test_manager.update_metrics(
                    coverage=float(current_cum_cov),
                    unique_ops=int(current_cov or 0),
                    requests=int(total_request_count or 0),
                    crashes=0,
                )
            except Exception:
                pass
            
            def save_and_stop(reason_key: str, extra: dict | None = None):
                reason = {
                    "reason": reason_key,
                    "iteration": i,
                    "coverage_pct_iteration": current_cov,
                    "coverage_pct_cumulative": current_cum_cov,
                    "elapsed_minutes": elapsed_min,
                    "requests_child_total": total_request_count,
                }
                if extra:
                    reason.update(extra)
                (Path(exec_folder) / "stop_reason.json").write_text(json.dumps(reason, indent=2), encoding="utf-8")
                return reason

            
            # 1) Target coverage achieved
            if current_cum_cov >= float(target_op_cov):
                st.success(f"Target static coverage reached: {current_cum_cov:.2f}% ≥ {target_op_cov}%")
                stop_reason = save_and_stop("target_static_coverage_reached")
                test_manager.stop(stop_reason)
                break

            # 2) Plateau detection
            if plateau_window > 0 and len(st.session_state["ops_from_codegen_hist"]) >= plateau_window:
                window = st.session_state["ops_from_codegen_hist"][-plateau_window:]
                if (max(window) - min(window)) < PLATEAU_DELTA:
                    st.info(f"Static coverage plateau detected over last {plateau_window} iterations.")
                    stop_reason = save_and_stop("static_plateau_detected", {
                        "delta_ops": max(window) - min(window),
                        "plateau_window": int(plateau_window),
                    })
                    test_manager.stop(stop_reason)
                    break

            # 3) Max requests
            if total_request_count >= int(max_requests):
                st.warning(f"Max requests reached: {total_request_count} ≥ {max_requests}")
                stop_reason = save_and_stop("max_requests_reached")
                test_manager.stop(stop_reason)
                break

            # 4) Max minutes
            if elapsed_min >= float(max_minutes):
                st.warning(f"Max minutes reached: {elapsed_min:.2f} ≥ {max_minutes}")
                stop_reason = save_and_stop("max_minutes_reached")
                test_manager.stop(stop_reason)
                break

            if len(undiscovered_ops) == 0:
                st.write("All operations have been discovered!")
                test_manager.stop(stop_reason)
                break

            # Continue to next iteration
            test_manager.next_iteration()

    # === End-of-run MR Summary ===
    def _uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out

    all_ids = _uniq(st.session_state["agg_summary"].get("mr_ids_all", []))
    passed_ids = _uniq(st.session_state["agg_summary"].get("mr_ids_passed", []))
    failed_ids = _uniq(st.session_state["agg_summary"].get("mr_ids_failed", []))
    total_crashes = st.session_state["agg_summary"].get("server_crash_count", 0)


    st.subheader("🧾 MR Run Summary")
    summary_rows = [
        {"Metric": "Total MRs generated", "Count": len(all_ids), "MR IDs": ", ".join(all_ids) or "—"},
        {"Metric": "Passed",               "Count": len(passed_ids), "MR IDs": ", ".join(passed_ids) or "—"},
        {"Metric": "Failed",               "Count": len(failed_ids), "MR IDs": ", ".join(failed_ids) or "—"},
        {"Metric": "Internal Server Crashes (500)", "Count": total_crashes, "MR IDs": "—"}
    ]
    executed = len(passed_ids) + len(failed_ids)
    fdr = (len(failed_ids) / executed * 100) if executed > 0 else 0.0
    summary_rows.append({"Metric": "Fault Detection Rate", "Count": f"{fdr:.2f}%", "MR IDs": "—"})
    st.table(summary_rows)


    # Save run summary
    run_summary = {
        "run_id":exec_folder.split(os.sep)[-1],
        "mr_ids_all": all_ids,
        "mr_ids_passed": passed_ids,
        "mr_ids_failed": failed_ids,
        "fault_detection_rate_pct": fdr,
        "total_api_ops": total_ops,
        "no_of_mrs": len(all_ids),
        "server_500_count": total_crashes,
        "no_of_iterations": i + 1,
        "api_coverage": len(st.session_state["ops_from_codegen"]) / max(total_ops, 1) * 100.0,
        "manual_review": "{'wrong_tests':[],'useful_tests':[],'correct_mrs':[]}",
        "stop_reason": stop_reason or {"reason": "completed_without_explicit_stop"},
        "num_mrs_passed":len(passed_ids),
        "num_mrs_failed":len(failed_ids)
    }

    # --- 1. Write the usual run_summary.json (unchanged) ---
    (Path(exec_folder) / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8"
    )

    # --- 2. Append to summary_of_runs.jsonl (NEW FEATURE) ---
    summary_jsonl_path = Path("summary_of_runs.jsonl")
    summary_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary) + "\n")

    append_run_to_csv(run_summary)
    # === Combine outputs across iterations ===
    parent_dir = Path(exec_folder)

    # 1️⃣ Combine all .feature files
    combined_feature_path = parent_dir / "all_iterations.feature"
    with open(combined_feature_path, "w", encoding="utf-8") as fout:
        for iter_dir in sorted(parent_dir.glob("iteration_*")):
            feature_dir = iter_dir / "features"
            for feat_file in feature_dir.glob("*.feature"):
                fout.write(f"\n# --- From {feat_file.relative_to(parent_dir)} ---\n\n")
                fout.write(feat_file.read_text(encoding="utf-8").strip())
                fout.write("\n\n")

    # 2️⃣ Combine all behave_report.json files
    combined_behave_path = parent_dir / "combined_behave_report.json"
    combined_data = []

    for iter_dir in sorted(parent_dir.glob("iteration_*")):
        behave_json = iter_dir / "behave_report.json"
        if behave_json.exists():
            try:
                data = json.loads(behave_json.read_text(encoding="utf-8"))
                for feature in data:
                    feature["_iteration"] = iter_dir.name
                combined_data.extend(data)
            except Exception as e:
                print(f"⚠️ Failed to read {behave_json}: {e}")

    combined_behave_path.write_text(json.dumps(combined_data, indent=2), encoding="utf-8")

    st.success("✅ Combined outputs created successfully!")
    st.write("📂 Files generated:")
    st.code(str(combined_feature_path))
    st.code(str(combined_behave_path))
