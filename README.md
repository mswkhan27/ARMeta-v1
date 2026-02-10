# Agentic Rest API Metamorphic Testing Tool (ARMeta v1)

CrewAI-based metamorphic testing for REST APIs with a Streamlit UI. The app generates metamorphic relations (MRs), generates Behave step code, executes tests, and tracks operation coverage against an OpenAPI specification.

## Prerequisites

- **Python:** 3.10+ (recommended: 3.11)
- **Pip:** recent version (`python -m pip install --upgrade pip`)
- **(Optional) Ollama:** if you want to run the LLM locally

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Configure LLM access

### Option A: OpenAI API

Set your key via environment variable:

```bash
# Windows (PowerShell)
Update the API key at line 49 in approach_gui.py

## Run the app

```bash
streamlit run approach_gui.py
```

## Typical workflow (in the UI)

1. Choose MR backend (OpenAI) and model.
2. Choose Code backend (OpenAI) and model.
3. Provide the OpenAPI spec (upload JSON/YAML or paste text) e.g from apis\my-petstore\specifications\API-Specs\openapi.yaml.
4. Set the **Base URL of the System Under Test** you want to test.
5. Click **Run MR Pipeline**.

## Outputs

- Each run is saved under `case_studies/<CaseStudyName>-<YYYYMMDD_HHMMSS>/`.
- The app also has json per  to `runs.csv` (created automatically).

## Notes


- Do not hardcode API keys in source code; use env vars or Streamlit secrets.
