import argparse
import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI


ADZUNA_API_BASE = "https://api.adzuna.com/v1/api/jobs"


def _safe_get(d: Dict[str, Any], path: List[str], default: Optional[Any] = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fetch_jobs(title: str, location: str, *, results: int = 20) -> pd.DataFrame:
    load_dotenv()

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    country = (os.getenv("ADZUNA_COUNTRY") or "us").strip().lower()

    if not app_id or not app_key:
        raise RuntimeError(
            "Missing Adzuna credentials. Set ADZUNA_APP_ID and ADZUNA_APP_KEY in your environment (or .env)."
        )

    url = f"{ADZUNA_API_BASE}/{country}/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": title,
        "where": location,
        "results_per_page": max(20, int(results)),
        "content-type": "application/json",
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling Adzuna API: {e}") from e

    if resp.status_code != 200:
        snippet = (resp.text or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        raise RuntimeError(f"Adzuna API error ({resp.status_code}): {snippet}")

    try:
        payload = resp.json()
    except ValueError as e:
        raise RuntimeError("Adzuna API returned non-JSON response.") from e

    results_list = payload.get("results", [])
    if not isinstance(results_list, list):
        raise RuntimeError("Unexpected Adzuna API response shape: 'results' is not a list.")

    rows: List[Dict[str, Any]] = []
    for item in results_list:
        if not isinstance(item, dict):
            continue

        rows.append(
            {
                "title": item.get("title"),
                "company": _safe_get(item, ["company", "display_name"]),
                "location": _safe_get(item, ["location", "display_name"]),
                "description": item.get("description"),
            }
        )

    return pd.DataFrame(rows, columns=["title", "company", "location", "description"])


def _combine_descriptions(df: pd.DataFrame, *, max_chars: int = 120_000) -> str:
    if "description" not in df.columns:
        return ""

    parts: List[str] = []
    for v in df["description"].tolist():
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        parts.append(s)

    combined = "\n\n---\n\n".join(parts)
    combined = re.sub(r"[ \t]+", " ", combined)
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()
    if len(combined) <= max_chars:
        return combined
    return combined[: max(0, max_chars - 200)] + "\n\n[TRUNCATED]\n"


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output.")

    # Best case: pure JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: try to find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in model output.")
    candidate = text[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON is not an object.")
    return obj


def analyze_descriptions_with_openai(
    df: pd.DataFrame,
    *,
    max_input_chars: int = 120_000,
) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in your environment (or .env).")

    text = _combine_descriptions(df, max_chars=max_input_chars)
    if not text:
        raise RuntimeError("No job descriptions found to analyze.")

    instructions = """You are analyzing a batch of job descriptions.
Return ONLY valid JSON (no markdown) with this exact structure:
{
  "top_skills": [{"skill": "string", "frequency": 123}],
  "tools_technologies": ["string", "..."],
  "categories": {
    "Programming": ["string", "..."],
    "Tools": ["string", "..."],
    "Cloud": ["string", "..."],
    "Soft Skills": ["string", "..."]
  }
}

Rules:
- "top_skills" must contain exactly 10 items, sorted by frequency desc, frequency must be an integer.
- Skills should be normalized (e.g., "python" -> "Python", "aws" -> "AWS").
- "tools_technologies" should be unique, normalized, and cover concrete tools/tech (e.g., AWS, Docker, Kubernetes, Terraform, Spark).
- Categories should contain unique, normalized strings. If unsure where an item fits, pick the best single category.
"""

    prompt = f"{instructions}\n\nJob descriptions (may be truncated):\n\n{text}"

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,  # Plain string input (not message format)
            temperature=0,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        # "Return text output cleanly" - best-effort fallback.
        output_text = str(response)

    return _extract_json_object(output_text)


def save_top_skills_to_csv(
    analysis_json: Dict[str, Any],
    *,
    job_role: str,
    location: str,
    csv_path: str = "skills_data.csv",
) -> pd.DataFrame:
    top_skills = analysis_json.get("top_skills", [])
    if not isinstance(top_skills, list):
        raise RuntimeError("OpenAI analysis JSON missing 'top_skills' list.")

    rows: List[Dict[str, Any]] = []
    for item in top_skills:
        if not isinstance(item, dict):
            continue
        skill = item.get("skill")
        freq = item.get("frequency")
        if skill is None:
            continue
        try:
            freq_int = int(freq) if freq is not None else 0
        except (TypeError, ValueError):
            freq_int = 0

        rows.append(
            {
                "Skill": str(skill).strip(),
                "Frequency": freq_int,
                "Job Role": job_role,
                "Location": location,
                "Date": date.today().isoformat(),
            }
        )

    out_df = pd.DataFrame(rows, columns=["Skill", "Frequency", "Job Role", "Location", "Date"])

    # Load existing dataset (if any), append, then clean + aggregate before saving.
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
        except Exception:
            existing = pd.DataFrame(columns=["Skill", "Frequency", "Job Role", "Location", "Date"])
    else:
        existing = pd.DataFrame(columns=["Skill", "Frequency", "Job Role", "Location", "Date"])

    combined = pd.concat([existing, out_df], ignore_index=True)
    for col in ["Skill", "Job Role", "Location", "Date"]:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.strip()
    if "Frequency" in combined.columns:
        combined["Frequency"] = pd.to_numeric(combined["Frequency"], errors="coerce").fillna(0).astype(int)

    # Remove duplicates for same date, role, location by aggregating per skill.
    aggregated = (
        combined.groupby(["Date", "Job Role", "Location", "Skill"], as_index=False)["Frequency"]
        .sum()
        .sort_values(["Date", "Job Role", "Location", "Frequency", "Skill"], ascending=[True, True, True, False, True])
    )

    # Write safely on Windows (handles file locks if CSV is open elsewhere).
    tmp_path = f"{csv_path}.tmp"
    aggregated.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, csv_path)
    except PermissionError:
        fallback_path = csv_path.replace(".csv", "_new.csv") if csv_path.lower().endswith(".csv") else f"{csv_path}_new"
        aggregated.to_csv(fallback_path, index=False)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Could not overwrite '{csv_path}' (file may be open/locked). Wrote cleaned data to '{fallback_path}' instead."
        )
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch job listings from the Adzuna API.")
    parser.add_argument("--title", required=True, help="Job title / keywords, e.g. 'Data Scientist'")
    parser.add_argument("--location", required=True, help="Location, e.g. 'New York' or 'London'")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many jobs to fetch (minimum 20). Default: 20",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze the fetched job descriptions using OpenAI and print structured JSON.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=120_000,
        help="Max characters of combined descriptions sent to OpenAI. Default: 120000",
    )
    args = parser.parse_args()

    try:
        df = fetch_jobs(args.title, args.location, results=args.limit)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if df.empty:
        print("No job listings returned.")
        return 0

    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", None)
    print(df)

    if args.analyze:
        try:
            result = analyze_descriptions_with_openai(df, max_input_chars=args.max_input_chars)
        except Exception as e:
            print(f"\nOpenAI analysis error: {e}")
            return 1

        print("\n=== OpenAI job description analysis (JSON) ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        try:
            saved_df = save_top_skills_to_csv(result, job_role=args.title, location=args.location)
        except Exception as e:
            print(f"\nCSV save error: {e}")
            return 1

        print("\n=== Saved top skills to skills_data.csv ===")
        print(saved_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
