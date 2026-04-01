import json
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument


st.set_page_config(page_title="Job Market Intelligence Dashboard", layout="wide")
st.title("Job Market Intelligence Dashboard")


CSV_PATH = "skills_data.csv"
ADZUNA_API_BASE = "https://api.adzuna.com/v1/api/jobs"


def _safe_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output.")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: locate first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in model output.")
    candidate = text[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON is not an object.")
    return obj


def _normalize_skill(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_key(s: str) -> str:
    s = _normalize_skill(s).casefold()
    # Remove punctuation that can differ between resume and extracted skills.
    s = re.sub(r"[^a-z0-9 +#.-]", "", s)
    return s


@st.cache_data
def load_market_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Date", "Job Role", "Location", "Skill", "Frequency"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0).astype(int)
    for col in ["Job Role", "Location", "Skill"]:
        df[col] = df[col].astype(str).str.strip()

    return df.dropna(subset=["Date"])


def extract_resume_text(uploaded_file) -> str:
    import io

    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if filename.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        parts: List[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts)

    if filename.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError("Unsupported resume format. Upload .txt, .pdf, or .docx.")


def extract_skills_from_resume(resume_text: str, *, max_skills: int = 30) -> List[str]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in your environment (or .env).")

    client = OpenAI(api_key=api_key)
    prompt = f"""
Extract a concise list of skills from the resume text.
Return ONLY valid JSON (no markdown) with this exact structure:
{{"skills": ["string", "..."]}}

Rules:
- Include technical skills (languages, frameworks, tools), domain skills, and relevant soft skills if present.
- Remove duplicates.
- Prefer shorter, canonical skill names.
- Maximum {max_skills} skills.

Resume text:
{resume_text}
""".strip()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0,
    )

    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError("OpenAI returned no readable text output.")

    obj = _safe_json_object(output_text)
    skills = obj.get("skills", [])
    if not isinstance(skills, list):
        raise ValueError("OpenAI output JSON missing 'skills' list.")
    return [_normalize_skill(s) for s in skills if isinstance(s, str) and s.strip()]


def analyze_job_descriptions_with_openai(descriptions_text: str) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in your environment (or .env).")

    client = OpenAI(api_key=api_key)

    instructions = """
You are analyzing a batch of job descriptions for a single role/location.
Return ONLY valid JSON (no markdown) with this exact structure:
{
  "top_skills": [{"skill": "string", "frequency": 123}],
  "top_tools": [{"tool": "string", "frequency": 123}],
  "categories": {
    "Programming": ["string", "..."],
    "Tools": ["string", "..."],
    "Cloud": ["string", "..."],
    "Soft Skills": ["string", "..."]
  }
}

Rules:
- "top_skills" must contain exactly 10 items, sorted by frequency desc, frequency must be an integer.
- "top_tools" must contain up to 10 items, sorted by frequency desc, frequency must be an integer.
- Normalize names (e.g., "python" -> "Python", "aws" -> "AWS").
- Categories should contain unique, normalized strings.
""".strip()

    max_chars = 110_000
    text = (descriptions_text or "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 200] + "\n\n[TRUNCATED]\n"

    prompt = f"{instructions}\n\nJob descriptions (may be truncated):\n\n{text}"
    response = client.responses.create(model="gpt-4.1-mini", input=prompt, temperature=0)
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError("OpenAI returned no readable text output.")
    return _safe_json_object(output_text)


def _save_latest_skills_csv(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _normalize_location(location: str) -> str:
    """Normalize location input to work better with Adzuna API.
    
    When user enters "United States", "USA", etc., return empty string to search all locations.
    This is because the Adzuna API doesn't handle broad country-level searches well.
    """
    loc = (location or "").strip().lower()
    
    # Broad location patterns that should search across all USA
    broad_patterns = [
        "united states",
        "usa",
        "us",
        "entire us",
        "all usa",
        "all us",
        "nationwide",
        "nation-wide",
    ]
    
    for pattern in broad_patterns:
        if pattern == loc or loc.startswith(pattern):
            return ""  # Empty location = search all
    
    return (location or "").strip()


def _fetch_adzuna_jobs(
    what: str,
    where: str,
    *, results: int = 40,
) -> pd.DataFrame:
    load_dotenv()
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError("Missing Adzuna credentials. Set ADZUNA_APP_ID and ADZUNA_APP_KEY in your environment (or .env).")

    country = (os.getenv("ADZUNA_COUNTRY") or "us").strip().lower()
    url = f"{ADZUNA_API_BASE}/{country}/search/1"

    what = (what or "").strip()
    where = _normalize_location(where)  # Normalize location input

    params: Dict[str, Any] = {
        "app_id": app_id,
        "app_key": app_key,
        "what": what,
        "where": where,
        "results_per_page": max(20, int(results)),
        # Prefer fresh postings to make filter changes visibly different.
        "sort_by": "date",
        "content-type": "application/json",
    }

    resp = requests.get(
        url,
        params=params,
        timeout=30,
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Adzuna API error ({resp.status_code}): {(resp.text or '').strip()[:500]}")

    payload = resp.json()
    results_list = payload.get("results", [])
    if not isinstance(results_list, list):
        return pd.DataFrame(columns=["title", "company", "location", "description", "created"])

    rows: List[Dict[str, Any]] = []
    for item in results_list:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": item.get("title"),
                "company": (item.get("company") or {}).get("display_name") if isinstance(item.get("company"), dict) else None,
                "location": (item.get("location") or {}).get("display_name") if isinstance(item.get("location"), dict) else None,
                "description": item.get("description"),
                "created": item.get("created"),
                "apply_url": item.get("redirect_url"),
            }
        )
    return pd.DataFrame(rows, columns=["title", "company", "location", "description", "created", "apply_url"])


def _parse_date_input(d: Any) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str) and d.strip():
        return date.fromisoformat(d.strip())
    return None


def _filter_jobs_by_date(
    jobs_df: pd.DataFrame,
    created_from: Optional[date],
    created_to: Optional[date],
    last_n_days: Optional[int],
) -> pd.DataFrame:
    if jobs_df.empty or "created" not in jobs_df.columns:
        return jobs_df

    created = pd.to_datetime(jobs_df["created"], errors="coerce", utc=True).dt.date
    out = jobs_df.copy()
    out["__created_date"] = created
    out = out.dropna(subset=["__created_date"])

    if last_n_days is not None and last_n_days > 0:
        cutoff = date.today() - pd.Timedelta(days=int(last_n_days))  # type: ignore[arg-type]
        out = out[out["__created_date"] >= cutoff]

    if created_from:
        out = out[out["__created_date"] >= created_from]
    if created_to:
        out = out[out["__created_date"] <= created_to]

    return out.drop(columns=["__created_date"])


def _filter_jobs_by_title_overlap(jobs_df: pd.DataFrame, role_query: str) -> pd.DataFrame:
    if jobs_df.empty or "title" not in jobs_df.columns:
        return jobs_df

    query = (role_query or "").strip().casefold()
    if not query:
        return jobs_df

    # Ignore very common filler words so overlap is meaningful.
    stop_words = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "with",
        "on",
        "at",
        "by",
        "from",
        "sr",
        "senior",
        "jr",
        "junior",
        "ii",
        "iii",
        "iv",
        "lead",
    }

    query_tokens = {
        t
        for t in re.findall(r"[a-z0-9+#.]+", query)
        if t and t not in stop_words and len(t) > 1
    }
    if not query_tokens:
        return jobs_df

    def has_overlap(title: Any) -> bool:
        title_tokens = {
            t
            for t in re.findall(r"[a-z0-9+#.]+", str(title or "").casefold())
            if t and t not in stop_words and len(t) > 1
        }
        return bool(query_tokens.intersection(title_tokens))

    return jobs_df[jobs_df["title"].apply(has_overlap)].copy()


def _match_market_filter(df: pd.DataFrame, role: str, location: str) -> pd.DataFrame:
    role_key = (role or "").strip().casefold()
    loc_key = (location or "").strip().casefold()
    if not role_key or not loc_key:
        return df.copy()

    # Exact match first, then case-insensitive substring fallback.
    role_mask = df["Job Role"].astype(str).str.casefold() == role_key
    loc_mask = df["Location"].astype(str).str.casefold() == loc_key
    filtered = df[role_mask & loc_mask]
    if not filtered.empty:
        return filtered

    role_mask = df["Job Role"].astype(str).str.casefold().str.contains(role_key, na=False)
    loc_mask = df["Location"].astype(str).str.casefold().str.contains(loc_key, na=False)
    return df[role_mask & loc_mask]


def _compute_top_skills(market_df: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame:
    if market_df.empty:
        return market_df
    return (
        market_df.groupby("Skill", as_index=False)["Frequency"]
        .sum()
        .sort_values("Frequency", ascending=False)
        .head(top_n)
    )


def _split_technical_domain(skills_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tech_keywords = [
        "python",
        "sql",
        "aws",
        "docker",
        "kubernetes",
        "terraform",
        "spark",
        "databricks",
        "pandas",
        "numpy",
        "java",
        "c++",
        "c#",
        "javascript",
        "react",
        "node",
        "ml",
        "machine learning",
        "deep learning",
        "nlp",
        "data engineering",
        "data analysis",
        "data science",
        "model development",
        "api",
        "mle",
    ]
    soft_keywords = [
        "communication",
        "collaboration",
        "stakeholder",
        "leadership",
        "mentoring",
        "problem solving",
        "critical thinking",
        "innovation",
        "agile",
        "execution",
        "strategy",
        "business alignment",
    ]

    if skills_df.empty:
        return skills_df, skills_df

    def is_soft(skill: str) -> bool:
        s = skill.casefold()
        return any(k in s for k in soft_keywords)

    def is_technical(skill: str) -> bool:
        s = skill.casefold()
        return any(k in s for k in tech_keywords)

    skills_df = skills_df.copy()
    skills_df["__is_soft"] = skills_df["Skill"].astype(str).apply(is_soft)
    skills_df["__is_tech"] = skills_df["Skill"].astype(str).apply(is_technical)

    technical = skills_df[(skills_df["__is_tech"]) & (~skills_df["__is_soft"])].drop(columns=["__is_soft", "__is_tech"])
    domain = skills_df[(~skills_df["__is_tech"]) | (skills_df["__is_soft"])].drop(columns=["__is_soft", "__is_tech"])
    return technical, domain


def _extract_resume_market_matches(
    resume_skills: List[str],
    market_skills_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    market_keys = {_normalize_key(s) for s in market_skills_df["Skill"].astype(str).tolist()}

    matched: List[str] = []
    missing: List[str] = []
    for s in resume_skills:
        key = _normalize_key(s)
        if not key:
            continue
        if key in market_keys:
            matched.append(s)
        else:
            missing.append(s)
    # Deduplicate while preserving order.
    seen: set = set()
    matched = [x for x in matched if not (x in seen or seen.add(x))]
    seen.clear()
    missing = [x for x in missing if not (x in seen or seen.add(x))]
    return matched, missing


def _compute_resume_score(matched_skills: List[str], resume_skills: List[str]) -> int:
    if not resume_skills:
        return 0
    return int(round((len(matched_skills) / max(1, len(resume_skills))) * 100))


def _make_gauge(score: int) -> go.Figure:
    score = max(0, min(100, int(score)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "rgb(26, 115, 232)"},
                "steps": [
                    {"range": [0, 50], "color": "rgb(224, 235, 255)"},
                    {"range": [50, 75], "color": "rgb(255, 242, 204)"},
                    {"range": [75, 100], "color": "rgb(227, 255, 224)"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
    return fig


def main() -> None:
    # Theme selector in the top right
    col1, col2, col3 = st.columns([1, 1, 0.3])
    with col3:
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0,
            label_visibility="collapsed",
            key="theme_selector",
        )
    
    # Apply theme styling
    if theme == "Dark":
        bg_style = "background: #1a1a1a;"
        text_color = "color: #ffffff;"
        card_bg = "rgba(40, 40, 40, 0.6)"
        card_border = "rgba(100, 100, 100, 0.4)"
    else:
        bg_style = "background: radial-gradient(1200px 600px at 10% -10%, rgba(126, 178, 255, 0.30), transparent 60%), radial-gradient(1000px 540px at 90% 0%, rgba(193, 145, 255, 0.20), transparent 55%), linear-gradient(180deg, #f3f7ff 0%, #eef2ff 45%, #e9f0ff 100%);"
        text_color = "color: #1a1f36;"
        card_bg = "rgba(255, 255, 255, 0.46)"
        card_border = "rgba(255, 255, 255, 0.72)"
    
    st.markdown(
        f"""
        <style>
          .stApp {{
            {bg_style}
          }}
          
          .stApp h1 {{
            {text_color}
            letter-spacing: 0.2px;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.55);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(1200px 600px at 10% -10%, rgba(126, 178, 255, 0.30), transparent 60%),
              radial-gradient(1000px 540px at 90% 0%, rgba(193, 145, 255, 0.20), transparent 55%),
              linear-gradient(180deg, #f3f7ff 0%, #eef2ff 45%, #e9f0ff 100%);
          }

          .stApp h1 {
            color: #1a1f36;
            letter-spacing: 0.2px;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.55);
          }

          /* iOS-like frosted controls */
          .stTextInput > div > div > input,
          .stNumberInput > div > div > input,
          .stDateInput input,
          .stFileUploader section,
          .stCheckbox {
            background: rgba(255, 255, 255, 0.40) !important;
            border: 1px solid rgba(255, 255, 255, 0.65) !important;
            border-radius: 14px !important;
            box-shadow: 0 8px 24px rgba(31, 51, 105, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px) saturate(130%);
            -webkit-backdrop-filter: blur(12px) saturate(130%);
          }

          .stButton > button {
            border-radius: 14px !important;
            border: 1px solid rgba(255, 255, 255, 0.7) !important;
            background: linear-gradient(180deg, rgba(122, 170, 255, 0.95), rgba(76, 129, 255, 0.95)) !important;
            box-shadow: 0 10px 24px rgba(52, 92, 184, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.45);
          }

          .stButton > button:hover {
            filter: brightness(1.03);
          }

          .blueBox, .yellowBox {
            background: linear-gradient(
              135deg,
              rgba(255, 255, 255, 0.46) 0%,
              rgba(255, 255, 255, 0.30) 100%
            );
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 18px;
            padding: 14px;
            box-shadow: 0 14px 32px rgba(25, 44, 94, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(16px) saturate(135%);
            -webkit-backdrop-filter: blur(16px) saturate(135%);
          }

          .boxTitle {
            font-weight: 700;
            margin-bottom: 10px;
            color: #1e2742;
          }

          /* Data tables inside glass cards */
          [data-testid="stDataFrame"],
          [data-testid="stDataEditor"] {
            background: rgba(255, 255, 255, 0.36);
            border: 1px solid rgba(255, 255, 255, 0.64);
            border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Inputs (blue): single horizontal row ---
    c_role, c_location, c_upload, c_btn = st.columns([2.1, 2.1, 2.6, 1.0], gap="small")

    with c_role:
        role = st.text_input("Enter Role", placeholder="e.g., Business Analyst", label_visibility="collapsed")

    with c_location:
        location = st.text_input("Enter location", placeholder="e.g., Texas", label_visibility="collapsed")

    # Default range_days and date filters
    range_days = 90
    from_date = None
    to_date = None

    with c_upload:
        uploaded = st.file_uploader("Upload Resume (optional)", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    with c_btn:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if not analyze_clicked:
        st.info("Enter role/location, optionally upload a resume, then click Analyze.")
        return

    if not role.strip() or not location.strip():
        st.error("Role and location are required.")
        return

    try:
        created_from = _parse_date_input(from_date)
        created_to = _parse_date_input(to_date)
        last_n_days = int(range_days) if range_days and int(range_days) > 0 else None

        # --- Fetch live jobs (Adzuna) ---
        with st.spinner("Fetching latest job listings (Adzuna)..."):
            jobs_df = _fetch_adzuna_jobs(role, location, results=40)

        jobs_df = _filter_jobs_by_date(jobs_df, created_from, created_to, last_n_days)
        jobs_df = _filter_jobs_by_title_overlap(jobs_df, role)
        if jobs_df.empty:
            st.warning(
                "No Adzuna results matched your role keywords in the selected time window. "
                "Try broadening the role text, increasing 'Last days', or disabling date filters."
            )

        jobs_df["description"] = jobs_df["description"].fillna("").astype(str)
        jobs_df["title"] = jobs_df["title"].fillna("").astype(str)

        # --- Live skills extraction (OpenAI) ---
        with st.spinner("Extracting top skills/tools from live job descriptions..."):
            combined_desc = "\n\n---\n\n".join([d for d in jobs_df["description"].tolist() if isinstance(d, str) and d.strip()])
            analysis = analyze_job_descriptions_with_openai(combined_desc)

        top_skills_raw = analysis.get("top_skills", [])
        if not isinstance(top_skills_raw, list):
            raise RuntimeError("OpenAI analysis did not return a valid 'top_skills' list.")

        skills_rows: List[Dict[str, Any]] = []
        for item in top_skills_raw:
            if not isinstance(item, dict):
                continue
            skill = item.get("skill")
            freq = item.get("frequency", 0)
            if not isinstance(skill, str) or not skill.strip():
                continue
            try:
                freq_int = int(freq)
            except (TypeError, ValueError):
                freq_int = 0

            skills_rows.append(
                {
                    "Date": date.today().isoformat(),
                    "Job Role": role.strip(),
                    "Location": location.strip(),
                    "Skill": _normalize_skill(skill),
                    "Frequency": freq_int,
                }
            )

        market_df = pd.DataFrame(skills_rows, columns=["Date", "Job Role", "Location", "Skill", "Frequency"])
        _save_latest_skills_csv(market_df, CSV_PATH)

        top_market = market_df.sort_values("Frequency", ascending=False).head(25).reset_index(drop=True)
        technical_df, domain_df = _split_technical_domain(top_market.rename(columns={"Skill": "Skill"}))
        top_technical = technical_df.head(10)
        top_domain = domain_df.head(10)

        # Resume is optional.
        resume_skills: List[str] = []
        if uploaded is not None:
            with st.spinner("Extracting skills from your resume..."):
                resume_text = extract_resume_text(uploaded)
                resume_text = (resume_text or "").strip()
                if not resume_text:
                    raise RuntimeError("Could not extract text from your resume file.")

                max_resume_chars = 25_000
                if len(resume_text) > max_resume_chars:
                    resume_text = resume_text[:max_resume_chars] + "\n[TRUNCATED]"

                resume_skills = extract_skills_from_resume(resume_text, max_skills=30)

        # Tools in demand: heuristic from top skills.
        tool_keywords = [
            "aws",
            "docker",
            "kubernetes",
            "terraform",
            "spark",
            "sql",
            "python",
            "git",
            "databricks",
            "airflow",
            "hadoop",
            "gcp",
            "azure",
            "kafka",
            "snowflake",
            "tableau",
            "power bi",
        ]

        # Fallback lists for when no data is available
        fallback_technical_skills = [
            ("Python", "N/A"),
            ("SQL", "N/A"),
            ("Java", "N/A"),
            ("C++", "N/A"),
            ("JavaScript", "N/A"),
            ("React", "N/A"),
            ("Machine Learning", "N/A"),
            ("Data Analysis", "N/A"),
            ("API Development", "N/A"),
            ("Cloud Computing", "N/A"),
        ]

        fallback_domain_skills = [
            ("Leadership", "N/A"),
            ("Problem Solving", "N/A"),
            ("Communication", "N/A"),
            ("Project Management", "N/A"),
            ("Agile", "N/A"),
            ("Collaboration", "N/A"),
            ("Critical Thinking", "N/A"),
            ("Stakeholder Management", "N/A"),
            ("Business Analysis", "N/A"),
            ("Strategy", "N/A"),
        ]

        fallback_tools = [
            ("Python", "N/A"),
            ("AWS", "N/A"),
            ("Docker", "N/A"),
            ("SQL", "N/A"),
            ("Git", "N/A"),
            ("Kubernetes", "N/A"),
            ("Azure", "N/A"),
            ("GCP", "N/A"),
            ("Terraform", "N/A"),
            ("Snowflake", "N/A"),
        ]

        def is_tool(skill: str) -> bool:
            s = skill.casefold()
            return any(k in s for k in tool_keywords)

        tools_df = top_market[top_market["Skill"].astype(str).apply(is_tool)].head(10)
        if tools_df.empty:
            tools_df = top_technical.head(10)

        # Company demand: count companies in the fetched listings.
        top_companies = (
            jobs_df.dropna(subset=["company"])
            .groupby("company", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(5)
        )

        # Extract industries from job descriptions - prioritizes business domains over generic tech keywords
        def extract_industry(title: str, description: str) -> str:
            text = f"{title} {description}".lower()
            
            # Industry keyword mappings (ordered by specificity - more specific domains checked first)
            industries = {
                "Finance": ["finance", "bank", "banking", "investment", "trading", "insurance", "fintech", "financial", "capital markets", "hedge fund", "equity", "mortgage"],
                "Healthcare": ["healthcare", "hospital", "medical", "pharma", "pharmaceutical", "health", "clinic", "nursing", "dentist", "physician", "patient care"],
                "Retail": ["retail", "e-commerce", "ecommerce", "store", "shopping", "consumer goods", "apparel", "footwear"],
                "Automotive": ["automotive", "auto", "vehicle", "car", "motor", "dealership", "tesla", "ford", "gm"],
                "Manufacturing": ["manufacturing", "factory", "production", "industrial", "supplier", "component"],
                "Logistics": ["logistics", "supply chain", "transportation", "shipping", "warehouse", "fulfillment", "distribution center"],
                "Telecommunications": ["telecom", "telecomm", "communication", "wireless", "broadband", "carrier", "spectrum"],
                "Energy": ["energy", "oil", "gas", "utilities", "power", "renewable", "renewable energy", "solar", "wind"],
                "Real Estate": ["real estate", "property", "construction", "development", "realty", "commercial real estate", "residential"],
                "Education": ["education", "university", "school", "training", "learning", "college", "edtech"],
                "Government": ["government", "federal", "state", "public sector", "agency", "civil service"],
                "Media": ["media", "entertainment", "broadcast", "publishing", "news", "streaming", "studio"],
                "Hospitality": ["hospitality", "hotel", "restaurant", "tourism", "travel", "airline", "leisure"],
                "Food & Beverage": ["food", "beverage", "restaurant", "food service", "beverage company", "restaurant chain"],
                "Technology": ["tech", "software", "saas", "startup", "cloud", "developer", "engineer", "platform"],
            }
            
            # Check each industry (order matters - specific domains checked before generic tech)
            for industry, keywords in industries.items():
                if any(keyword in text for keyword in keywords):
                    return industry
            
            return "Other"
        
        jobs_df["Industry"] = jobs_df.apply(
            lambda row: extract_industry(str(row.get("title", "")), str(row.get("description", ""))),
            axis=1
        )
        
        top_industries = (
            jobs_df.dropna(subset=["Industry"])
            .groupby("Industry", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(10)
        )

        # Resume matching is optional.
        resume_skill_keys = [_normalize_key(s) for s in resume_skills] if resume_skills else []
        resume_skill_keys = [k for k in resume_skill_keys if k]

        def job_matches(skill_keys: List[str], title: str, desc: str) -> bool:
            hay = f"{title}\n{desc}".casefold()
            return any(k in hay for k in skill_keys)

        if resume_skill_keys:
            match_mask = jobs_df.apply(
                lambda r: job_matches(resume_skill_keys, r.get("title", ""), r.get("description", "")),
                axis=1,
            )

            matching_companies_df = jobs_df[match_mask].dropna(subset=["company"])
            top_matching_companies = (
                matching_companies_df.groupby("company", as_index=False).size().sort_values("size", ascending=False).head(5)
            )

            matched_skills, missing_skills = _extract_resume_market_matches(resume_skills, top_market.rename(columns={"Skill": "Skill"}))
            resume_score = _compute_resume_score(matched_skills, resume_skills)
        else:
            top_matching_companies = pd.DataFrame(columns=["company", "size"])
            matched_skills, missing_skills, resume_score = [], [], 0

        # Extract required experience from job descriptions
        def extract_required_experience(description: str) -> str:
            if not description or not isinstance(description, str):
                return "Not mentioned"
            
            desc_lower = description.lower()
            
            # Look for patterns like "X years", "X+ years", "X-Y years"
            patterns = [
                r'(\d+)\s*\+?\s*years?\s+of\s+experience',
                r'(\d+)\s*\+?\s*years?\s+experience',
                r'experience\s+of\s+(\d+)\s*\+?\s*years?',
                r'(\d+)\s*-\s*(\d+)\s+years?\s+of\s+experience',
                r'(\d+)\s*-\s*(\d+)\s+years?\s+experience',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, desc_lower)
                if match:
                    if len(match.groups()) == 1:
                        return f"{match.group(1)}+ years"
                    else:
                        return f"{match.group(1)}-{match.group(2)} years"
            
            return "Not mentioned"
        
        # Extract job level (Senior/Junior) from title and description
        def extract_job_level(title: str, description: str) -> str:
            text = f"{title} {description}".lower()
            
            senior_keywords = ["senior", "sr.", "sr ", "lead", "principal", "architect", "director", "manager", "staff"]
            junior_keywords = ["junior", "jr.", "jr ", "entry", "entry-level", "entry level", "graduate", "intern", "trainee"]
            
            # Check for senior indicators first
            for keyword in senior_keywords:
                if keyword in text:
                    return "Senior"
            
            # Check for junior indicators
            for keyword in junior_keywords:
                if keyword in text:
                    return "Junior"
            
            # Default to Mid-level if not explicitly stated
            return "Mid-level"
        
        jobs_report = jobs_df.copy()
        for col in ["title", "company", "location", "apply_url"]:
            if col in jobs_report.columns:
                jobs_report[col] = jobs_report[col].fillna("").astype(str)

        created_dates = pd.to_datetime(jobs_report.get("created"), errors="coerce", utc=True).dt.date
        today = date.today()

        def posted_label(d: Any) -> str:
            if pd.isna(d):
                return "Unknown"
            days = max(0, (today - d).days)
            if days == 0:
                return "Today"
            if days == 1:
                return "1 day ago"
            return f"{days} days ago"

        jobs_report["Job posted"] = created_dates.apply(posted_label)
        jobs_report["Required Experience"] = jobs_report["description"].apply(extract_required_experience)
        jobs_report["Level"] = jobs_report.apply(
            lambda row: extract_job_level(str(row.get("title", "")), str(row.get("description", ""))),
            axis=1
        )
        has_resume_for_match = uploaded is not None and bool(resume_skills)
        jobs_report["MatchResults"] = [False for _ in range(len(jobs_report))]
        jobs_report["row_idx"] = list(range(len(jobs_report)))
        
        # Calculate match scores for each job when resume is uploaded
        match_scores_list = []
        if has_resume_for_match:
            market_skill_names = top_market["Skill"].astype(str).tolist()
            for idx, job_row in jobs_df.iterrows():
                job_text = f"{job_row.get('title', '')}\n{job_row.get('description', '')}".lower()
                job_required = []
                for s in market_skill_names:
                    k = _normalize_key(s)
                    if k and k in job_text:
                        job_required.append(s)
                if not job_required:
                    job_required = market_skill_names[:12]
                
                common = [s for s in job_required if _normalize_key(s) in [_normalize_key(rs) for rs in resume_skills]]
                match_pct = round((len(common) / max(1, len(job_required))) * 100) if job_required else 0
                match_scores_list.append(match_pct)
        else:
            match_scores_list = [0] * len(jobs_df)
        
        jobs_report["match_score"] = match_scores_list
        jobs_report = jobs_report.rename(
            columns={
                "title": "Job title",
                "company": "Company",
                "location": "Location",
                "apply_url": "Application URL",
            }
        )
        report_cols = [
            c
            for c in ["Job title", "Company", "Location", "Level", "Job posted", "Required Experience", "Application URL", "MatchResults", "row_idx"]
            if c in jobs_report.columns
        ]

        # Generate and display market summary
        st.markdown("---")
        total_jobs = len(jobs_df)
        top_skill_names = top_market["Skill"].head(5).tolist() if not top_market.empty else []
        top_skills_text = ", ".join(top_skill_names) if top_skill_names else "various domain skills"
        top_tool_names = tools_df["Skill"].head(5).tolist() if not tools_df.empty else []
        top_tools_text = ", ".join(top_tool_names) if top_tool_names else "industry-standard tools"
        top_industry = top_industries["Industry"].iloc[0] if not top_industries.empty else "Technology"
        top_industry_count = top_industries["size"].iloc[0] if not top_industries.empty else 0
        industry_pct = round((top_industry_count / total_jobs * 100)) if total_jobs > 0 else 0
        exp_data = jobs_report["Required Experience"].value_counts()
        most_common_exp = exp_data.index[0] if len(exp_data) > 0 else "unspecified years"
        level_data = jobs_report["Level"].value_counts()
        senior_count = int(level_data.get("Senior", 0))
        junior_count = int(level_data.get("Junior", 0))
        mid_count = int(level_data.get("Mid-level", 0))
        top_company = top_companies["company"].iloc[0] if not top_companies.empty else "the market leaders"
        
        summary_md = f"**Market Overview for {role.strip()} in {location.strip()}**\n\n"
        summary_md += f"Our analysis examined **{total_jobs} active job openings** in your target market. Here are the key insights:\n\n"
        summary_md += f"**Domain Expertise in Demand:** Organizations are seeking professionals with expertise in {top_skills_text}. These are foundational competencies that appear repeatedly in job descriptions and are essential for competitive candidacy in this market segment.\n\n"
        summary_md += f"**Technology & Tools Stack:** The most in-demand tools and technologies include {top_tools_text}. Strong proficiency in these platforms is highly valued and significantly enhances your marketability.\n\n"
        summary_md += f"**Industry Hotspot:** The **{top_industry}** sector dominates this talent market with {top_industry_count} openings ({industry_pct}% of all positions). This indicates where the strongest hiring momentum and career growth opportunities currently exist.\n\n"
        summary_md += f"**Experience Profile:** The majority of roles require {most_common_exp}, suggesting a mature market that accommodates both seasoned professionals seeking new challenges and motivated candidates looking to advance their careers.\n\n"
        summary_md += f"**Seniority Mix:** The market offers diverse opportunities across experience levels: {senior_count} Senior positions, {mid_count} Mid-level roles, and {junior_count} Junior opportunities. This diversity indicates healthy organizational growth with opportunities for career progression at all stages.\n\n"
        summary_md += f"**Hiring Hubs:** {top_company} and other major employers are actively hiring, suggesting consolidation around established industry leaders with strong growth trajectories.\n\n"
        summary_md += "**Market Verdict:** This is a vibrant market with clearly defined skill requirements and multiple pathways for career advancement. The presence of opportunities across all seniority levels indicates strong organizational maturity and investment in workforce development."
        
        with st.expander("📊 Market Overview & Trends", expanded=False):
            st.markdown(summary_md)
        st.markdown("---")

        # Resume vs Trend Analysis (only if resume was uploaded)
        if has_resume_for_match:
            with st.expander("👤 My Resume vs the Trend", expanded=False):
                st.markdown(f"### Your Profile Analysis Against Market Demands")
                
                # Display resume score gauge
                col_gauge, col_summary = st.columns([1, 2])
                with col_gauge:
                    gauge_fig = _make_gauge(resume_score)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col_summary:
                    st.markdown(f"**Your Match Score: {resume_score}%**")
                    if resume_score >= 80:
                        st.success("🟢 You're well-aligned with market demands! Your skills closely match what employers are seeking.")
                    elif resume_score >= 60:
                        st.info("🟡 Good alignment! You have many key skills, but could strengthen your profile with a few in-demand areas.")
                    elif resume_score >= 40:
                        st.warning("🟠 Moderate alignment. There are several in-demand skills worth developing to improve your competitiveness.")
                    else:
                        st.warning("🔴 Significant opportunity to grow! Focus on the high-demand skills listed below to boost your market fit.")
                    
                    st.markdown(f"**Total Resume Skills:** {len(resume_skills)}")
                    st.markdown(f"**Market-Aligned Skills:** {len(matched_skills)}")
                    st.markdown(f"**Recommended Skills Gap:** {len(missing_skills)}")
                
                st.markdown("---")
                
                # Three-column layout for matched, missing, and certifications
                col1, col2, col3 = st.columns(3, gap="large")
                
                with col1:
                    st.markdown("<div class='yellowBox'><div class='boxTitle'>✓ Skills You Have (In Demand)</div>", unsafe_allow_html=True)
                    if matched_skills:
                        matched_df = pd.DataFrame(
                            [{"Skill": s} for s in matched_skills],
                            columns=["Skill"]
                        )
                        st.dataframe(
                            matched_df,
                            use_container_width=True,
                            hide_index=True,
                            height=250,
                        )
                    else:
                        st.write("_No resume uploaded or no matching skills detected yet._")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='yellowBox'><div class='boxTitle'>⚠️ Skills to Develop (In Demand)</div>", unsafe_allow_html=True)
                    if missing_skills:
                        missing_df = pd.DataFrame(
                            [{"Skill": s} for s in missing_skills],
                            columns=["Skill"]
                        )
                        st.dataframe(
                            missing_df,
                            use_container_width=True,
                            hide_index=True,
                            height=250,
                        )
                    else:
                        st.success("All top in-demand skills are on your resume! 🎉")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='yellowBox'><div class='boxTitle'>📚 Certifications & Tools in Demand</div>", unsafe_allow_html=True)
                    # Extract tools that appear in top demand but not on resume
                    tools_in_demand = []
                    if not tools_df.empty:
                        for tool in tools_df["Skill"].head(8).tolist():
                            tool_key = _normalize_key(tool)
                            if tool_key and tool_key not in [_normalize_key(s) for s in resume_skills]:
                                tools_in_demand.append(tool)
                    
                    # Also add frequently mentioned skills not on resume
                    if not tools_in_demand and top_technical is not None and not top_technical.empty:
                        for tool in top_technical["Skill"].head(5).tolist():
                            tool_key = _normalize_key(tool)
                            if tool_key and tool_key not in [_normalize_key(s) for s in resume_skills]:
                                tools_in_demand.append(tool)
                    
                    if tools_in_demand:
                        tools_df_display = pd.DataFrame(
                            [{"Tool/Cert": t} for t in tools_in_demand[:8]],
                            columns=["Tool/Cert"]
                        )
                        st.dataframe(
                            tools_df_display,
                            use_container_width=True,
                            hide_index=True,
                            height=250,
                        )
                    else:
                        st.write("You have the key tools covered! ✅")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommendations section
                st.markdown("---")
                st.markdown("### 🎯 Personalized Learning Recommendations")
                
                recommendations = []
                
                # High priority recommendations
                if missing_skills:
                    top_missing = missing_skills[:3]
                    recommendations.append(f"**Priority 1 - Core Skills:** Focus on mastering {', '.join(top_missing)}. These are the most frequently mentioned requirements across all {total_jobs} job openings.")
                
                if tools_in_demand:
                    top_tools = tools_in_demand[:3]
                    recommendations.append(f"**Priority 2 - Technology Stack:** Get hands-on experience with {', '.join(top_tools)}. These tools appear in {industry_pct}% of the top {top_industry} sector openings.")
                
                if resume_score < 60:
                    recommendations.append(f"**Priority 3 - Skill Diversity:** Your current skill set covers {resume_score}% of market demands. Consider expanding into complementary areas like system design, cloud platforms, or leadership skills depending on your career goals.")
                
                if resume_score >= 80:
                    recommendations.append(f"**Consolidation:** Your profile is strong (${resume_score}% match). Focus on deepening expertise in your matched areas and consider leadership/architectural skills to advance to senior roles.")
                
                if top_matching_companies.empty:
                    recommendations.append("**Company Targeting:** Expand your skill search to include adjacent roles or industries to increase the number of companies actively hiring in your target area.")
                else:
                    top_company_names = top_matching_companies["company"].head(3).tolist() if not top_matching_companies.empty else []
                    if top_company_names:
                        recommendations.append(f"**Target Companies:** Companies like {', '.join(top_company_names[:2])} are actively hiring and your skills align well. Research their tech stacks and engineering practices.")
                
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(rec)
                
                st.markdown("---")
                st.markdown(f"**Certification Opportunities:** Based on the {top_industry} sector's demands, consider pursuing certifications in:")
                
                cert_recommendations = []
                if any(tool in [_normalize_key(t) for t in tools_in_demand] for tool in ["aws", "azure", "gcp", "cloud"]):
                    cert_recommendations.append("☁️ Cloud Platforms (AWS Solutions Architect, Azure Administrator, GCP Associate)")
                if any(skill in [_normalize_key(s) for s in missing_skills] for skill in ["machine learning", "ml", "data science"]):
                    cert_recommendations.append("🤖 Machine Learning (Google ML Engineer, AWS ML Specialty, or Andrew Ng's ML courses)")
                if any(skill in [_normalize_key(s) for s in missing_skills] for skill in ["kubernetes", "docker", "devops"]):
                    cert_recommendations.append("🐳 DevOps & Containerization (Kubernetes CKA, Docker Associate)")
                if any(skill in [_normalize_key(s) for s in missing_skills] for skill in ["sql", "database"]):
                    cert_recommendations.append("📊 Data & Databases (SQL performance tuning, Snowflake, BigQuery certifications)")
                if not cert_recommendations:
                    cert_recommendations.append(f"💼 Leadership & Management (to advance in the {top_industry} sector)")
                
                for cert in cert_recommendations[:4]:
                    st.markdown(f"- {cert}")
            
            st.markdown("---")

        # Full-width table so links are visible without being cramped by side columns.
        st.markdown("<div class='yellowBox'><div class='boxTitle'>Latest jobs and application links</div>", unsafe_allow_html=True)
        
        if jobs_report.empty:
            st.write("No data.")
        else:
            edited_jobs = st.data_editor(
                jobs_report.loc[:, report_cols].head(20),
                use_container_width=True,
                hide_index=True,
                height=280,
                disabled=[
                    "Job title",
                    "Company",
                    "Location",
                    "Level",
                    "Job posted",
                    "Required Experience",
                    "Application URL",
                    "row_idx",
                ]
                + ([] if has_resume_for_match else ["MatchResults"]),
                column_config={
                    "Application URL": st.column_config.LinkColumn(
                        "Application URL",
                        help="Open the job posting",
                        display_text="Job",
                    ),
                    "MatchResults": st.column_config.CheckboxColumn(
                        "MatchResults",
                        help="✓ Check to see detailed match analysis: skills overlap, gaps, and learning recommendations",
                    ),
                    "row_idx": None,
                },
                key="jobs_report_editor",
            )
            if not has_resume_for_match:
                st.caption("Upload a resume to enable MatchResults.")
            else:
                selected_rows = edited_jobs[edited_jobs["MatchResults"] == True]  # noqa: E712
                if not selected_rows.empty:
                    selected_idx = int(selected_rows.iloc[0]["row_idx"])
                    if 0 <= selected_idx < len(jobs_df):
                        selected_job = jobs_df.iloc[selected_idx]

                        resume_key_to_name: Dict[str, str] = {}
                        for s in resume_skills:
                            k = _normalize_key(s)
                            if k and k not in resume_key_to_name:
                                resume_key_to_name[k] = s

                        job_text = f"{selected_job.get('title', '')}\n{selected_job.get('description', '')}"
                        job_text_key = _normalize_key(job_text)

                        market_skill_names = top_market["Skill"].astype(str).tolist()
                        job_required: List[str] = []
                        for s in market_skill_names:
                            k = _normalize_key(s)
                            if k and k in job_text_key:
                                job_required.append(s)
                        if not job_required:
                            job_required = market_skill_names[:12]

                        common = [s for s in job_required if _normalize_key(s) in resume_key_to_name][:12]
                        missing = [s for s in job_required if _normalize_key(s) not in resume_key_to_name][:12]
                        tool_candidates = tools_df["Skill"].astype(str).tolist() if not tools_df.empty else []
                        tools_to_learn = [t for t in tool_candidates if _normalize_key(t) not in resume_key_to_name][:8]

                        job_title = str(selected_job.get("title", "") or "Selected job")
                        job_company = str(selected_job.get("company", "") or "Unknown company")
                        job_location = str(selected_job.get("location", "") or "Unknown location")
                        match_score = jobs_report.iloc[selected_idx]["match_score"] if selected_idx < len(jobs_report) else 0
                        
                        # Determine match level label
                        if match_score >= 80:
                            match_label = "🟢 Excellent Match"
                        elif match_score >= 60:
                            match_label = "🟡 Good Match"
                        elif match_score >= 40:
                            match_label = "🟠 Fair Match"
                        else:
                            match_label = "🔴 Growth Opportunity"

                        # Show match summary inline below the table
                        st.markdown("---")
                        st.markdown(f"### {match_label} - {match_score}% Match")
                        st.markdown(f"**{job_title}** at **{job_company}** | {job_location}")
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**✓ Skills You Have**")
                            st.write(", ".join(common) if common else "None of the top required skills found in your profile yet.")
                        with col2:
                            st.markdown("**⚠️ Skills to Develop**")
                            st.write(", ".join(missing) if missing else "You have all the key required skills!")
                        
                        st.markdown("**📚 Tools to Learn**")
                        st.write(", ".join(tools_to_learn) if tools_to_learn else "No additional tool recommendations right now.")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Output layout (yellow) - 3 tables in one row ---
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("<div class='yellowBox'><div class='boxTitle'>Top Domain skills</div>", unsafe_allow_html=True)
            if top_domain.empty:
                fallback_df = pd.DataFrame(fallback_domain_skills, columns=["Skill", "Frequency"])
                st.dataframe(
                    fallback_df.rename(columns={"Frequency": "Mentions"}),
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )
            else:
                st.dataframe(
                    top_domain.loc[:, ["Skill", "Frequency"]]
                    .rename(columns={"Frequency": "Mentions"})
                    .sort_values("Mentions", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='yellowBox'><div class='boxTitle'>Top companies hiring</div>", unsafe_allow_html=True)
            if top_companies.empty:
                st.write("No data.")
            else:
                st.dataframe(
                    top_companies.rename(columns={"company": "Company", "size": "Openings"})
                    .loc[:, ["Company", "Openings"]]
                    .sort_values("Openings", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='yellowBox'><div class='boxTitle'>Top Industries</div>", unsafe_allow_html=True)
            if top_industries.empty:
                st.write("No data.")
            else:
                st.dataframe(
                    top_industries.rename(columns={"Industry": "Industry", "size": "Openings"})
                    .loc[:, ["Industry", "Openings"]]
                    .sort_values("Openings", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Mentions shows how often a domain skill appears in analyzed job descriptions.")

        # Small debug info
        with st.expander("Debug / assumptions", expanded=False):
            location_samples = (
                jobs_df["location"].dropna().astype(str).head(10).tolist()
                if "location" in jobs_df.columns
                else []
            )
            st.write(
                {
                    "skills_data_csv_overwritten": True,
                    "query_role": role.strip(),
                    "query_location": location.strip(),
                    "adzuna_rows_used": int(len(jobs_df)),
                    "adzuna_location_samples": location_samples,
                    "resume_uploaded": uploaded is not None,
                    "resume_skills_extracted": resume_skills[:20] if uploaded is not None else [],
                    "resume_skills_count": len(resume_skills) if uploaded is not None else 0,
                }
            )

    except Exception as e:
        st.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
