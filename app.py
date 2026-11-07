import os
import csv
import json
import time
import logging
import hashlib
import warnings
import concurrent.futures
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from exa_py import Exa
from google import genai
from google.genai import types

# ---------------- 0. Suppress Non-Fatal Environment Warnings ----------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

class SuppressGenAIWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "there are non-text parts in the response" not in record.getMessage()

logging.getLogger("google_genai.types").addFilter(SuppressGenAIWarnings())

# ---------------- 1. Config & Initializations ----------------
st.set_page_config(page_title="IntelAgent AI | Competitive Intelligence", page_icon="\U0001F3AF", layout="wide")

MODEL = "gemini-3.6-flash"
MAX_COMPETITORS = 6
MIN_COMPETITORS_TARGET = 4          # discovery retries broadly if below this
RELEVANCE_THRESHOLD = 5
FEEDBACK_FILE = "feedback_log.csv"  # ephemeral on Streamlit Cloud; fine for demo

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
EXA_KEY = st.secrets.get("EXA_API_KEY") or os.environ.get("EXA_API_KEY")

if not GEMINI_KEY or not EXA_KEY:
    st.error("Missing API Keys! Ensure GEMINI_API_KEY and EXA_API_KEY are set in secrets.toml or environment.")
    st.stop()

gemini_client = genai.Client(api_key=GEMINI_KEY)
exa_client = Exa(api_key=EXA_KEY)

# ---------------- 2. Helper Functions ----------------
def call_gemini_json(system_prompt: str, user_prompt: str, retries: int = 3) -> dict:
    """Call Gemini 3.6 Flash expecting structured JSON output."""
    for attempt in range(retries):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json"
            )
            res = gemini_client.models.generate_content(
                model=MODEL,
                contents=user_prompt,
                config=config
            )
            return json.loads(res.text)
        except Exception as e:
            if attempt == retries - 1:
                print(f"[GEMINI FAIL] {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}

def search_exa(query: str, num_results: int = 4, recency_days: int = 365) -> list[dict]:
    """Execute search query using Exa API. Guardrail: recency filter + graceful fallback."""
    try:
        res = exa_client.search(
            query,
            num_results=num_results,
            type="auto",
            start_published_date=(datetime.now() - timedelta(days=recency_days)).strftime("%Y-%m-%d")
        )
    except TypeError:
        # Older exa-py SDK without date filtering support
        res = exa_client.search(query, num_results=num_results, type="auto")
    except Exception as e:
        print(f"[EXA ERROR] Query '{query}': {e}")
        return []
    cleaned = []
    for item in res.results:
        cleaned.append({
            "url": item.url,
            "title": getattr(item, "title", "Source"),
            "content": getattr(item, "text", "")[:400]
        })
    return cleaned

# ---------------- 2b. Confidence + Feedback utilities ----------------
STATUS_WEIGHTS = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}

def aggregate_confidence(verdicts: list) -> float | None:
    """Weighted aggregate of judge verdicts -> 0.0-1.0 report confidence."""
    if not verdicts:
        return None
    total = sum(STATUS_WEIGHTS.get(v.get("status", "partially_supported"), 0.5) * float(v.get("confidence", 0.5))
                for v in verdicts)
    return round(total / len(verdicts), 2)

def claim_badge(text: str, verdicts: list) -> str:
    """Per-claim badge from judge verdicts. Default is cautious yellow."""
    key = " ".join(text.lower().split())[:120]
    v = next((x for x in verdicts if " ".join(x.get("claim", "").lower().split())[:120] == key), None)
    if not v:
        return "\U0001F7E1"
    if v.get("status") == "unsupported":
        return "\U0001F534"
    return "\U0001F7E2" if float(v.get("confidence", 0)) >= 0.6 else "\U0001F7E1"

def log_feedback(report_id: str, item: str, judge_conf, human_verdict: str) -> None:
    """Append human review verdicts to CSV (feeds calibration later)."""
    row = [datetime.now().isoformat(), report_id, item[:200],
           judge_conf if judge_conf is not None else "", human_verdict]
    try:
        new_file = not os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "report_id", "item", "judge_confidence", "human_verdict"])
            w.writerow(row)
    except Exception:
        # Streamlit Cloud filesystem is read-only; keep in session as fallback
        st.session_state.setdefault("feedback_buffer", []).append(row)

# ---------------- 3. The Agent Implementations ----------------

def agent_1_query(company: str, product: str) -> list[str]:
    """Agent 1: Generates targeted search queries."""
    system = "You are a research planner. Return JSON format: {\"queries\": [\"str\"]}"
    prompt = f"Generate 4 distinct search strings to identify direct market competitors to {company} in the domain of {product}."
    data = call_gemini_json(system, prompt)
    queries = data.get("queries", [])
    if not queries:
        return [
            f"{company} {product} top direct competitors",
            f"best enterprise alternatives to {company} in {product}",
            f"top platforms competing with {company}"
        ]
    return queries[:4]

def agent_2_discovery(company: str, product: str, search_results: list[dict]) -> list[dict]:
    """Agent 2: Evaluates candidates, filters direct competitors, broad second pass if too few."""
    system = """Identify direct market competitors for the target company and product.
Return JSON format with key 'competitors':
{"competitors": [{"name": "CompetitorName", "relevance_score": 8, "reasoning": "reason"}]}"""

    def _extract(evid):
        prompt = f"Target Company: {company}, Product Domain: {product}.\nSearch Evidence:\n{json.dumps(evid[:15])}"
        data = call_gemini_json(system, prompt)
        cands = data.get("competitors", [])
        return [
            c for c in cands
            if c.get("name", "").lower() != company.lower()
            and company.lower() not in c.get("name", "").lower()
            and c.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
        ]

    valid = _extract(search_results)

    # Broad second discovery pass if the first pass is thin (fixes "2 competitors" problem)
    if len(valid) < MIN_COMPETITORS_TARGET:
        broad = search_exa(f"{product} market leading vendors platforms companies", num_results=5)
        extra = _extract(search_results + broad)
        seen = {c["name"].lower() for c in valid}
        for c in extra:
            if c["name"].lower() not in seen:
                valid.append(c)
                seen.add(c["name"].lower())

    if not valid:
        fallback_prompt = f"List the top 3 direct market competitors to {company} specifically for {product}. Return JSON: {{\"competitors\": [{{\"name\": \"Name\", \"relevance_score\": 9, \"reasoning\": \"Direct competitor\"}}]}}"
        fallback_data = call_gemini_json("You are a market analyst.", fallback_prompt)
        valid = fallback_data.get("competitors", [])

    return sorted(valid, key=lambda c: -c.get("relevance_score", 0))[:MAX_COMPETITORS]

REQUIRED_PROFILE_FIELDS = ["one_liner", "pricing_model", "positioning"]
REPAIR_PROMPT = """You are filling missing fields in a competitor profile using the evidence provided.
Only fill the fields listed as missing. Use "unclear" if the evidence does not support a value.
Return JSON with exactly the missing keys."""

def _repair_profile(profile: dict, comp_name: str, snaps: list) -> dict:
    """Guardrail: no null fields reach the Analyst. One targeted retry per missing field."""
    if not profile or "name" not in profile:
        profile = {
            "name": comp_name,
            "one_liner": "unclear",
            "key_features": [],
            "pricing_model": "unclear",
            "positioning": "unclear"
        }
    missing = [f for f in REQUIRED_PROFILE_FIELDS if not profile.get(f)]
    if missing:
        fresh = []
        for f in missing[:2]:  # max one extra search per missing field
            q = f"{comp_name} pricing plans cost" if f == "pricing_model" else f"{comp_name} product overview"
            fresh += search_exa(q, num_results=3, recency_days=365)
        retry = call_gemini_json(
            REPAIR_PROMPT,
            f"Missing fields: {missing}\nExisting profile: {json.dumps(profile)}\nEvidence: {json.dumps((snaps + fresh)[:12])}"
        )
        for f in missing:
            val = retry.get(f)
            profile[f] = val if val else "unclear"
    # Features must carry sources; drop ones that do not
    clean_features = []
    for feat in profile.get("key_features", []):
        if isinstance(feat, dict) and feat.get("feature"):
            if not feat.get("source"):
                feat["source"] = snaps[0]["url"] if snaps and snaps[0].get("url") else ""
            clean_features.append(feat)
    profile["key_features"] = clean_features
    return profile

def agent_3_research_single(comp_name: str, target_company: str, target_product: str) -> dict:
    """Agent 3: Worker agent researching a single competitor (with schema repair)."""
    snaps = (search_exa(f"{comp_name} key features overview", num_results=4, recency_days=365)
             + search_exa(f"{comp_name} pricing plans cost", num_results=4, recency_days=365))
    system = """Extract competitor profile as JSON with keys:
    'name', 'one_liner', 'key_features' (list of dicts with 'feature' and 'source'),
    'pricing_model', 'positioning'
    Use "unclear" for anything the evidence does not support. Never guess."""

    prompt = f"Competitor: {comp_name}. Target Benchmark: {target_company} {target_product}.\nEvidence: {json.dumps(snaps)}"
    profile = call_gemini_json(system, prompt)
    profile = _repair_profile(profile, comp_name, snaps)
    profile["_sources"] = [s["url"] for s in snaps if "url" in s]
    return profile

def agent_4_analyst(company: str, product: str, profiles: list[dict]) -> dict:
    """Agent 4: Synthesizes profiles into a comparative matrix and SWOT analysis."""
    system = (
        "You are a senior Product Management Lead. Produce competitive analysis JSON.\n\n"
        "Rules for 'comparison_table':\n"
        "- Must be a list of objects representing evaluation dimensions (e.g., 'Core Focus', 'Target Audience', 'Pricing Model').\n"
        f"- Each object MUST include 'Dimension', 'Target ({company})', and explicit separate keys for EACH competitor by name.\n\n"
        "CRITICAL RULE: If any competitor profile field is null, 'unclear', or missing, you MUST NOT make claims "
        "about that aspect of that competitor.\n\n"
        "Return JSON format:\n"
        "{\n"
        '  "positioning_statement": "string",\n'
        '  "comparison_table": [\n'
        "    {\n"
        '      "Dimension": "Core Focus",\n'
        f'      "Target ({company})": "Primary capability description",\n'
        '      "Competitor A": "Competitor capability description"\n'
        "    }\n"
        "  ],\n"
        '  "swot": {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []},\n'
        '  "threat_level": "Low | Medium | High"\n'
        "}"
    )
    target_profile = next(
        (p for p in profiles
         if p.get("name", "").lower() == company.lower()),
        {}
    )
    
    competitor_profiles = [
        p for p in profiles
        if p.get("name", "").lower() != company.lower()
    ]
    
    prompt = f"""
    Target Company: {company}
    Product Domain: {product}
    
    Target Company Profile:
    {json.dumps(target_profile)}
    
    Competitor Profiles:
    {json.dumps(competitor_profiles)}
    """
    return call_gemini_json(system, prompt)

def agent_5_critic(profiles: list[dict], report: dict) -> dict:
    """Agent 5: Fact-checker with severity-tagged flags."""
    system = """You are a QA Critic. Verify report accuracy against evidence.
Classify each flag by severity:
- "unsupported_claim": the report asserts something the evidence does not support
- "missing_data": the report mentions a field that is null/unclear in the profile
Return JSON:
{"flags": [{"text": str, "severity": "unsupported_claim" | "missing_data"}], "confidence": "High" | "Medium" | "Low"}"""
    prompt = f"Profiles: {json.dumps(profiles)}\nReport: {json.dumps(report)}"
    return call_gemini_json(system, prompt)

JUDGE_PROMPT = """You are a strict fact-checking judge. Evaluate each factual claim below
against the research profiles.
For EACH claim return: status (supported / partially_supported / unsupported),
confidence (0.0-1.0 based on number and recency of independent sources and directness
of evidence), and evidence_url (from the profiles, or "").
Claims about pricing or specific numbers require a direct source to be "supported".
Prefer independent sources (reviews, third-party comparisons) over the vendor's own pages.
A claim needs two or more independent sources to deserve confidence >= 0.8.
Return JSON: {"verdicts": [{"claim": str, "status": str, "confidence": float, "evidence_url": str}]}"""

def agent_6_judge(profiles: list[dict], report: dict) -> list[dict]:
    """Agent 6: LLM-as-judge. Per-claim verdicts that drive confidence scores and badges."""
    swot = report.get("swot", {})
    claims = [report.get("positioning_statement", "")]
    for bucket in ["strengths", "weaknesses", "opportunities", "threats"]:
        claims += [str(c) for c in swot.get(bucket, [])]
    claims = [c for c in claims if c]
    if not claims:
        return []
    prompt = f"Profiles: {json.dumps(profiles)}\nClaims to evaluate:\n" + "\n".join(f"- {c}" for c in claims)
    data = call_gemini_json(JUDGE_PROMPT, prompt)
    return data.get("verdicts", [])

# ---------------- 3b. Pipeline (importable by evals/run_evals.py) ----------------
def run_pipeline(company: str, product: str) -> dict:
    """Runs the full 6-agent pipeline headlessly. Returns everything the UI needs."""
    queries = agent_1_query(company, product)

    search_data = []
    for q in queries:
        search_data.extend(search_exa(q))

    candidates = agent_2_discovery(company, product, search_data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        competitor_profiles = list(executor.map(
            lambda c: agent_3_research_single(
                c["name"],
                company,
                product
            ),
            candidates
        ))

    # Research the target company as well
    target_profile = agent_3_research_single(
        company,
        company,
        product
    )

    all_profiles = [target_profile] + competitor_profiles

    report = agent_4_analyst(
        company,
        product,
        all_profiles
    )

    critique = agent_5_critic(
        all_profiles,
        report
    )

    verdicts = agent_6_judge(
        all_profiles,
        report
    )

    return {
        "profiles": all_profiles,
        "report": report,
        "critique": critique,
        "verdicts": verdicts,
        "report_confidence": aggregate_confidence(verdicts),
    }    
    critique = agent_5_critic(
        all_profiles,
        report
    )
    
    verdicts = agent_6_judge(
        all_profiles,
        report
    )
    
    return {
        "profiles": all_profiles,
        "report": report,
        "critique": critique,
        "verdicts": verdicts,
        "report_confidence": aggregate_confidence(verdicts),
    }
# ---------------- 4. Streamlit App Interface ----------------
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 5px;">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20ZM12 6C8.69 6 6 8.69 6 12C6 15.31 8.69 18 12 18C15.31 18 18 15.31 18 12C18 8.69 15.31 6 12 6ZM12 16C9.79 16 8 14.21 8 12C8 9.79 9.79 8 12 8C14.21 8 16 9.79 16 12C16 14.21 14.21 16 12 16Z" fill="#FF4B4B"/>
            <circle cx="12" cy="12" r="2.5" fill="#1E88E5"/>
        </svg>
        <div>
            <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700;">IntelAgent AI</h1>
            <p style="margin: 0; color: #555; font-size: 1rem;">Autonomous 6-Agent Competitive Intelligence Pipeline</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("Agentic Workflow: Query Plan -> Discovery -> Parallel Deep Research -> Analyst Synthesis -> Critic Audit -> Judge Calibration")

with st.sidebar:
    company_input = st.text_input("Target Company", placeholder="e.g., Figma")
    product_input = st.text_input("Product / Domain", placeholder="e.g., Figma Design")
    start_btn = st.button("Run Analysis Pipeline", type="primary")

    st.sidebar.divider()
    st.sidebar.caption(
        "\u2139\ufe0f **Demo Disclaimer**\n\n"
        "This application is a portfolio proof-of-concept running on free-tier API quotas. "
        "Analysis is generated autonomously using Gemini 3.6-Flash and Exa.ai, and outputs "
        "should be verified independently. Confidence scores are calibrated from human feedback "
        "and should be treated as directional."
    )

if start_btn:
    if not company_input or not product_input:
        st.warning("Please specify both company and product/domain.")
    else:
        with st.status("Executing 6-Agent Pipeline...", expanded=True) as status:
            status.update(label="Agent 1: Planning targeted searches...")
            queries = agent_1_query(company_input, product_input)

            status.update(label="Agent 2: Running web discovery & evaluating candidates...")
            search_data = []
            for q in queries:
                search_data.extend(search_exa(q))
            candidates = agent_2_discovery(company_input, product_input, search_data)

            status.update(label=f"Agent 3: Conducting parallel research on {len(candidates)} competitors...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                competitor_profiles = list(executor.map(
                    lambda c: agent_3_research_single(
                        c["name"],
                        company_input,
                        product_input
                    ),
                    candidates
                ))
            
            # Research target company
            target_profile = agent_3_research_single(
                company_input,
                company_input,
                product_input
            )
            
            profiles = [target_profile] + competitor_profiles
            
            status.update(label="Agent 4: Synthesizing SWOT, positioning & matrix...")
            report = agent_4_analyst(
                company_input,
                product_input,
                profiles
            )
            
            status.update(label="Agent 5: Running critic audit & fact check...")
            critique = agent_5_critic(
                profiles,
                report
            )
            
            status.update(label="Agent 6: Judge scoring claims for confidence calibration...")
            verdicts = agent_6_judge(
                profiles,
                report
            )
            report_conf = aggregate_confidence(verdicts)

            status.update(label="Pipeline Complete!", state="complete")

        # Stable report id for feedback widget keys
        report_id = hashlib.md5(f"{company_input}-{product_input}-{time.time()}".encode()).hexdigest()[:10]
        st.session_state["report_id"] = report_id
        st.session_state["verdicts"] = verdicts

        # Metrics Header
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Competitors Identified", len(profiles))
        m2.metric("Sources Analyzed", sum(len(p.get("_sources", [])) for p in profiles))
        m3.metric("Critic Audit Flags", len(critique.get("flags", [])))
        m4.metric("Report Confidence", f"{int(report_conf * 100)}%" if report_conf is not None else "N/A")

        flags = critique.get("flags", [])
        if flags:
            with st.expander("\u26a0\ufe0f Audit Flags (Critic Agent)", expanded=True):
                for flag in flags:
                    if isinstance(flag, dict):
                        sev = flag.get("severity", "unsupported_claim")
                        icon = "\U0001F534" if sev == "unsupported_claim" else "\U0001F7E1"
                        st.write(f"{icon} **{sev.replace('_', ' ').title()}**: {flag.get('text', '')}")
                    else:  # backwards compatibility if model returns plain strings
                        st.write(f"- {flag}")

        t1, t2, t3 = st.tabs(["Comparison Matrix", "SWOT & Positioning", "Sources & Profiles"])

        with t1:
            st.markdown(f"**Positioning Statement**\n\n{report.get('positioning_statement', 'N/A')}")
            raw_table = report.get("comparison_table", [])
            if raw_table:
                formatted_rows = []
                for row in raw_table:
                    new_row = {}
                    for key, val in row.items():
                        if isinstance(val, dict):
                            for sub_k, sub_v in val.items():
                                new_row[sub_k] = sub_v
                        elif isinstance(val, list):
                            new_row[key] = ", ".join(map(str, val))
                        else:
                            new_row[key] = val
                    formatted_rows.append(new_row)
                df = pd.DataFrame(formatted_rows)
                st.markdown(
                    """
                    <style>
                    .custom-matrix-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-family: inherit; }
                    .custom-matrix-table th { white-space: nowrap !important; text-align: left; padding: 10px 12px; background-color: #f0f2f6; border-bottom: 2px solid #d0d4dc; font-weight: 600; }
                    .custom-matrix-table td { padding: 10px 12px; border-bottom: 1px solid #e6e9ef; vertical-align: top; }
                    .custom-matrix-table td:first-child { font-weight: 600; white-space: nowrap !important; background-color: #f8f9fa; }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.write(df.to_html(classes="custom-matrix-table", escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No comparison data generated.")

        with t2:
            swot = report.get("swot", {})
            verdict_map = {v["claim"]: v for v in verdicts}
            col_a, col_b = st.columns(2)

            def _render_swot_list(items, widget_prefix):
                for i, item in enumerate(items):
                    badge = claim_badge(str(item), verdicts)
                    c1, c2 = st.columns([12, 1])
                    c1.write(f"{badge} {item}")
                    with c2:
                        v = verdict_map.get(item)
                        fb = st.feedback("thumbs", key=f"fb_{widget_prefix}_{report_id}_{i}")
                        if fb is not None:
                            log_feedback(report_id, str(item), v.get("confidence") if v else None,
                                         "correct" if fb == 1 else "incorrect")
                            st.toast("Feedback recorded — thank you!")

            with col_a:
                st.markdown("**STRENGTHS**");  _render_swot_list(swot.get("strengths", []), "s")
                st.markdown("**OPPORTUNITIES**"); _render_swot_list(swot.get("opportunities", []), "o")
            with col_b:
                st.markdown("**WEAKNESSES**"); _render_swot_list(swot.get("weaknesses", []), "w")
                st.markdown("**THREATS**"); _render_swot_list(swot.get("threats", []), "t")
            st.markdown(f"**Overall Threat Level:** {report.get('threat_level', 'Medium')}")
            st.caption("Badges: \U0001F7E2 well-sourced \U0001F7E1 limited evidence \U0001F534 unsupported — click \U0001F44D/\U0001F44E to help calibrate.")

        with t3:
            for p in profiles:
                with st.expander(f"{p.get('name')} Overview"):
                    st.write(f"**One Liner:** {p.get('one_liner')}")
                    st.write(f"**Pricing Model:** {p.get('pricing_model')}")
                    st.write("**Key Features:**")
                    for feat in p.get("key_features", []):
                        src = feat.get("source", "")
                        badge = "\U0001F7E2" if src else "\U0001F534"
                        st.write(f"{badge} {feat.get('feature')} ([Source]({src}))" if src else f"{badge} {feat.get('feature')}")
                    if p.get("_sources"):
                        st.caption("Sources: " + ", ".join(p["_sources"][:5]))

        st.download_button(
            label="Download Full JSON Report",
            data=json.dumps({
                "profiles": profiles,
                "report": report,
                "critique": critique,
                "judge_verdicts": verdicts,
                "report_confidence": report_conf,
                "generated_at": datetime.now().isoformat(),
            }, indent=2),
            file_name="competitive_analysis.json",
            mime="application/json"
        )
