import os
import json
import time
import logging
import warnings
import concurrent.futures
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
st.set_page_config(page_title="Competitive Intelligence System", page_icon="⚡", layout="wide")

MODEL = "gemini-3.6-flash"
MAX_COMPETITORS = 3
RELEVANCE_THRESHOLD = 5

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

def search_exa(query: str, num_results: int = 4) -> list[dict]:
    """Execute search query using Exa API."""
    try:
        res = exa_client.search(query, num_results=num_results, type="auto")
        cleaned = []
        for item in res.results:
            cleaned.append({
                "url": item.url,
                "title": getattr(item, "title", "Source"),
                "content": getattr(item, "text", "")[:400]
            })
        return cleaned
    except Exception as e:
        print(f"[EXA ERROR] Query '{query}': {e}")
        return []

# ---------------- 3. The 5 Agent Implementations ----------------

def agent_1_query(company: str, product: str) -> list[str]:
    """Agent 1: Generates targeted search queries."""
    system = "You are a research planner. Return JSON format: {\"queries\": [\"str\"]}"
    prompt = f"Generate 3 distinct search strings to identify direct market competitors to {company} in the domain of {product}."
    data = call_gemini_json(system, prompt)
    queries = data.get("queries", [])
    if not queries:
        return [
            f"{company} {product} top direct competitors",
            f"best enterprise alternatives to {company} in {product}",
            f"top platforms competing with {company}"
        ]
    return queries

def agent_2_discovery(company: str, product: str, search_results: list[dict]) -> list[dict]:
    """Agent 2: Evaluates candidates and filters direct competitors dynamically."""
    system = """Identify direct market competitors for the target company and product.
Return JSON format with key 'competitors':
{"competitors": [{"name": "CompetitorName", "relevance_score": 8, "reasoning": "reason"}]}"""
    
    prompt = f"Target Company: {company}, Product Domain: {product}.\nSearch Evidence:\n{json.dumps(search_results[:12])}"
    data = call_gemini_json(system, prompt)
    candidates = data.get("competitors", [])
    
    valid = [
        c for c in candidates 
        if c.get("name", "").lower() != company.lower() 
        and company.lower() not in c.get("name", "").lower()
        and c.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
    ]
    
    if not valid:
        fallback_prompt = f"List the top 3 direct market competitors to {company} specifically for {product}. Return JSON: {{\"competitors\": [{{\"name\": \"Name\", \"relevance_score\": 9, \"reasoning\": \"Direct competitor\"}}]}}"
        fallback_data = call_gemini_json("You are a market analyst.", fallback_prompt)
        valid = fallback_data.get("competitors", [])

    return valid[:MAX_COMPETITORS]

def agent_3_research_single(comp_name: str, target_company: str, target_product: str) -> dict:
    """Agent 3: Worker agent researching a single competitor."""
    snaps = search_exa(f"{comp_name} features pricing comparison", num_results=3)
    system = """Extract competitor profile as JSON with keys:
    'name', 'one_liner', 'key_features' (list of dicts with 'feature' and 'source'),
    'pricing_model', 'positioning'"""
    
    prompt = f"Competitor: {comp_name}. Target Benchmark: {target_company} {target_product}.\nEvidence: {json.dumps(snaps)}"
    profile = call_gemini_json(system, prompt)
    
    if not profile or "name" not in profile:
        profile = {
            "name": comp_name,
            "one_liner": "Enterprise platform rival",
            "key_features": [{"feature": "Core capability solution", "source": snaps[0]["url"] if snaps else ""}],
            "pricing_model": "Custom / Enterprise Tier",
            "positioning": "Direct market competitor"
        }
    profile["_sources"] = [s["url"] for s in snaps if "url" in s]
    return profile

def agent_4_analyst(company: str, product: str, profiles: list[dict]) -> dict:
    """Agent 4: Synthesizes profiles into a comparative matrix and SWOT analysis."""
    system = f"""You are a senior Product Management Lead. Produce competitive analysis JSON.

Rules for 'comparison_table':
- Must be a list of objects representing evaluation dimensions (e.g., 'Core Focus', 'Target Audience', 'Pricing Model').
- Each object MUST include 'Dimension', 'Target ({company})', and explicit separate keys for EACH competitor by name.

Return JSON format:
{{
  "positioning_statement": "string",
  "comparison_table": [
    {{
      "Dimension": "Core Focus",
      "Target ({company})": "Primary capability description",
      "Competitor A": "Competitor capability description"
    }}
  ],
  "swot": {{"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}},
  "threat_level": "Low | Medium | High"
}}"""
    
    prompt = f"Target: {company} ({product}). Competitor Profiles:\n{json.dumps(profiles)}"
    return call_gemini_json(system, prompt)

def agent_5_critic(profiles: list[dict], report: dict) -> dict:
    """Agent 5: Fact-checker and validator."""
    system = """You are a QA Critic. Verify report accuracy against evidence. Return JSON:
    {'flags': list of strings, 'confidence': 'High' | 'Medium' | 'Low'}"""
    
    prompt = f"Profiles: {json.dumps(profiles)}\nReport: {json.dumps(report)}"
    return call_gemini_json(system, prompt)

# ---------------- 4. Streamlit App Interface ----------------

st.title("⚡ 5-Agent Competitive Intelligence System")
st.caption("Agentic Workflow: Query Plan -> Discovery -> Parallel Deep Research -> Analyst Synthesis -> Critic Audit")

with st.sidebar:
    company_input = st.text_input("Target Company", value="ScienceLogic")
    product_input = st.text_input("Product / Domain", value="AIOps & Infrastructure Monitoring")
    start_btn = st.button("Run Analysis Pipeline", type="primary")

if start_btn:
    if not company_input or not product_input:
        st.warning("Please specify both company and product/domain.")
    else:
        with st.status("Executing 5-Agent Pipeline...", expanded=True) as status:
            # Agent 1
            status.update(label="Agent 1: Planning targeted searches...")
            queries = agent_1_query(company_input, product_input)
            
            # Agent 2
            status.update(label="Agent 2: Running web discovery & evaluating candidates...")
            search_data = []
            for q in queries:
                search_data.extend(search_exa(q))
            candidates = agent_2_discovery(company_input, product_input, search_data)
            
            # Agent 3
            status.update(label=f"Agent 3: Conducting parallel research on {len(candidates)} competitors...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                profiles = list(executor.map(
                    lambda c: agent_3_research_single(c["name"], company_input, product_input),
                    candidates
                ))
            
            # Agent 4
            status.update(label="Agent 4: Synthesizing SWOT, positioning & matrix...")
            report = agent_4_analyst(company_input, product_input, profiles)
            
            # Agent 5
            status.update(label="Agent 5: Running QA audit & fact check...")
            critique = agent_5_critic(profiles, report)
            
            status.update(label="Pipeline Complete!", state="complete")

        # Metrics Header
        m1, m2, m3 = st.columns(3)
        m1.metric("Competitors Identified", len(profiles))
        m2.metric("Sources Analyzed", sum(len(p.get("_sources", [])) for p in profiles))
        m3.metric("Critic Audit Flags", len(critique.get("flags", [])))

        if critique.get("flags"):
            with st.expander("⚠️ Audit Flag (Critic Agent5"):
                for flag in critique["flags"]:
                    st.write(f"- {flag}")

        # Core Display Tabs
        t1, t2, t3 = st.tabs(["Comparison Matrix", "SWOT & Positioning", "Sources & Profiles"])

        with t1:
            st.markdown(f"**Positioning Statement**\n {report.get('positioning_statement', 'N/A')}")
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

                # Custom CSS for single-line headers, no empty row offsets, and clean formatting
                st.markdown(
                    """
                    <style>
                    .custom-matrix-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                        font-family: inherit;
                    }
                    .custom-matrix-table th {
                        white-space: nowrap !important;
                        text-align: left;
                        padding: 10px 12px;
                        background-color: #f0f2f6;
                        border-bottom: 2px solid #d0d4dc;
                        font-weight: 600;
                    }
                    .custom-matrix-table td {
                        padding: 10px 12px;
                        border-bottom: 1px solid #e6e9ef;
                        vertical-align: top;
                    }
                    .custom-matrix-table td:first-child {
                        font-weight: 600;
                        white-space: nowrap !important;
                        background-color: #f8f9fa;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                # Render HTML table with index disabled to eliminate blank row
                st.write(
                    df.to_html(classes="custom-matrix-table", escape=False, index=False),
                    unsafe_allow_html=True
                )
            else:
                st.info("No comparison data generated.")

        with t2:
            swot = report.get("swot", {})
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**STRENGTHS**")
                for item in swot.get("strengths", []): st.write(f"- {item}")
                st.markdown("**OPPORTUNITIES**")
                for item in swot.get("opportunities", []): st.write(f"- {item}")
            with col_b:
                st.markdown("**WEAKNESSES**")
                for item in swot.get("weaknesses", []): st.write(f"- {item}")
                st.markdown("**THREATS**")
                for item in swot.get("threats", []): st.write(f"- {item}")
            st.markdown(f"**Overall Threat Level:** {report.get('threat_level', 'Medium')}")

        with t3:
            for p in profiles:
                with st.expander(f"{p.get('name')} Overview"):
                    st.write(f"**One Liner:** {p.get('one_liner')}")
                    st.write(f"**Pricing Model:** {p.get('pricing_model')}")
                    st.write("**Key Features:**")
                    for feat in p.get("key_features", []):
                        st.write(f"- {feat.get('feature')} ([Source]({feat.get('source')}))")

        st.download_button(
            label="Download Full JSON Report",
            data=json.dumps({"profiles": profiles, "report": report, "critique": critique}, indent=2),
            file_name="competitive_analysis.json",
            mime="application/json"
        )
