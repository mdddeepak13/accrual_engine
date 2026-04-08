"""
Accrual AI — Streamlit Web App
Upload one or more accrual Excel/CSV files, then chat with your data.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Community Cloud
    Set ANTHROPIC_API_KEY as a secret in Streamlit Cloud settings.
"""

from __future__ import annotations

import json
import os

import anthropic
import pandas as pd
import streamlit as st

from tools.data_loader import (
    load_accrual_data,
    load_from_uploads,
    merge_datasets,
    get_periods,
)
from tools.analyzer import compute_baselines, period_trends
from tools.anomaly_detector import detect_anomalies
from tools.accrual_suggester import suggest_next_period
from tools.query_engine import (
    query_accruals,
    get_summary,
    get_anomalies,
    get_suggested_accruals,
    compare_periods,
    get_vendor_profile,
    get_period_overview,
)

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Accrual AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Gradient header */
.accrual-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    color: white;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 20px;
}
.accrual-header h1 { margin: 0; font-size: 26px; color: white; }
.accrual-header p  { margin: 4px 0 0; font-size: 14px; opacity: 0.85; color: white; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f0f4ff;
    border: 1px solid #d0d9f0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 6px;
}

/* Upload zone */
.upload-hero {
    background: #f8faff;
    border: 2px dashed #90b4e8;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin: 40px auto;
    max-width: 640px;
}
.upload-hero h2 { color: #1e3a5f; margin-bottom: 8px; }
.upload-hero p  { color: #555; font-size: 15px; }

/* Source file badges */
.src-badge {
    display: inline-block;
    background: #e0edff;
    color: #1e3a5f;
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 3px;
}

/* Quick-action buttons */
div[data-testid="column"] button {
    width: 100%;
    border-radius: 8px;
    font-size: 13px;
    text-align: left;
    white-space: normal;
    height: auto;
    padding: 8px 12px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state initialisation ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "api_key":       os.environ.get("ANTHROPIC_API_KEY", ""),
        "df":            None,       # combined DataFrame
        "baselines":     None,
        "trends":        None,
        "anomalies":     None,
        "suggestions":   None,
        "suggest_period": None,
        "source_files":  [],         # names of uploaded files
        "messages":      [],         # chat history
        "data_version":  0,          # bumped whenever data changes → clears chat
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

DEFAULT_FILE = "Accrual_Sample_Data.xlsx"

_init_state()


# ─── Data processing ──────────────────────────────────────────────────────────

def _compute_store(df: pd.DataFrame) -> dict:
    """Derive all in-memory analytics from a DataFrame."""
    periods = get_periods(df)
    latest  = periods[-1] if periods else None

    df_hist = df[df["period"] != latest] if latest else df
    df_last = df[df["period"] == latest] if latest else pd.DataFrame()

    baselines   = compute_baselines(df_hist)
    trends      = period_trends(df)
    anomalies   = detect_anomalies(df_last, baselines) if not df_last.empty else []

    # Suggest for the period after the latest
    if latest:
        y, m    = int(latest[:4]), int(latest[5:])
        m      += 1
        if m > 12:
            m, y = 1, y + 1
        suggest_period = f"{y}/{m:02d}"
    else:
        suggest_period = "2025/04"

    suggestions = suggest_next_period(df, baselines, suggest_period)

    return dict(
        baselines=baselines,
        trends=trends,
        anomalies=anomalies,
        suggestions=suggestions,
        suggest_period=suggest_period,
    )


def _apply_store(computed: dict):
    for k, v in computed.items():
        st.session_state[k] = v


def _load_default():
    """Auto-load the bundled sample file on first run if no data uploaded yet."""
    import pathlib
    path = pathlib.Path(DEFAULT_FILE)
    if not path.exists():
        return
    try:
        df = load_accrual_data(str(path))
        computed = _compute_store(df)
        st.session_state.df = df
        st.session_state.source_files = [path.name]
        for k, v in computed.items():
            st.session_state[k] = v
    except Exception:
        pass  # let the user upload manually if default fails


def process_uploads(uploaded_files, mode: str = "replace"):
    """Load uploaded files, merge if needed, recompute analytics."""
    with st.spinner(f"Processing {len(uploaded_files)} file(s)…"):
        new_df = load_from_uploads(uploaded_files)

    if mode == "append" and st.session_state.df is not None:
        with st.spinner("Merging with existing data…"):
            combined = merge_datasets(st.session_state.df, new_df)
        new_names = [f.name for f in uploaded_files]
        st.session_state.source_files = list(
            dict.fromkeys(st.session_state.source_files + new_names)
        )
    else:
        combined = new_df
        st.session_state.source_files = [f.name for f in uploaded_files]

    with st.spinner("Computing baselines, anomalies, suggestions…"):
        computed = _compute_store(combined)

    st.session_state.df = combined
    _apply_store(computed)
    st.session_state.data_version += 1
    st.session_state.messages = []   # fresh chat for new/changed data


# ─── Claude tools ─────────────────────────────────────────────────────────────

CHAT_TOOLS: list[dict] = [
    {
        "name": "query_accruals",
        "description": (
            "Search and filter historical accrual entries. Use for questions like "
            "'show utilities accruals for 2025/03', 'accruals above 100000', "
            "'McKinsey entries', 'travel accruals for Acme Corp'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period":         {"type": "string",  "description": "YYYY/MM, e.g. '2025/03'"},
                "company_name":   {"type": "string",  "description": "Partial company name"},
                "gl_description": {"type": "string",  "description": "Partial GL name, e.g. 'Utilities', 'Travel'"},
                "vendor_name":    {"type": "string",  "description": "Partial vendor name"},
                "min_amount":     {"type": "number"},
                "max_amount":     {"type": "number"},
                "sort_by":        {"type": "string",  "enum": ["amount_desc","amount_asc","period","company","gl"]},
                "limit":          {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "get_summary",
        "description": (
            "Aggregate accruals by a dimension. Use for: "
            "'top GL accounts in March', 'total by company', 'vendor spend breakdown'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by":       {"type": "string", "enum": ["gl_description","company_name","vendor_name","period"]},
                "period":         {"type": "string"},
                "company_name":   {"type": "string"},
                "gl_description": {"type": "string"},
                "sort_by":        {"type": "string", "enum": ["total_desc","total_asc","count_desc","avg_desc"]},
                "limit":          {"type": "integer"},
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "get_anomalies",
        "description": (
            "Get detected anomalies. Use for: "
            "'show anomalies for 2025/03', 'high severity issues', 'new vendor combinations'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period":         {"type": "string"},
                "severity":       {"type": "string", "enum": ["HIGH","MEDIUM","LOW"]},
                "anomaly_type":   {"type": "string"},
                "company_name":   {"type": "string"},
                "gl_description": {"type": "string"},
                "vendor_name":    {"type": "string"},
                "limit":          {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "get_suggested_accruals",
        "description": (
            "AI-suggested accruals for a future period. Use for: "
            "'suggest accruals for April', 'high confidence utilities suggestions'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_period":  {"type": "string", "description": "YYYY/MM, e.g. '2025/04'"},
                "company_name":   {"type": "string"},
                "gl_description": {"type": "string"},
                "vendor_name":    {"type": "string"},
                "confidence":     {"type": "string", "enum": ["HIGH","MEDIUM","LOW"]},
                "limit":          {"type": "integer"},
            },
            "required": ["target_period"],
        },
    },
    {
        "name": "compare_periods",
        "description": (
            "Side-by-side comparison of two periods. Use for: "
            "'compare Oct vs Nov', 'MoM variance by GL', 'period over period change'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period1":  {"type": "string"},
                "period2":  {"type": "string"},
                "group_by": {"type": "string", "enum": ["gl_description","company_name","vendor_name"]},
            },
            "required": ["period1","period2"],
        },
    },
    {
        "name": "get_vendor_profile",
        "description": (
            "Full vendor profile: spend, GL breakdown, period trend, anomalies. "
            "Use for: 'tell me about Deloitte', 'McKinsey history'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {"type": "string"},
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "get_period_overview",
        "description": (
            "Dashboard for a period: totals, top GLs, top companies, anomaly count. "
            "Use for: 'overview of March 2025', 'summary for 2025/01'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {"type": "string"},
            },
            "required": ["period"],
        },
    },
]


def _dispatch(tool_name: str, tool_input: dict) -> str:
    df         = st.session_state.df
    anomalies  = st.session_state.anomalies or []
    suggestions= st.session_state.suggestions or []
    suggest_pd = st.session_state.suggest_period or ""
    limit      = min(int(tool_input.get("limit", 25)), 100)

    def _kw(*keys):
        return {k: tool_input[k] for k in keys if tool_input.get(k) is not None}

    if tool_name == "query_accruals":
        r = query_accruals(df, limit=limit,
                **_kw("period","company_name","gl_description","vendor_name",
                      "min_amount","max_amount","sort_by"))
    elif tool_name == "get_summary":
        r = get_summary(df, group_by=tool_input["group_by"], limit=limit,
                **_kw("period","company_name","gl_description","sort_by"))
    elif tool_name == "get_anomalies":
        r = get_anomalies(anomalies, limit=limit,
                **_kw("period","severity","anomaly_type","company_name",
                      "gl_description","vendor_name"))
    elif tool_name == "get_suggested_accruals":
        r = get_suggested_accruals(suggestions,
                target_period=tool_input.get("target_period", suggest_pd),
                limit=limit,
                **_kw("company_name","gl_description","vendor_name","confidence"))
    elif tool_name == "compare_periods":
        r = compare_periods(df,
                period1=tool_input["period1"], period2=tool_input["period2"],
                group_by=tool_input.get("group_by","gl_description"))
    elif tool_name == "get_vendor_profile":
        r = get_vendor_profile(df, anomalies, vendor_name=tool_input["vendor_name"])
    elif tool_name == "get_period_overview":
        r = get_period_overview(df, anomalies, period=tool_input["period"])
    else:
        r = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(r, default=str)


def _get_ai_response(messages: list[dict]) -> str:
    """Run one full Claude turn (may call multiple tools) and return final text."""
    client  = anthropic.Anthropic(api_key=st.session_state.api_key)
    df      = st.session_state.df
    periods = get_periods(df)
    total   = float(df["amount_usd"].sum())
    anom    = st.session_state.anomalies or []
    sugg    = st.session_state.suggestions or []
    sp      = st.session_state.suggest_period

    system = f"""You are an expert financial AI assistant specialised in accrual accounting.
You help the finance team query, analyse, and understand their accrual data.

## Loaded data
- Source files: {', '.join(st.session_state.source_files)}
- Periods: {', '.join(periods)}
- Total records: {len(df):,} | Grand total: ${total:,.0f}
- Anomaly scan on '{periods[-1] if periods else "N/A"}': \
{len(anom)} flagged \
({sum(1 for a in anom if a['severity']=='HIGH')} HIGH, \
{sum(1 for a in anom if a['severity']=='MEDIUM')} MEDIUM)
- Suggested accruals pre-computed for '{sp}': \
{len(sugg)} entries, ${sum(x['suggested_amount_usd'] for x in sugg):,.0f}

## Response style
- Be concise. Lead with the key figure or insight.
- Format amounts with commas and 2 dp: $1,234,567.89
- Use markdown tables for lists of entries.
- Bold important figures and anomaly flags.
- Available periods: {', '.join(periods)}"""

    work = list(messages)
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system,
            tools=CHAT_TOOLS,
            messages=work,
        )
        text_parts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason == "end_turn" or not tool_calls:
            return "\n".join(text_parts)

        tool_results = []
        for tc in tool_calls:
            result_str = _dispatch(tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result_str,
            })
        work.append({"role": "assistant", "content": response.content})
        work.append({"role": "user",      "content": tool_results})


# Auto-load bundled sample data on very first run (before any upload)
if st.session_state.df is None:
    _load_default()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    api_val = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-ant-…",
        help="Or set the ANTHROPIC_API_KEY environment variable.",
    )
    if api_val:
        st.session_state.api_key = api_val

    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")

    uploaded = st.file_uploader(
        "Upload accrual file(s)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Excel (.xlsx / .xls) or CSV. You can upload multiple files at once.",
        label_visibility="collapsed",
    )

    col_r, col_a = st.columns(2)
    load_btn    = col_r.button("Load / Replace",  use_container_width=True,
                               disabled=not uploaded, type="primary")
    append_btn  = col_a.button("Add to Existing", use_container_width=True,
                               disabled=not (uploaded and st.session_state.df is not None))

    if load_btn and uploaded:
        try:
            process_uploads(uploaded, mode="replace")
            st.success(f"Loaded {len(uploaded)} file(s).")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if append_btn and uploaded:
        try:
            process_uploads(uploaded, mode="append")
            st.success(f"Appended {len(uploaded)} file(s). Total rows: {len(st.session_state.df):,}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    # Current sources
    if st.session_state.source_files:
        st.markdown("**Active sources:**")
        for f in st.session_state.source_files:
            st.markdown(f'<span class="src-badge">📄 {f}</span>', unsafe_allow_html=True)

    st.divider()

    # ── Data metrics ──────────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df   = st.session_state.df
        anom = st.session_state.anomalies or []
        sugg = st.session_state.suggestions or []
        sp   = st.session_state.suggest_period

        st.markdown("### 📊 Dataset")
        st.metric("Total Records",     f"{len(df):,}")
        st.metric("Grand Total",       f"${df['amount_usd'].sum()/1e6:.2f}M")
        st.metric("Periods",           len(get_periods(df)))
        high = sum(1 for a in anom if a["severity"] == "HIGH")
        med  = sum(1 for a in anom if a["severity"] == "MEDIUM")
        st.metric("Anomalies Flagged", f"{len(anom)}  ({high}H / {med}M)")
        st.metric("Suggestions",       f"{len(sugg)} for {sp}")

        st.divider()
        st.markdown("### 📅 Periods")
        for p in get_periods(df):
            pt = df[df["period"] == p]["amount_usd"].sum()
            st.caption(f"**{p}** — ${pt/1e6:.2f}M")

        st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True,
                 disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

# ─── Main area ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="accrual-header">
  <h1>📊 Accrual AI Assistant</h1>
  <p>Upload your accrual data, then ask anything — anomalies, trends, vendor spend, period comparisons, and more.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload prompt (no data loaded) ────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
<div class="upload-hero">
  <h2>👋 Welcome to Accrual AI</h2>
  <p>Start by uploading one or more accrual Excel or CSV files using the sidebar.<br>
  You can load a single file, multiple files at once, or add files to an existing dataset.</p>
  <br>
  <p style="font-size:13px;color:#888;">Supported formats: <strong>.xlsx</strong>, <strong>.xls</strong>, <strong>.csv</strong></p>
</div>
""", unsafe_allow_html=True)

    # Show expected format
    with st.expander("📋 Expected file format"):
        st.markdown("""
The file should contain these columns (names are flexible — common aliases are handled automatically):

| Column | Example |
|--------|---------|
| Company Code | 1000 |
| Company Name | Acme Corp USA |
| Posting Date | 2025/03/01 |
| GL Account Number | 600500 |
| GL Description | Utilities Expense |
| Vendor Number | V1008 |
| Vendor Name | Oracle Corp |
| Accrual From Period | 2025/03/01 |
| Accrual To Period | 2025/03/31 |
| Amount (USD) | 185435.45 |

Extra columns are preserved but ignored.
""")
    st.stop()

# ── Quick-action buttons ───────────────────────────────────────────────────────
df      = st.session_state.df
periods = get_periods(df)
latest  = periods[-1] if periods else "N/A"
sp      = st.session_state.suggest_period or "next period"

quick_prompts = [
    ("🔍 Period overview",        f"Give me an overview of {latest}"),
    ("⚠️ Show anomalies",         f"Show all anomalies detected in {latest}"),
    ("🔴 High severity",          "Show only HIGH severity anomalies"),
    ("💡 Suggest accruals",       f"Suggest accruals for {sp}"),
    ("⚡ Utilities accruals",     f"Show utilities accruals for {latest}"),
    ("📊 Top GL accounts",        f"Which GL accounts had the highest accruals in {latest}?"),
    ("🏢 Company breakdown",      f"Break down accruals by company for {latest}"),
    ("🔄 Compare last 2 periods", f"Compare {periods[-2]} vs {periods[-1]} by GL account"
                                   if len(periods) >= 2 else "Compare periods"),
    ("🤝 Vendor profiles",        "Show me the top 5 vendors by total spend"),
    ("📈 Trend across periods",   "Show the total accrual trend across all periods"),
]

st.markdown("**Quick questions:**")
cols = st.columns(5)
for idx, (label, prompt) in enumerate(quick_prompts):
    if cols[idx % 5].button(label, key=f"q_{idx}_{st.session_state.data_version}"):
        st.session_state._pending = prompt

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "user" else "📊"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Pending quick-action injection ────────────────────────────────────────────
pending = st.session_state.pop("_pending", None)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about accruals, anomalies, vendors, periods…") or pending

if user_input:
    if not st.session_state.api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    claude_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Analysing…"):
            answer = _get_ai_response(claude_messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
