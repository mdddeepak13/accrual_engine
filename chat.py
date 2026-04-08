"""
Accrual Conversational AI
─────────────────────────
Interactive chat interface for querying accrual data using natural language.

Usage:
    python3 chat.py
    python3 chat.py --file path/to/data.xlsx --suggest-period 2025/04
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import anthropic

from tools.data_loader import load_accrual_data, get_periods
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

# ─── Chat tool definitions ────────────────────────────────────────────────────

CHAT_TOOLS: list[dict] = [
    {
        "name": "query_accruals",
        "description": (
            "Search and filter historical accrual entries. Use this for questions like "
            "'show utilities accruals for 2025/03', 'accruals above 100000', "
            "'McKinsey entries', 'travel accruals for Acme Corp'. "
            "All filters are optional — omit any you don't need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Period filter e.g. '2025/03'. Format: YYYY/MM.",
                },
                "company_name": {
                    "type": "string",
                    "description": "Partial company name, e.g. 'Acme' or 'GlobalTech EMEA'.",
                },
                "gl_description": {
                    "type": "string",
                    "description": "Partial GL account name, e.g. 'Utilities', 'Travel', 'Professional'.",
                },
                "vendor_name": {
                    "type": "string",
                    "description": "Partial vendor name, e.g. 'McKinsey', 'Deloitte'.",
                },
                "min_amount": {
                    "type": "number",
                    "description": "Minimum amount in USD.",
                },
                "max_amount": {
                    "type": "number",
                    "description": "Maximum amount in USD.",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["amount_desc", "amount_asc", "period", "company", "gl"],
                    "description": "Sort order. Default: amount_desc.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 20, max 100).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_summary",
        "description": (
            "Aggregate and rank accruals by a dimension. Use for: "
            "'which GL has the highest accruals in 2025/01', "
            "'top vendors by spend', 'total by company for Q4', "
            "'breakdown by period'. "
            "group_by is required; all other filters are optional."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["gl_description", "company_name", "vendor_name", "period"],
                    "description": "Dimension to aggregate by.",
                },
                "period": {
                    "type": "string",
                    "description": "Restrict to a specific period, e.g. '2025/03'.",
                },
                "company_name": {
                    "type": "string",
                    "description": "Partial company name filter.",
                },
                "gl_description": {
                    "type": "string",
                    "description": "Partial GL description filter.",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["total_desc", "total_asc", "count_desc", "avg_desc"],
                    "description": "Sort order. Default: total_desc.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max groups to return (default 20).",
                },
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "get_anomalies",
        "description": (
            "Retrieve detected anomalies / irregularities. Use for: "
            "'show anomalies for 2025/03', 'high severity anomalies', "
            "'anomalies for Acme Corp', 'new vendor combinations flagged'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Period filter, e.g. '2025/03'.",
                },
                "severity": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Filter by severity level.",
                },
                "anomaly_type": {
                    "type": "string",
                    "description": "Filter by type: NEW_VENDOR_GL_COMBO, STATISTICAL_OUTLIER, AMOUNT_SPIKE, ZERO_AMOUNT.",
                },
                "company_name": {
                    "type": "string",
                    "description": "Partial company name.",
                },
                "gl_description": {
                    "type": "string",
                    "description": "Partial GL description.",
                },
                "vendor_name": {
                    "type": "string",
                    "description": "Partial vendor name.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max anomalies to return (default 25).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_suggested_accruals",
        "description": (
            "Retrieve AI-suggested accruals for a future period. Use for: "
            "'suggest accruals for April 2025', 'high confidence suggestions for Utilities', "
            "'what should we accrue for GlobalTech next month'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_period": {
                    "type": "string",
                    "description": "Target period in YYYY/MM format, e.g. '2025/04'.",
                },
                "company_name": {
                    "type": "string",
                    "description": "Partial company name filter.",
                },
                "gl_description": {
                    "type": "string",
                    "description": "Partial GL description filter, e.g. 'Utilities', 'Travel'.",
                },
                "vendor_name": {
                    "type": "string",
                    "description": "Partial vendor name filter.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Filter by confidence level.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max suggestions to return (default 25).",
                },
            },
            "required": ["target_period"],
        },
    },
    {
        "name": "compare_periods",
        "description": (
            "Compare accrual amounts between two periods side by side. Use for: "
            "'compare October vs November', 'how did utilities change from 2024/12 to 2025/01', "
            "'period over period variance'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period1": {
                    "type": "string",
                    "description": "First period, e.g. '2024/10'.",
                },
                "period2": {
                    "type": "string",
                    "description": "Second period, e.g. '2024/11'.",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["gl_description", "company_name", "vendor_name"],
                    "description": "Dimension for comparison. Default: gl_description.",
                },
            },
            "required": ["period1", "period2"],
        },
    },
    {
        "name": "get_vendor_profile",
        "description": (
            "Full profile of a vendor: total spend, GL breakdown, period trend, companies, anomalies. "
            "Use for: 'tell me about Deloitte', 'McKinsey profile', 'Oracle vendor history'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "Partial vendor name, e.g. 'Deloitte'.",
                },
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "get_period_overview",
        "description": (
            "High-level dashboard for a specific period: totals, top GL accounts, "
            "top companies, biggest entries, anomaly count. "
            "Use for: 'overview of March 2025', 'summary for 2025/01', 'what happened in Q4 2024'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Period in YYYY/MM format, e.g. '2025/03'.",
                },
            },
            "required": ["period"],
        },
    },
]


# ─── In-memory data store ─────────────────────────────────────────────────────

class DataStore:
    """Holds all pre-computed data so every chat turn can query it instantly."""

    def __init__(self, file_path: str, suggest_period: str):
        print(f"\nLoading accrual data from {file_path} ...", end=" ", flush=True)
        self.df = load_accrual_data(file_path)
        self.periods = get_periods(self.df)
        self.latest_period = self.periods[-1]
        self.suggest_period = suggest_period

        df_history = self.df[self.df["period"] != self.latest_period]
        self.baselines = compute_baselines(df_history)
        self.trends = period_trends(self.df)
        self.anomalies = detect_anomalies(
            self.df[self.df["period"] == self.latest_period],
            self.baselines,
        )
        self.suggestions = suggest_next_period(self.df, self.baselines, suggest_period)
        print("done.\n")

    def context_summary(self) -> str:
        """Short summary injected into every system prompt."""
        total = self.df["amount_usd"].sum()
        return (
            f"Historical data spans {len(self.periods)} periods: {', '.join(self.periods)}. "
            f"Total records: {len(self.df):,}. "
            f"Grand total accruals: ${total:,.0f}. "
            f"Anomaly scan was run on period '{self.latest_period}' — "
            f"{len(self.anomalies)} anomalies detected "
            f"({sum(1 for a in self.anomalies if a['severity']=='HIGH')} HIGH, "
            f"{sum(1 for a in self.anomalies if a['severity']=='MEDIUM')} MEDIUM). "
            f"Suggested accruals are pre-computed for '{self.suggest_period}' "
            f"({len(self.suggestions)} entries, "
            f"${sum(s['suggested_amount_usd'] for s in self.suggestions):,.0f} total)."
        )


# ─── Tool dispatcher ──────────────────────────────────────────────────────────

def dispatch(store: DataStore, tool_name: str, tool_input: dict) -> str:
    limit = min(int(tool_input.get("limit", 25)), 100)

    if tool_name == "query_accruals":
        result = query_accruals(
            store.df,
            period=tool_input.get("period"),
            company_name=tool_input.get("company_name"),
            gl_description=tool_input.get("gl_description"),
            vendor_name=tool_input.get("vendor_name"),
            min_amount=tool_input.get("min_amount"),
            max_amount=tool_input.get("max_amount"),
            sort_by=tool_input.get("sort_by", "amount_desc"),
            limit=limit,
        )

    elif tool_name == "get_summary":
        result = get_summary(
            store.df,
            group_by=tool_input["group_by"],
            period=tool_input.get("period"),
            company_name=tool_input.get("company_name"),
            gl_description=tool_input.get("gl_description"),
            vendor_name=tool_input.get("vendor_name"),
            sort_by=tool_input.get("sort_by", "total_desc"),
            limit=limit,
        )

    elif tool_name == "get_anomalies":
        result = get_anomalies(
            store.anomalies,
            period=tool_input.get("period"),
            severity=tool_input.get("severity"),
            anomaly_type=tool_input.get("anomaly_type"),
            company_name=tool_input.get("company_name"),
            gl_description=tool_input.get("gl_description"),
            vendor_name=tool_input.get("vendor_name"),
            limit=limit,
        )

    elif tool_name == "get_suggested_accruals":
        target = tool_input.get("target_period", store.suggest_period)
        result = get_suggested_accruals(
            store.suggestions,
            target_period=target,
            company_name=tool_input.get("company_name"),
            gl_description=tool_input.get("gl_description"),
            vendor_name=tool_input.get("vendor_name"),
            confidence=tool_input.get("confidence"),
            limit=limit,
        )

    elif tool_name == "compare_periods":
        result = compare_periods(
            store.df,
            period1=tool_input["period1"],
            period2=tool_input["period2"],
            group_by=tool_input.get("group_by", "gl_description"),
        )

    elif tool_name == "get_vendor_profile":
        result = get_vendor_profile(
            store.df,
            store.anomalies,
            vendor_name=tool_input["vendor_name"],
        )

    elif tool_name == "get_period_overview":
        result = get_period_overview(
            store.df,
            store.anomalies,
            period=tool_input["period"],
        )

    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, default=str)


# ─── Conversational agent loop ────────────────────────────────────────────────

def chat(store: DataStore, verbose_tools: bool = False):
    client = anthropic.Anthropic()
    history: list[dict] = []

    system_prompt = f"""You are an expert financial AI assistant specialised in accrual accounting. \
You help the finance team query, analyse, and understand their accrual data through natural conversation.

## Data available to you
{store.context_summary()}

## How to respond
- Be concise and business-focused. Lead with the key number or insight.
- Format monetary amounts with commas and 2 decimal places: $1,234,567.89
- When showing lists of entries, use a compact markdown table where helpful.
- Highlight anomalies and risks clearly — use bold or bullet points.
- If a question is ambiguous (e.g. "this period"), ask for clarification or state your assumption.
- Remember previous questions in the conversation — the user may refer to earlier context.
- Available periods: {', '.join(store.periods)}
- Suggested accruals are pre-computed for: {store.suggest_period}
- Anomaly detection was run on the most recent period: {store.latest_period}

## Available tools
Use the tools to fetch data before answering. Never guess numbers — always query first."""

    print("=" * 60)
    print("  Accrual Conversational AI")
    print(f"  Data: {len(store.df):,} records | {len(store.periods)} periods")
    print(f"  Anomalies: {len(store.anomalies)} | Suggestions: {len(store.suggestions)}")
    print("=" * 60)
    print("  Ask anything about your accrual data.")
    print("  Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            print("Goodbye.")
            break

        history.append({"role": "user", "content": user_input})

        # Inner tool-use loop for this turn
        while True:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=system_prompt,
                tools=CHAT_TOOLS,
                messages=history,
            )

            # Collect content blocks
            text_parts: list[str] = []
            tool_calls: list = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(block)

            if response.stop_reason == "end_turn" or not tool_calls:
                # Final answer — print and add to history
                answer = "\n".join(text_parts)
                print(f"\nAssistant: {answer}\n")
                history.append({"role": "assistant", "content": response.content})
                break

            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                if verbose_tools:
                    print(f"  [tool] {tc.name}({json.dumps(tc.input)})")
                result_str = dispatch(store, tc.name, tc.input)
                if verbose_tools:
                    preview = json.loads(result_str)
                    # Print compact preview without large entry lists
                    compact = {k: v for k, v in preview.items() if k not in ("entries", "rows")}
                    print(f"  [result] {json.dumps(compact)}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_str,
                })

            # Append assistant turn + tool results, continue loop
            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Accrual Conversational AI")
    parser.add_argument(
        "--file", default="Accrual_Sample_Data.xlsx",
        help="Path to accrual Excel file",
    )
    parser.add_argument(
        "--suggest-period", default="2025/04",
        help="Period to pre-compute accrual suggestions for (default: 2025/04)",
    )
    parser.add_argument(
        "--verbose-tools", action="store_true",
        help="Show tool calls and results for debugging",
    )
    args = parser.parse_args()

    store = DataStore(file_path=args.file, suggest_period=args.suggest_period)
    chat(store, verbose_tools=args.verbose_tools)
