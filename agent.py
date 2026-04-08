"""
Accrual Data Processing Agent
──────────────────────────────
Uses Claude with tool use to:
  1. Load and analyze historical accrual data
  2. Detect anomalies / irregularities
  3. Suggest accruals for the next period
  4. Generate Excel + JSON reports
"""

import json
import sys
from pathlib import Path

import anthropic

from tools.data_loader import load_accrual_data, get_periods, filter_by_period
from tools.analyzer import compute_baselines, period_trends, top_accruals_by_gl, summary_by_company, format_for_llm
from tools.anomaly_detector import detect_anomalies
from tools.accrual_suggester import suggest_next_period
from tools.reporter import generate_excel_report, generate_json_report

# ── Tool definitions exposed to Claude ────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "load_and_analyze_data",
        "description": (
            "Load historical accrual data from the Excel file and return a statistical summary "
            "including period trends, top GL accounts by amount, and company-level summaries. "
            "Call this first before any other tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the accrual Excel file.",
                },
                "target_period": {
                    "type": "string",
                    "description": "The period to generate accruals for, e.g. '2025/04'.",
                },
            },
            "required": ["file_path", "target_period"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Scan historical or new accrual entries for irregularities: statistical outliers, "
            "new vendor-GL combinations, zero amounts, and amount spikes. "
            "Returns a list of flagged entries with severity and explanation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period_filter": {
                    "type": "string",
                    "description": "Optional period to restrict anomaly scan, e.g. '2025/03'. If omitted, all periods are scanned.",
                },
                "z_threshold": {
                    "type": "number",
                    "description": "Z-score threshold for statistical outlier detection. Default is 2.0.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "suggest_accruals",
        "description": (
            "Generate suggested accrual entries for the target period based on historical patterns. "
            "Each suggestion includes the recommended amount, confidence level, and the statistical basis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "generate_reports",
        "description": (
            "Write the final Excel and JSON reports to disk. "
            "Returns the file paths of the generated reports."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Directory where reports should be saved. Defaults to current directory.",
                },
            },
            "required": [],
        },
    },
]


# ── Agent state (populated as tools run) ──────────────────────────────────────
_state: dict = {}


def _run_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call and return the result as a JSON string."""

    if tool_name == "load_and_analyze_data":
        file_path = tool_input["file_path"]
        target_period = tool_input["target_period"]

        df = load_accrual_data(file_path)
        periods = get_periods(df)
        latest_period = periods[-1]

        # Build baselines from all periods EXCEPT the latest, so anomaly
        # detection has a clean holdout set to evaluate.
        df_history = df[df["period"] != latest_period].copy()
        df_latest  = df[df["period"] == latest_period].copy()

        baselines = compute_baselines(df_history)
        trends = period_trends(df)
        top_gl = top_accruals_by_gl(df, top_n=10)
        company_sum = summary_by_company(df)

        _state["df"] = df
        _state["df_history"] = df_history
        _state["df_latest"] = df_latest
        _state["baselines"] = baselines
        _state["trends"] = trends
        _state["target_period"] = target_period
        _state["latest_period"] = latest_period
        _state["periods"] = periods

        summary = format_for_llm(baselines, trends, top_gl, company_sum)
        return json.dumps({
            "status": "ok",
            "historical_periods": periods,
            "latest_period_for_anomaly_scan": latest_period,
            "target_period": target_period,
            "total_records": len(df),
            "history_records": len(df_history),
            "latest_period_records": len(df_latest),
            "unique_combinations": len(baselines),
            "analysis": json.loads(summary),
        })

    if tool_name == "detect_anomalies":
        if "df" not in _state:
            return json.dumps({"error": "Call load_and_analyze_data first."})

        baselines = _state["baselines"]
        z_threshold = float(tool_input.get("z_threshold", 2.0))

        # Default: scan only the latest period (held out from baseline computation).
        period_filter = tool_input.get("period_filter") or _state.get("latest_period")
        if period_filter:
            scan_df = filter_by_period(_state["df"], period_filter)
        else:
            scan_df = _state["df_latest"]

        anomalies = detect_anomalies(scan_df, baselines, z_threshold=z_threshold)
        _state["anomalies"] = anomalies

        severity_counts = {
            "HIGH": sum(1 for a in anomalies if a["severity"] == "HIGH"),
            "MEDIUM": sum(1 for a in anomalies if a["severity"] == "MEDIUM"),
            "LOW": sum(1 for a in anomalies if a["severity"] == "LOW"),
        }
        return json.dumps({
            "status": "ok",
            "total_anomalies": len(anomalies),
            "severity_breakdown": severity_counts,
            "anomalies": anomalies[:20],  # cap to keep payload reasonable
            "truncated": len(anomalies) > 20,
        })

    if tool_name == "suggest_accruals":
        if "df" not in _state:
            return json.dumps({"error": "Call load_and_analyze_data first."})

        suggestions = suggest_next_period(
            _state["df"], _state["baselines"], _state["target_period"]
        )
        _state["suggestions"] = suggestions

        confidence_counts = {
            "HIGH": sum(1 for s in suggestions if s["confidence"] == "HIGH"),
            "MEDIUM": sum(1 for s in suggestions if s["confidence"] == "MEDIUM"),
            "LOW": sum(1 for s in suggestions if s["confidence"] == "LOW"),
        }
        return json.dumps({
            "status": "ok",
            "target_period": _state["target_period"],
            "total_suggestions": len(suggestions),
            "total_amount_usd": round(sum(s["suggested_amount_usd"] for s in suggestions), 2),
            "confidence_breakdown": confidence_counts,
            "sample_suggestions": suggestions[:10],
        })

    if tool_name == "generate_reports":
        required = ["df", "baselines", "suggestions", "anomalies", "trends"]
        missing = [k for k in required if k not in _state]
        if missing:
            return json.dumps({"error": f"Missing state: {missing}. Run previous tools first."})

        output_dir = tool_input.get("output_dir", ".")
        agent_analysis = _state.get("agent_analysis", "")

        excel_path = generate_excel_report(
            _state["suggestions"],
            _state["anomalies"],
            _state["trends"],
            _state["target_period"],
            output_dir=output_dir,
        )
        json_path = generate_json_report(
            _state["suggestions"],
            _state["anomalies"],
            agent_analysis,
            _state["target_period"],
            output_dir=output_dir,
        )

        return json.dumps({
            "status": "ok",
            "excel_report": excel_path,
            "json_report": json_path,
            "total_accruals": len(_state["suggestions"]),
            "total_amount_usd": round(sum(s["suggested_amount_usd"] for s in _state["suggestions"]), 2),
            "anomalies_flagged": len(_state["anomalies"]),
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Main agentic loop ─────────────────────────────────────────────────────────

def run_agent(
    file_path: str,
    target_period: str,
    output_dir: str = ".",
    verbose: bool = True,
) -> dict:
    """
    Run the full accrual processing pipeline via Claude tool use.
    Returns the final agent response and report paths.
    """
    client = anthropic.Anthropic()

    system_prompt = """You are an expert financial AI agent specializing in accrual accounting.
Your job is to automate the month-end accrual process for the finance team by:
1. Analyzing historical accrual data to understand patterns and baselines
2. Detecting any irregularities or anomalies in the data
3. Suggesting accrual entries for the upcoming period based on trends
4. Generating formal reports for the finance team

Always follow this exact sequence:
  Step 1 → load_and_analyze_data
  Step 2 → detect_anomalies
  Step 3 → suggest_accruals
  Step 4 → generate_reports

After each tool call, reason about the results before proceeding to the next step.
In your final response, provide a concise executive summary covering:
- Key pattern insights from historical data
- Top anomalies requiring human review
- Confidence in the suggested accruals
- Any risks or recommendations for the finance team"""

    user_message = f"""Please process the accrual data and generate the month-end accrual report.

Data file: {file_path}
Target period: {target_period}
Output directory: {output_dir}

Run the full pipeline: analyze historical data, detect anomalies, suggest accruals for {target_period}, and generate reports."""

    messages = [{"role": "user", "content": user_message}]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Accrual Processing Agent")
        print(f"  Target period: {target_period}")
        print(f"{'='*60}\n")

    # Agentic loop
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Collect text and tool-use blocks
        text_blocks = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        if text_blocks and verbose:
            print("[Agent]", "\n".join(text_blocks))

        # If stop_reason is end_turn or no tool calls, we're done
        if response.stop_reason == "end_turn" or not tool_calls:
            _state["agent_analysis"] = "\n".join(text_blocks)
            break

        # Execute all tool calls
        tool_results = []
        for tc in tool_calls:
            if verbose:
                print(f"\n[Tool] {tc.name}({json.dumps(tc.input, indent=2)})")

            result_str = _run_tool(tc.name, tc.input)

            if verbose:
                result_preview = json.loads(result_str)
                # Print a compact preview
                preview = {k: v for k, v in result_preview.items() if k != "analysis"}
                print(f"[Result] {json.dumps(preview, indent=2)}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result_str,
            })

        # Append assistant turn + tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return {
        "agent_analysis": _state.get("agent_analysis", ""),
        "suggestions": _state.get("suggestions", []),
        "anomalies": _state.get("anomalies", []),
        "target_period": target_period,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Accrual Data Processing Agent")
    parser.add_argument("--file", default="Accrual_Sample_Data.xlsx", help="Path to accrual Excel file")
    parser.add_argument("--period", default="2025/04", help="Target period, e.g. 2025/04")
    parser.add_argument("--output", default=".", help="Output directory for reports")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    result = run_agent(
        file_path=args.file,
        target_period=args.period,
        output_dir=args.output,
        verbose=not args.quiet,
    )

    print(f"\n{'='*60}")
    print("  Processing Complete")
    print(f"{'='*60}")
    print(f"  Suggested accruals: {len(result['suggestions'])}")
    print(f"  Anomalies detected: {len(result['anomalies'])}")
    print(f"  Target period:      {result['target_period']}")
    print(f"{'='*60}\n")
