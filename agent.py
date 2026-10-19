"""
agent.py - turns metrics + management claims into a report using an LLM.

Steps:  1) metrics (code)  2) analyst LLM: hypothesis  3) critic LLM: challenge  4) report.md

Setup:
  pip install huggingface_hub openai pandas tabulate
  Either: HF_TOKEN + HF_MODEL (Hugging Face; needs credits)
  Or:     LLM_BASE_URL + LLM_API_KEY + LLM_MODEL (any OpenAI-compatible API)

Usage:
  python agent.py britannia.csv claims.txt Britannia
  python agent.py britannia.csv claims.txt Britannia --dry-run   # prints prompts, no LLM call
"""
import os
import sys
import pandas as pd
from metrics import load, add_metrics, flags

SYSTEM = (
    "You are a careful financial research analyst. "
    "Use ONLY the numbers in the data table and the claims provided. "
    "Never invent figures, sources, or events. "
    "Every claim must cite its source_file and page from the table or claims list. "
    "If the data is not enough to support a finding, say so plainly."
)


def facts_table(df):
    cols = ["quarter", "revenue_cr", "revenue_yoy_pct", "volume_growth_pct",
            "receivable_days", "inventory_days", "source_file", "page"]
    return df[cols].round(2).to_markdown(index=False)


def build_analyst_prompt(company, table, claims, flag_list):
    return f"""Company: {company}

DATA TABLE (consolidated, Rs crore; revenue = Total revenue from operations):
{table}

AUTO-FLAGS FROM CODE (leads, not conclusions):
{chr(10).join('- ' + f for f in flag_list) or '- none'}

MANAGEMENT CLAIMS (claim | source | page):
{claims}

TASK: Compare what management says with what the numbers show.
Write:
1. DISCOVERY: one sentence. If nothing is supported by the data, say "No supported finding".
2. WHY IT MATTERS: 3 bullets.
3. TIMELINE: when the pattern appeared, quarter by quarter.
4. NUMBERS: only the calculations used (show the arithmetic).
5. EVIDENCE: list each source_file and page used.
Note: 'Sale of goods' and 'Total revenue from operations' are different lines; do not mix them."""


def build_critic_prompt(company, table, hypothesis, has_volume, has_peers):
    cap = ""
    if not has_volume or not has_peers:
        missing = []
        if not has_volume:
            missing.append("volume data")
        if not has_peers:
            missing.append("peer comparison")
        cap = f"Missing: {', '.join(missing)}. Evidence Strength cannot be higher than Weak."
    return f"""Company: {company}

DATA TABLE:
{table}

ANALYST HYPOTHESIS:
{hypothesis}

TASK: Try to DISPROVE the hypothesis. Use only the table and the hypothesis text.
Write:
1. COUNTERARGUMENT: the strongest alternative explanation (seasonality, one-off, definition mismatch, sector effect).
2. FALSIFICATION CONDITION: what future data would overturn the conclusion.
3. EVIDENCE STRENGTH: Strong, Moderate or Weak, with reasoning based on number of independent sources and consistency.
4. DATA GAPS: what is missing to be sure.
{cap}"""


def call_llm(prompt):
    """Backends:
    1) OpenAI-compatible (Groq, Gemini, OpenRouter...): set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    2) Hugging Face Inference Providers (default): set HF_TOKEN, HF_MODEL
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    if os.environ.get("LLM_BASE_URL"):
        from openai import OpenAI
        key, model = os.environ.get("LLM_API_KEY"), os.environ.get("LLM_MODEL")
        if not key or not model:
            sys.exit("Set LLM_API_KEY and LLM_MODEL environment variables first.")
        client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=key)
        resp = client.chat.completions.create(model=model, messages=messages,
                                              max_tokens=900, temperature=0.2)
        return resp.choices[0].message.content

    from huggingface_hub import InferenceClient
    token, model = os.environ.get("HF_TOKEN"), os.environ.get("HF_MODEL")
    if not token or not model:
        sys.exit("Set HF_TOKEN and HF_MODEL (or the LLM_* variables) first.")
    client = InferenceClient(api_key=token)
    resp = client.chat.completions.create(model=model, messages=messages,
                                          max_tokens=900, temperature=0.2)
    return resp.choices[0].message.content


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if len(args) < 3:
        sys.exit(__doc__)
    csv_path, claims_path, company = args[0], args[1], args[2]

    df = add_metrics(load(csv_path))
    table = facts_table(df)
    flag_list = flags(df)
    claims = "\n".join(l for l in open(claims_path, encoding="utf-8").read().splitlines()
                       if l.strip() and not l.startswith("#"))

    analyst_prompt = build_analyst_prompt(company, table, claims, flag_list)
    if dry:
        print(analyst_prompt)
        print("\n[dry run: no LLM call made]")
        return

    hypothesis = call_llm(analyst_prompt)
    has_volume = df["volume_growth_pct"].notna().any()
    critic_prompt = build_critic_prompt(company, table, hypothesis, has_volume, has_peers=False)
    critique = call_llm(critic_prompt)

    report = (f"# {company}: research report (auto-generated)\n\n"
              f"## Analyst\n{hypothesis}\n\n## Critic\n{critique}\n\n"
              f"## Data table\n{table}\n")
    open("report.md", "w", encoding="utf-8").write(report)
    print(report)


if __name__ == "__main__":
    main()
