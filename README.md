# TruthTrack: autonomous research agent for Consumer, FMCG & Retail (Track C)

Finding the Next Signal, ATMOS 2026 (BITS Pilani Hyderabad). Solo entry.

## What this is
A prototype of an agent that compares what a company says (management claims) with what its
filings show, then has a second LLM step try to disprove the result.

## Current status (honest)
- Prototype tested on one company: **Britannia**.
- Data was collected **by hand** from BSE/company filings (consolidated results) into `britannia.csv`,
  with source file and page for each row.
- **No volume data yet**, so the price vs volume split is not computed.
- Automatic data collection, peer comparison and a report UI are **planned, not built**.

## Files
- `metrics.py` - year-on-year growth, receivable days, inventory days, simple flags
- `agent.py` - analyst LLM step, critic LLM step, writes `report.md`
- `britannia.csv` - hand-collected data with sources
- `claims.txt` - management claims with sources

## Run
```
pip install openai huggingface_hub pandas tabulate
python metrics.py britannia.csv
python agent.py britannia.csv claims.txt Britannia --dry-run   # no LLM call
```
To call an LLM, set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (any OpenAI-compatible API),
or `HF_TOKEN` and `HF_MODEL`. Never commit keys.

## Design rules
- Numbers come from code, not the LLM.
- The LLM may use only the table and claims provided and must cite source file and page.
- Evidence strength is capped at Weak when volume data or peer comparison is missing.
