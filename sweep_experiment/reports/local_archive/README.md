# Local analysis archive (laptop only — not pushed to GitHub)

**Purpose:** Store full cluster dump output on your personal machine so Cursor
agents have complete numbers without re-querying the cluster. Contents are
**gitignored**; only this README is tracked.

## Workflow

1. On cluster, run:
   ```bash
   bash scripts/dump_analysis_reports.sh 2026-06-30
   ```
2. Copy the **output** section (starts with `========== FILE INVENTORY ==========`) and paste into Cursor.
3. Say: **"File this under local_archive"** — agent writes to `local_archive/YYYY-MM-DD/` on your laptop.

Optional (on laptop, if repo is checked out locally):
```bash
bash scripts/ingest_local_archive_dump.sh 2026-06-30 cluster_output.txt
```

## Layout per date

```
local_archive/2026-06-30/
  SNAPSHOT.md           # distilled key numbers (agent-maintained)
  cluster_dump.txt      # full terminal paste (optional)
  reports/              # individual .md / .json extracted from dump
```

## What stays in git

| Tracked | Local only |
|---|---|
| `ANALYSIS_LOG.md` (headline findings) | `local_archive/**` |
| `experiment_outputs/YYYY-MM-DD.md` (interpretation) | Full report markdown, CSV heads, PNG paths |
| `scripts/dump_analysis_reports.sh` | |

Cluster remains source of truth for CSV/PNG binaries; laptop archive holds
**text** copies for agent context.
