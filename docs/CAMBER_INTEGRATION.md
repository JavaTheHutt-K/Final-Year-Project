# Camber research environment

Camber is an optional research and data-analysis environment for this project. It
does not replace the FastAPI backend, Supabase, Google Cloud Storage, Cloud Run,
Flower, or the existing centralised/federated experiment.

## Compatibility and authentication

The official Camber MCP documentation describes a remote HTTP server at
`https://camber-mcp.cambercloud.com/mcp`. Authentication is OAuth through a
supported client; Camber does not require an API key. The documented clients are
Claude, Cursor, Antigravity IDE, and ChatGPT. This VS Code Copilot session has
not configured or authenticated that server, and no `.vscode` MCP configuration
was added. Use the official setup instructions for one of those supported
clients, then verify the connection there. Never copy OAuth tokens or client
secrets into this repository.

## Specialist agent brief

Create a Camber agent with:

- **Name:** Intrusion Detection Research Analyst
- **Alias:** `intrusion-detection-research`

Use the following instruction as the agent's system brief:

> Analyse this intrusion-detection project using its code, documented
> configuration, approved sample data, and measured experiment outputs as the
> source of truth. Distinguish centralised training, federated training,
> per-client training partitions, global-model evaluation, training metrics,
> and holdout test metrics. Produce reproducible Python scripts, charts, and
> tables for CICIDS2017 and compare accuracy, precision, recall, F1-score,
> false-positive rate, and training time only when those values are measured.
> Explain limitations, possible leakage, and fairness of comparisons. Label
> hypotheses as hypotheses. Never invent results, change production records,
> trigger training jobs, change configuration, or expose credentials.

The local utility in `research/analyze.py` produces dataset and results summaries
that can be supplied to this agent after review.

## Safe data access

Camber connectors are configured in the Camber account, not in this repository.
The existing application uses a backend-only Supabase service credential, so that
credential must never be reused for Camber. If a PostgreSQL connector is
available in the Camber account, use a separate read-only role restricted to
approved views or tables (`experiment_runs` and selected stored results). Do not
grant write, schema, training-job, or configuration permissions. Do not query
`app_settings` unless a specific non-secret setting is required.

The repository does not assume that Camber supports direct Google Cloud Storage
access. Do not migrate the dataset or upload the 684 MB source file. Start with a
small, de-identified, representative sample only after explicit approval.

## Context Mirror review manifest

No synchronization was performed. Before any future Context Mirror sync, show and
approve the exact file list. The proposed allowlist is:

- source files under `backend/`, `frontend/src/`, and `research/`
- `README.md`, `config.yaml`, and dependency manifests
- this document and the specialist-agent brief
- selected, reviewed summary JSON or charts with no credentials or personal data

Always exclude:

- `.env*`, passwords, tokens, OAuth material, private keys, and service-account JSON
- `.git/`, `.venv/`, `venv/`, `frontend/node_modules/`, and build caches
- `data/` raw datasets and all CSV files unless a specific sample is approved
- `results/models/`, joblib/pickle artifacts, and unreviewed experiment outputs
- `results/` by default, temporary files, logs, and personal documents
- `staged-review.txt` and any other local review material

These exclusions supplement the repository's `.gitignore` and `.dockerignore`;
they are a review policy, not an instruction to delete local files.

## Local analysis

The analysis utility performs no network access, upload, database write, model
training, or cloud operation:

```powershell
python -m research.analyze dataset data\raw\cicids2017_cleaned.csv --output results\research_dataset_summary.json
python -m research.analyze results results\results.json --output results\research_experiment_summary.json
```

The dataset command reads CSVs in chunks and reports row/column counts, numeric
feature count, missing values, duplicates, label distribution, and numeric
mean/standard deviation. CSV fields are read as strings first, then numeric
values are coerced per chunk. This prevents pandas from silently dropping a
column's values when different chunks infer different types; mixed columns
contribute all values that can be parsed as numbers.

Exact duplicate detection is enabled by default and uses a temporary,
disk-backed SQLite index. It counts every occurrence after the first, including
duplicates within one chunk, across chunks, and repeated multiple times. The
index uses disk space proportional to the number of distinct rows rather than
Python memory proportional to the full dataset. It is deleted when the command
finishes. Use `--duplicate-detection none` when duplicate counts are not needed;
this avoids the index but reports `duplicate_rows` as zero. Exact detection is
still subject to the CSV parser's representation of fields, while numeric
statistics are based on per-value numeric coercion.

The results command extracts measured metrics and evaluation metadata from the
existing experiment output. It does not infer accuracy or training duration
when the source result does not contain them.

## Limits and remaining setup

Check the actual Camber account terms before use; the project request's
40 compute hours, 50 GB storage, and 50 agent messages are treated as limits,
not guarantees. Begin with CPU analysis and small samples, reuse artifacts, and
obtain approval before any connector, synchronization, upload, or long-running
job. Future dashboard integration should use a documented, authenticated
server-side artifact flow; no Camber credentials or unverified Camber API were
added to the React application.
