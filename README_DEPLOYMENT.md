# Google Cloud Run deployment

The root Docker image runs the existing Gradio/FastAPI application from
`app:app`. Cloud Run supplies `PORT`; Uvicorn binds to `0.0.0.0` and uses that
port. The Vite frontend remains a separate Vercel deployment.

## What is deployed

- The FastAPI instance is `backend.api.app`; `app.py` mounts the Gradio
  dashboard on that same application and exports the ASGI application as
  `app`.
- API routes include:

  | Method | Path | Purpose |
  | --- | --- | --- |
  | GET | `/health` | Cloud Run health check |
  | GET | `/api/config` | Read experiment settings |
  | PUT | `/api/config` | Update validated dashboard settings |
  | GET | `/api/results` | Read results |
  | GET | `/api/training-history` | Read round history |
  | GET | `/api/metrics?model=...&test_set=...` | Read model metrics |
  | POST | `/api/auth/login` | Exchange the administrator password for a short-lived bearer token |
  | POST | `/api/ai/compare` | Request a post-run Groq interpretation (admin only) |
  | POST | `/api/insights` | Generate insights from the latest completed run; accepts optional `{"question":"..."}` (admin only) |
  | POST | `/api/run` | Queue an experiment in the configured Cloud Run Job (admin only) |
  | GET | `/api/run/status` | Read persisted run status |

- `POST /api/run` saves the active configuration and queued run in Supabase,
  then launches the configured Cloud Run Job. Training does not run in the API
  service process. The Job downloads the configured GCS object and executes the
  existing Flower/Ray experiment.
- The app uses Flower 1.13.1 with Ray 2.10.0. Ray starts a local simulation
  with one CPU per simulated client (three by default). This is CPU- and
  memory-intensive; allocate enough Cloud Run CPU for the clients plus the API
  process and benchmark memory using the real dataset before increasing the
  sample fraction or client count.
- PyTorch is pinned to the CPU-only `2.8.0+cpu` wheel. The dashboard's CUDA
  check reports availability but does not accelerate the scikit-learn or Ray
  training path; it normally reports `false` on CPU-only Cloud Run instances.

## Build and run locally

From the repository root:

```sh
docker build -t ids-api .
docker run --rm -p 8080:8080 \
  -e FRONTEND_URL=http://localhost:5173 \
  -e IDS_DATA_DIR=/mnt/datasets \
  -v /absolute/path/to/dataset-directory:/mnt/datasets:ro \
  ids-api
```

The mounted directory must contain
`cicids2017_cleaned.csv`. Open `http://localhost:8080/health` to verify the
service and `http://localhost:8080/` for the Gradio dashboard.

## Cloud Run configuration

1. Put `cicids2017_cleaned.csv` in a Cloud Storage bucket. Do not copy the
   dataset into the image or commit it to the repository.
2. Build and deploy the image in your Google Cloud project. The API service
   does not need the dataset; the Cloud Run Job downloads it through the
   Google Cloud Storage API using `IDS_GCS_BUCKET` and the configured object
   name.
3. Configure `FRONTEND_URL` to the exact HTTPS origin of the Vercel deployment,
   with no path or trailing slash, for example
   `https://your-project.vercel.app`. `FRONTEND_URL` takes precedence over the
   legacy `IDS_CORS_ORIGINS` setting. CORS only accepts exact HTTP(S) origins;
   wildcard origins are rejected. Set a comma-separated `IDS_CORS_ORIGINS`
   only if multiple production origins (for example, a Vercel preview) are
   intentionally allowed.
4. Set `VITE_API_BASE_URL` in Vercel to the canonical Cloud Run service URL
   `https://ids-api-bo5h4puefa-bq.a.run.app` (without a trailing slash and
   without `/api`), then rebuild/redeploy the frontend.
5. Keep one Uvicorn worker for predictable request handling. Supabase is the
   authoritative store for configuration, run status, history, and results;
   local YAML/JSON files are only development and worker scratch copies.
6. Configure `SUPABASE_URL` and `SUPABASE_KEY` on both the API service and Job.
   The key is a backend secret and must never be exposed to Vercel.
   Managed deployments automatically require Supabase; `IDS_REQUIRE_DATABASE`
   can also be set to `true` outside Cloud Run to prevent local storage fallback.
7. Configure `CLOUD_RUN_PROJECT`, `CLOUD_RUN_REGION`, and `CLOUD_RUN_JOB` on
   the API service. Its service account needs permission to run that Job.
   Configure `IDS_GCS_BUCKET` on the Job and grant its service account object
   read access. Deploy the same image as a Cloud Run Job with the command
   `python -m backend.job_worker`; the API passes `IDS_RUN_ID` when it starts
   an execution. The object name comes from `dataset.gcs_object` (or
   `dataset.source_file` when omitted).
8. Configure `AI_PROVIDER=groq`, `GROQ_API_KEY`, and optionally `GROQ_MODEL` on
   the API service. The default model is `openai/gpt-oss-120b`, called through
   the official `groq` Python SDK. Never use `VITE_*` variables for this key.
   Groq failures,
   rate limits, exhausted quotas, timeouts, invalid credentials, unavailable
   models, and empty responses use the deterministic metrics fallback.
9. Create a PBKDF2-SHA256 password hash for `ADMIN_PASSWORD_HASH`, set a random
   `ADMIN_TOKEN_SECRET` of at least 32 bytes, and set `GRADIO_ADMIN_USERNAME`.
   The API login endpoint returns a one-hour signed bearer token. The Gradio
   start action requires its configured administrator login. Read-only API
   endpoints and `/health` remain public.

For example, the following shows the relevant Cloud Run deployment options;
replace the project, region, bucket, and image values with your own:

```sh
gcloud run deploy ids-api \
  --image REGION-docker.pkg.dev/PROJECT/REPOSITORY/ids-api:TAG \
  --region REGION \
  --port 8080 \
  --cpu 4 \
  --memory 8Gi \
  --timeout 3600 \
  --max-instances 1 \
  --set-env-vars FRONTEND_URL=https://your-project.vercel.app,IDS_DATA_DIR=/mnt/datasets,IDS_RESULTS_DIR=/tmp/ids-results \
  --no-allow-unauthenticated

gcloud run jobs deploy ids-training-job \
  --image REGION-docker.pkg.dev/PROJECT/REPOSITORY/ids-api:TAG \
  --region REGION \
  --command python \
  --args=-m,backend.job_worker \
  --tasks 1 \
  --max-retries 0 \
  --set-env-vars IDS_DATA_DIR=/tmp/ids-data,IDS_RESULTS_DIR=/tmp/ids-results,IDS_GCS_BUCKET=DATA_BUCKET
```

The CPU, memory, request timeout, and maximum instances above are only an
initial example, not measured sizing guarantees. Tune them using the actual
dataset, feature count, sample fraction, and concurrent request load. The
simulation requests one CPU for each Flower client. A 3-client run therefore
needs capacity beyond those client CPUs for Ray, preprocessing, and the API.

## Training reliability and persistence

Supabase is the authoritative run/config/results store. Worker output files are
temporary; completed results and training history are persisted to the
`experiment_runs` table. The API checks Supabase before accepting another run,
and does not report a run as started if Cloud Run Job launch fails. The worker
validates the UUID, downloads the dataset into its temporary directory, and
marks failures in Supabase before exiting non-zero.

The API and Job use the same image but different commands: the service starts
`app:app`, while the Job starts `python -m backend.job_worker`. Training is
therefore not run in the API process.

## Security

Administrative settings, experiment start, and Groq endpoints require a
valid short-lived signed administrator bearer token. Password verification uses
a PBKDF2-SHA256 hash; tokens are HMAC-signed and expire after one hour. Gradio's
administrative launch action is gated by its authenticated username and the
same password hash. Configure the service with `--no-allow-unauthenticated`
when possible. Public read-only API routes do not reveal service credentials.

The example deployment uses `--no-allow-unauthenticated`. That protects the
service, but a browser on Vercel cannot directly call a private Cloud Run
service without an authentication-aware proxy or frontend server-side
integration. Do not change to `--allow-unauthenticated` for direct Vercel
browser access until authentication, authorization, and run limits have been
added or an equivalent trusted access-control layer is in place. Do not put a
service credential or admin token in a `VITE_*` variable; Vite embeds those
values in publicly served frontend assets.

Incoming settings are validated by the API. Cloud Run launch errors and
worker failures are persisted to the run record. The interactive docs/OpenAPI
routes remain disabled.

## Supabase environment

The application expects the existing `app_settings` and `experiment_runs`
tables with the documented columns. Store `SUPABASE_URL` and `SUPABASE_KEY` as
secrets in both API and Job environments. No live database or cloud resources
are created or modified by this repository.

Use Secret Manager for credentials when database integration is added. Do not
commit a real `.env` file. `.env.example` contains placeholders only, and the
repository and Docker build ignore real `.env` files.
