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
  | POST | `/api/ai/compare` | Request a post-run Gemini interpretation (admin only) |
  | POST | `/api/insights` | Generate insights from the latest completed run (admin only) |
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
docker build -t adaptive-ids-api .
docker run --rm -p 8080:8080 \
  -e FRONTEND_URL=http://localhost:5173 \
  -e IDS_DATA_DIR=/mnt/datasets \
  -v /absolute/path/to/dataset-directory:/mnt/datasets:ro \
  adaptive-ids-api
```

The mounted directory must contain
`cicids2017_cleaned.csv`. Open `http://localhost:8080/health` to verify the
service and `http://localhost:8080/` for the Gradio dashboard.

## Cloud Run configuration

1. Put `cicids2017_cleaned.csv` in a Cloud Storage bucket. Do not copy the
   dataset into the image or commit it to the repository.
2. Build and deploy the image in your Google Cloud project. Mount the bucket
   read-only at `/mnt/datasets` (the image's default `IDS_DATA_DIR`), or set
   `IDS_DATA_DIR` to the mount path containing the CSV.
3. Configure `FRONTEND_URL` to the exact HTTPS origin of the Vercel deployment,
   with no path or trailing slash, for example
   `https://your-project.vercel.app`. `FRONTEND_URL` takes precedence over the
   legacy `IDS_CORS_ORIGINS` setting. CORS only accepts exact HTTP(S) origins;
   wildcard origins are rejected. Set a comma-separated `IDS_CORS_ORIGINS`
   only if multiple production origins (for example, a Vercel preview) are
   intentionally allowed.
4. Set `VITE_API_BASE_URL` in Vercel to the deployed Cloud Run service URL
   (without `/api`), then rebuild/redeploy the frontend.
5. Keep one Uvicorn worker. Run status, active training history, and
   configuration edits are held in process memory or local files and are not
   shared across Cloud Run instances.
6. Configure `SUPABASE_URL` and `SUPABASE_KEY` on both the API service and Job.
   The key is a backend secret and must never be exposed to Vercel.
7. Configure `CLOUD_RUN_PROJECT`, `CLOUD_RUN_REGION`, and `CLOUD_RUN_JOB` on
   the API service. Its service account needs permission to run that Job.
   Configure `IDS_GCS_BUCKET` on the Job and grant its service account object
   read access. The object name comes from `dataset.gcs_object` (or
   `dataset.source_file` when omitted).
8. Configure `GEMINI_API_KEY` and, optionally, `GEMINI_MODEL` on the API
   service. The default model is `gemini-2.5-flash`, called through the
   `google-genai` SDK. Never use `VITE_*` variables for this key.
9. Create a PBKDF2-SHA256 password hash for `ADMIN_PASSWORD_HASH`, set a random
   `ADMIN_TOKEN_SECRET` of at least 32 bytes, and set `GRADIO_ADMIN_USERNAME`.
   The API login endpoint returns a one-hour signed bearer token. The Gradio
   start action requires its configured administrator login. Read-only API
   endpoints and `/health` remain public.

For example, the following shows the relevant Cloud Run deployment options;
replace the project, region, bucket, and image values with your own:

```sh
gcloud run deploy adaptive-ids-api \
  --image REGION-docker.pkg.dev/PROJECT/REPOSITORY/adaptive-ids-api:TAG \
  --region REGION \
  --port 8080 \
  --cpu 4 \
  --memory 8Gi \
  --timeout 3600 \
  --max-instances 1 \
  --set-env-vars FRONTEND_URL=https://your-project.vercel.app,IDS_DATA_DIR=/mnt/datasets,IDS_RESULTS_DIR=/tmp/ids-results \
  --add-volume name=dataset,type=cloud-storage,bucket=DATA_BUCKET,readonly=true \
  --add-volume-mount volume=dataset,mount-path=/mnt/datasets \
  --no-allow-unauthenticated
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
and does not report a run as started if Cloud Run Job launch fails. Use a single
Cloud Run API instance to reduce start races; for strict multi-instance
concurrency guarantees, enforce a unique active-run constraint or an atomic
database lock in Supabase.

The current in-service training route is retained for compatibility, but
should be treated as best-effort only. For reliable production training, move
the computation to a Cloud Run Job (or another durable worker), launch it
through a queue/job API, and store job state/results in durable storage such as
Cloud Storage or the existing planned Supabase integration. This separation
has not been implemented here; it does not remove or replace the current
Flower/Ray training behavior.

## Security

Administrative settings, experiment start, and Gemini endpoints require a
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
