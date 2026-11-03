# Google Cloud Run deployment

The root Docker image runs the existing Gradio/FastAPI application from
`app:app`. Cloud Run supplies `PORT`; Uvicorn binds to `0.0.0.0` and uses that
port. The Vite frontend remains a separate Vercel deployment.

## What is deployed

- The FastAPI instance is `backend.api.app`; `app.py` mounts the Gradio
  dashboard on that same application and exports the ASGI application as
  `app`.
- Existing API routes are unchanged:

  | Method | Path | Purpose |
  | --- | --- | --- |
  | GET | `/health` | Cloud Run health check |
  | GET | `/api/config` | Read experiment settings |
  | PUT | `/api/config` | Update validated dashboard settings |
  | GET | `/api/results` | Read results |
  | GET | `/api/training-history` | Read round history |
  | GET | `/api/metrics?model=...&test_set=...` | Read model metrics |
  | POST | `/api/run` | Queue an experiment in the API process |
  | GET | `/api/run/status` | Read in-process run status |

- `POST /api/run` does not load data at API startup. Dataset preparation
  begins only after a run is requested and reads the CSV in chunks. It then
  retains the configured sample in memory for splitting, preprocessing, and
  training. The default sample fraction is `0.1`; this is not fully streaming
  training.
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

The existing API starts Flower/Ray in a background thread and returns HTTP
202. Cloud Run manages request-serving instances, not durable background
workers: the service can be scaled down or restarted while a run is active.
The run status and progress are in memory, and local result/config files are
ephemeral and instance-local. `IDS_RESULTS_DIR` defaults to `/tmp/ids-results`
in the container, and Ray's temporary data defaults to `/tmp/ray`. These
locations are not durable storage.

The current in-service training route is retained for compatibility, but
should be treated as best-effort only. For reliable production training, move
the computation to a Cloud Run Job (or another durable worker), launch it
through a queue/job API, and store job state/results in durable storage such as
Cloud Storage or the existing planned Supabase integration. This separation
has not been implemented here; it does not remove or replace the current
Flower/Ray training behavior.

## Security

The API currently has **no authentication**. CORS is a browser-origin policy,
not access control: non-browser clients can still call `/api/config` and
`/api/run`. In particular, changing settings and starting CPU/memory-intensive
training are publicly available if the Cloud Run service allows unauthenticated
invocation.

The example deployment uses `--no-allow-unauthenticated`. That protects the
service, but a browser on Vercel cannot directly call a private Cloud Run
service without an authentication-aware proxy or frontend server-side
integration. Do not change to `--allow-unauthenticated` for direct Vercel
browser access until authentication, authorization, and run limits have been
added or an equivalent trusted access-control layer is in place. Do not put a
service credential or admin token in a `VITE_*` variable; Vite embeds those
values in publicly served frontend assets.

Incoming settings are validated by the existing API. Unexpected training and
file errors are logged server-side and the status response avoids returning
exception details. The interactive docs/OpenAPI routes remain disabled.

## Supabase environment

`DATABASE_URL`, `SUPABASE_URL`, and `SUPABASE_KEY` are reserved environment
variables for a future integration and are intentionally not set in the
image. The application does not currently connect to Supabase or any database;
settings, status, and results continue to use their existing YAML, in-memory,
and file-backed behavior. No database or credentials are created by this
deployment setup.

Use Secret Manager for credentials when database integration is added. Do not
commit a real `.env` file. `.env.example` contains placeholders only, and the
repository and Docker build ignore real `.env` files.
