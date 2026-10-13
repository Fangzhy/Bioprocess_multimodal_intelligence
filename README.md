# Bioprocess Multimodal Intelligence Platform

[Launch live demo](https://bioprocessmultimodalintelligence-mkdj9hrgkbmgj4y5jluwjv.streamlit.app/)

Analyze synthetic bioreactor runs by combining process sensor data, experimental metadata, scientist notes, and microscopy-style images to understand culture behavior, identify anomalous runs, retrieve similar historical experiments, and explain factors associated with final outcomes such as titer or viability.

## Application status

The application has twelve pages covering process analytics, semantic and image retrieval, event alignment, evidence fusion, tabular and multimodal modeling, new-run predictions, and an evidence-grounded OpenRouter scientific copilot. Predictions can run inside Streamlit or through an optional FastAPI service.

The repository includes the synthetic dataset, SQLite and ChromaDB snapshots, embedding matrices, and trained model artifacts, so the dashboard and built-in prediction examples can run without regenerating the data or retraining models. Free-text semantic search downloads the MiniLM checkpoint on first use if it is not cached, and copilot explanations need an OpenRouter API key.

## Application functions

| Page | Function |
| --- | --- |
| Overview | Summarize batches, process outcomes, and key observations |
| Batch Explorer | Plot selected process variables and inspect associated notes and images |
| Batch Comparison | Compare individual runs with other batches and historical references |
| Data Quality | Inspect missing values, duplicates, outliers, and data consistency |
| Predictive Modeling | Compare Linear Regression, PLS, Random Forest, and XGBoost using tabular process features |
| Text Intelligence | Search scientist notes semantically using Sentence Transformers and ChromaDB |
| Microscopy Intelligence | Retrieve similar microscopy images using CLIP and ChromaDB |
| Multimodal Investigation | Align sensor data, notes, and images within event windows and inspect retrospective outcome context |
| Multimodal Analytics | Rank sensor, text, and image anomaly evidence and combine it into an exploratory investigation score |
| Multimodal Predictive Modeling | Compare model families trained on fused sensor, text, and image features with tabular-only baselines |
| New Run Prediction | Upload metadata and sensor data for baseline predictions, with optional notes and images for multimodal predictions |
| Scientific Copilot | Explain retrieved evidence using an LLM through OpenRouter |

## Technology stack

| Component | Technology |
| --- | --- |
| Frontend web application | Streamlit |
| Model inference API | FastAPI and Pydantic |
| Relational database | SQLite |
| Vector database | ChromaDB |
| Text embeddings | Sentence Transformers |
| Image embeddings | CLIP |
| Modeling | scikit-learn and XGBoost |
| Visualization | Plotly |
| Data processing | NumPy and pandas |
| Generative AI | LLM explanations through OpenRouter |
| Frontend hosting target | Streamlit Community Cloud |
| Optional backend hosting | An HTTPS-accessible Python/ASGI service |

## Setup

The commands below target Windows PowerShell with Python 3.12 and uv. Run them from the repository root. Docker is not required.

Create the environment and install the application dependencies:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The default **Overview** page displays dataset counts, outcome charts, and a batch summary loaded from SQLite.

The commands use the virtual environment's Python directly. To use shorter commands such as `python` or `streamlit` interactively, activate the environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

To try inference, open **New Run Prediction**, keep **Built-in demo** selected, click **Load tabular example** or **Load multimodal example**, and then click **Request predictions**. Both examples return final-titer estimates from all four model families without starting FastAPI.

`app.py` is the navigation entry point. It registers every page explicitly with `st.navigation()` and `st.Page()`, including the default **Overview** page implemented in `views/overview.py`. Sidebar labels and icons therefore come from `app.py` rather than from filenames.

The core process dashboard includes:

- **Overview** summarizes dataset size, final outcomes, titer distribution, and the relationship between maximum VCD and final titer.
- **Batch Explorer** plots selected process trajectories and shows batch statistics, scientist notes, and synthetic microscopy images.
- **Batch Comparison** compares a selected run with a metadata-based historical cohort using the reference mean and one standard-deviation band.
- **Data Quality** reports missing values, duplicate keys, engineering-range violations, IQR outliers, and time-series coverage.

The comparison cohorts use cell-line and media metadata and always exclude the selected batch. They do not use the planted abnormal-scenario labels stored in the evaluation data.

## Tabular predictive modeling

Train the tabular models from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.models.train_tabular
```

The workflow creates one row per batch from process summaries and experimental metadata, then compares Linear Regression, PLS, Random Forest, and XGBoost. It uses a fixed 40-batch training set, a 10-batch holdout set, and five-fold cross-validation inside the training set. All preprocessing is part of each fitted pipeline.

The feature table excludes batch identifiers, dates, media lots, planted scenario labels, and all titer trajectory values. Because features use measurements through 240 hours, the experiment is an end-of-run prediction rather than an early forecast.

Generated outputs are stored in `artifacts/tabular/`:

- Complete versioned preprocessing-and-model pipelines for later FastAPI inference
- A manifest containing the input schema, batch split, target units, library versions, metrics, and limitations
- Batch features, held-out predictions, model metrics, and feature-importance summaries

Open **Predictive Modeling** in Streamlit to compare held-out RMSE, MAE, and R², inspect predicted-versus-actual and residual plots, and review model feature signals. Metrics from 50 synthetic batches are illustrative and have substantial sampling uncertainty.

## Embeddings and multimodal retrieval

Build both ChromaDB collections and reusable embedding matrices with:

```powershell
.\.venv\Scripts\python.exe -m src.embeddings.store
```

The text collection uses `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional normalized embeddings. The image collection uses OpenCLIP `ViT-B-32` with the `laion2b_s34b_b79k` checkpoint and 512-dimensional normalized embeddings. Both collections use cosine distance.

- **Text Intelligence** searches 200 synthetic scientist records by meaning.
- **Microscopy Intelligence** retrieves similar synthetic microscopy illustrations and joins their batch outcomes.
- **Multimodal Investigation** aligns sensor readings, notes, and image captures to an inclusive event window and shows the final outcome as retrospective context.

The first embedding build downloads public model weights to the local cache and requires internet access. Chroma data is stored under `database/chroma/`, while reusable matrices and the encoder manifest are stored under `data/processed/embeddings/`.

Image-to-image retrieval uses the committed Chroma vectors without loading CLIP. Free-text note search encodes each query with MiniLM, tries the local model cache first, and downloads missing checkpoint files from Hugging Face as needed. The loaded query encoder is reused across searches. A fresh Streamlit deployment does not need to rebuild the stored embeddings before searching, but its first query needs internet access to download the encoder. If loading or searching fails, the page displays a retry message and records the details in the app logs.

## Multimodal analytics and modeling

**Multimodal Analytics** combines percentile-ranked sensor, text, and image anomaly evidence with the exploratory weights 0.5, 0.2, and 0.3. Missing modality weights are renormalized. This score helps prioritize investigations and is not a calibrated failure probability.

Train the controlled feature-level fusion experiment with:

```powershell
.\.venv\Scripts\python.exe -m src.models.train_multimodal
```

The experiment applies PCA inside every training fold and compares the same four model families and held-out batches across tabular only, tabular plus text, tabular plus image, and complete fusion. It evaluates 20 text and 20 image principal components. The tabular manifest and stored embeddings must exist before training. Outputs in `artifacts/multimodal/` include ablation metrics, held-out predictions, fused batch features, a manifest, and four complete-fusion pipelines.

With the current 50-batch synthetic dataset, full multimodal Random Forest achieved the best held-out RMSE, but this result is highly uncertain because the sample count is small and notes and images derive from the same simulator state. **Multimodal Predictive Modeling** shows the complete ablation table, regression metrics, residuals, PCA dimensions, and low-titer event metrics.

## FastAPI inference backend

Start the backend from the repository root in a separate terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Verify that both trusted artifact registries load |
| `GET /v1/models` | List model IDs, versions, target units, and cutoff |
| `POST /v1/predict/tabular` | Predict from metadata JSON and a sensor CSV |
| `POST /v1/predict/multimodal` | Add an optional notes JSON file and microscopy images |

Requests supply raw run data; the backend calculates the same features and embeddings used during training. It never accepts uploaded models or filesystem paths. The multimodal response reports supplied and missing modalities, model version, target units, cutoff, low-titer threshold, and one prediction per model family.

Prediction endpoints accept `multipart/form-data`: `metadata` is a JSON string and `sensor_file` is a CSV upload. The multimodal endpoint also accepts `notes_file` and repeated `images` fields. The backend limits each file to 5,000,000 bytes and accepts up to six images.

Metadata must include `batch_id`, `cell_line`, `media_type`, `bioreactor_scale_l`, `seed_density_million_ml`, and `feed_strategy`. Required sensor columns are:

```text
time_hr, temperature_c, ph, do_pct, agitation_rpm, air_flow_slpm,
o2_flow_slpm, co2_flow_slpm, feed_rate_ml_hr, glucose_g_l, lactate_g_l,
viable_cell_density_million_ml, viability_pct
```

Sensor timestamps must be unique and between 0 and 240 hours. Notes are a JSON list of strings or objects with a `content` field. Download the built-in examples from **New Run Prediction** to inspect complete request files.

Open **New Run Prediction** and keep **Built-in demo** selected to run the trusted pipelines directly inside Streamlit. This is the default for Streamlit Community Cloud and does not require a second service.

Select **FastAPI service** when testing the service architecture locally or after deploying the backend separately. `localhost:8000` is valid only on the computer running FastAPI; it does not refer to your computer from Streamlit Community Cloud. For a hosted API, set `BIOPROCESS_API_URL` to its public HTTPS URL. Set the same optional `BIOPROCESS_API_TOKEN` in both environments to require the `X-API-Key` header. See `.env.example` for variable names.

The API settings are read from environment variables at startup. Set them in PowerShell before launching the processes, or add `--env-file .env` to the Uvicorn command to load a local configuration file. The prediction page reads its API URL and token from environment variables; copying `.env.example` alone does not configure that page.

The page includes **Load tabular example** and **Load multimodal example** buttons. Each loads an editable synthetic request derived from B018 and relabeled `NEW_DEMO_001`. Users can inspect and download the metadata JSON, 61-row sensor CSV, and timestamped notes JSON, and view three illustrative microscopy images. Requests run through the shared prediction service in **Built-in demo** mode or the normal API endpoints in **FastAPI service** mode. The example request excludes titer and final outcomes.

In **Built-in demo** mode, the unchanged multimodal example reuses its committed pooled text and image embeddings, so it works without downloading MiniLM or CLIP. Custom notes and images, edits to the example notes, and FastAPI requests containing notes or images require the corresponding encoders. Missing public checkpoints are downloaded as needed.

## Scientific copilot

For local copilot explanations, copy `.env.example` to `.env` if you do not already have a configuration file, then set:

```text
OPEN_ROUTER_API=your-key
OPENROUTER_MODEL=openrouter/free
```

The configured model is tried first, with `openrouter/free` as fallback. **Scientific Copilot** builds a traceable evidence bundle from SQLite, ChromaDB, and calculated cohort comparisons before making the request. The evidence stays visible if the external service is unavailable. The prompt requires measured observations, labels causal interpretations as hypotheses, and states that the demonstration data and microscopy are synthetic.

The copilot loads `.env` when an explanation is requested and preserves existing environment variables. `OPENROUTER_API_KEY` is also accepted as an alternative key name.

For Streamlit deployment, add top-level keys to the app's Secrets settings. Local Streamlit secrets can use `.streamlit/secrets.toml` with the same format:

```toml
OPEN_ROUTER_API = "your-key"
OPENROUTER_MODEL = "openrouter/free"
```

If using a separately hosted FastAPI service, also set `BIOPROCESS_API_URL` and the optional `BIOPROCESS_API_TOKEN` as top-level Streamlit secrets, and configure the same token in the backend environment. Both `.env` and `.streamlit/secrets.toml` are excluded by `.gitignore`.

OpenRouter availability and rate limits depend on the selected model and service. Evidence inspection remains available without a key or a successful external response.

## Demo data

The demo contains 50 synthetic bioreactor runs, each spanning 10 days with measurements every 4 hours. It includes 3,050 sensor rows, experimental metadata, 200 scientist notes, 150 microscopy-style images, and final outcomes. Five planted abnormal scenarios are stored separately for evaluation.

Regenerate the dataset from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.data.generate_data
```

Generated tables are stored under `data/raw/`, scenario truth is stored under `data/evaluation/`, and PNG images are stored under `images/`. The generated `data/raw/manifest.json` records the seed, simulator version, units, assumptions, row counts, and table hashes. The notebook `notebooks/01_generate_data.ipynb` explains and plots the data-generation workflow.

All generated records are synthetic educational data. The microscopy-style images are programmatic illustrations, not real microscopy measurements, and the simulator is not a validated biological process model.

## SQLite database

Build or rebuild the database from the generated CSV files:

```powershell
.\.venv\Scripts\python.exe -m src.data.build_database
```

This creates `database/bioprocess.db` from `database/schema.sql`. Rebuilding is idempotent: the loader validates the source relationships, builds a temporary database, runs SQLite integrity and foreign-key checks, and then atomically replaces the existing snapshot.

The database contains:

| Table | Rows | Purpose |
| --- | ---: | --- |
| `batches` | 50 | Experimental metadata |
| `sensor_data` | 3,050 | Time-series process measurements |
| `outcomes` | 50 | Final titer, viability, peak VCD, and quality score |
| `text_records` | 200 | Timestamped descriptions, notes, and observations |
| `images` | 150 | Metadata and paths for illustrative microscopy images |

Explore SELECT, WHERE, JOIN, GROUP BY, and parameterized queries in `notebooks/02_query_database.ipynb`.

## Rebuild the complete demo

To regenerate all dependent outputs, run these commands in order from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.data.generate_data
.\.venv\Scripts\python.exe -m src.data.build_database
.\.venv\Scripts\python.exe -m src.models.train_tabular
.\.venv\Scripts\python.exe -m src.embeddings.store
.\.venv\Scripts\python.exe -m src.models.train_multimodal
```

These commands replace the generated demo data, database, embedding collections, and model artifacts. Rebuild downstream outputs after changing the simulator or source data so the snapshots and models stay consistent. Encoder checkpoints require internet access on first use.

## Development and verification

`requirements.txt` contains the full application stack. `requirements-backend.txt` provides inference-service dependencies. `requirements-dev.txt` adds JupyterLab, pytest, HTTP test support, and Ruff to the application stack.

Install the development dependencies and run the existing tests:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover synthetic-data reproducibility, SQLite integrity, dashboard queries, feature exclusions, model artifacts, embeddings, event alignment, API predictions, built-in examples, copilot request handling, and Streamlit pages. Tests use the demo snapshots and artifacts; OpenRouter responses are mocked.

The tabular manifest records training-library versions, the feature schema, and batch split. The embedding manifest records the MiniLM and CLIP checkpoints used to build the stored vectors.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Main Streamlit entry point |
| `views/overview.py` | Default Overview page |
| `pages/` | Eleven additional Streamlit pages registered in `app.py` |
| `backend/` | FastAPI endpoints, request schemas, settings, and trusted artifact registry |
| `artifacts/tabular/` | Tabular pipelines, feature tables, metrics, and manifest |
| `artifacts/multimodal/` | Full-fusion pipelines, ablation results, feature tables, and manifest |
| `data/raw/` | Generated synthetic CSV tables and dataset manifest |
| `data/processed/embeddings/` | Reusable text and image embedding matrices and encoder manifest |
| `data/evaluation/` | Planted scenario truth kept separate from normal model inputs |
| `database/` | Versioned SQLite schema and reproducible database snapshot |
| `images/` | Synthetic microscopy-style illustrations used by the demo |
| `notebooks/` | Step-by-step experiments before stable code moves into `src/` |
| `src/data/` | Reusable data generation, validation, and loading code |
| `src/models/` | Feature engineering, model training, evaluation, and evidence fusion |
| `src/embeddings/` | Sentence Transformer, CLIP, and ChromaDB integration code |
| `src/inference/` | Input validation, built-in examples, and shared prediction service |
| `src/copilot.py` | Evidence bundle construction and OpenRouter client |
| `src/visualization/` | Shared Plotly and Streamlit visualization helpers |
| `tests/` | Data, model, retrieval, API, copilot, and Streamlit tests |
| `.env.example` | OpenRouter and optional FastAPI configuration template |
| `requirements.txt` | Runtime and application dependencies |
| `requirements-backend.txt` | Dependencies for the separate inference service |
| `requirements-dev.txt` | Runtime dependencies plus notebook and test tools |

## License

Released under the [MIT License](LICENSE).
