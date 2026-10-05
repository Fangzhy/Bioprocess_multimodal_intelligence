# Bioprocess Multimodal Intelligence Platform

Analyze bioreactor runs by combining process sensor data, experimental metadata, scientist notes, and microscopy images to understand culture behavior, identify anomalous runs, retrieve similar historical experiments, and explain factors associated with final outcomes such as titer or viability.

## Application status

All twelve milestones are complete. The project includes structured and multimodal analytics, semantic and image retrieval, event alignment, interpretable evidence fusion, controlled multimodal modeling, a FastAPI inference backend, and an evidence-grounded OpenRouter scientific copilot.

## Application functions

| Page | Function |
| --- | --- |
| Overview | Summarize batches, process outcomes, and key observations |
| Data Integration | Import and associate process measurements, metadata, notes, and images by batch and time |
| Data Quality | Inspect missing values, duplicates, outliers, and data consistency |
| Batch Explorer | Plot selected process variables and inspect associated notes and images |
| Batch Comparison | Compare individual runs with other batches and historical references |
| Statistical Analysis | Summarize distributions, variability, and relationships with outcomes |
| Predictive Modeling | Compare Linear Regression, PLS, Random Forest, and XGBoost using tabular process features |
| Text Intelligence | Search scientist notes semantically using Sentence Transformers and ChromaDB |
| Microscopy Intelligence | Retrieve similar microscopy images using CLIP and ChromaDB |
| Multimodal Investigation | Align sensor data, notes, and images within event windows and combine analytical evidence |
| Multimodal Predictive Modeling | Train the same model families on fused sensor, text, and image features and compare them with tabular-only baselines |
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
| Generative AI | Free LLM/VLM models through OpenRouter |
| Frontend hosting target | Streamlit Community Cloud |
| Backend hosting target | An HTTPS-accessible Python/ASGI service selected and documented during deployment |

## Setup

Local development targets Windows with Python and Git. The Streamlit frontend and FastAPI backend will run as separate local processes. Docker is not required for local learning, though a container may be added later if the selected backend host requires one.

Create and populate the development environment with uv:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
```

Activate it when running commands interactively:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the application from the repository root:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The welcome page should display the platform title and confirm that Streamlit and the Python environment are working.

`app.py` is the navigation entry point. It registers every page explicitly with `st.navigation()` and `st.Page()`, including the default **Overview** page implemented in `views/overview.py`. Sidebar labels and icons therefore come from `app.py` rather than from filenames.

The core process dashboard includes:

- **Overview** summarizes dataset size, final outcomes, titer distribution, and the relationship between maximum VCD and final titer.
- **Batch Explorer** plots selected process trajectories and shows batch statistics, scientist notes, and synthetic microscopy images.
- **Batch Comparison** compares a selected run with a metadata-based historical cohort using the reference mean and one standard-deviation band.
- **Data Quality** reports missing values, duplicate keys, engineering-range violations, IQR outliers, and time-series coverage.

The comparison cohorts use cell-line and media metadata and always exclude the selected batch. They do not use the planted abnormal-scenario labels stored in the evaluation data.

## Tabular predictive modeling

Train the Milestone 5 models from the repository root:

```powershell
python -m src.models.train_tabular
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
python -m src.embeddings.store
```

The text collection uses `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional normalized embeddings. The image collection uses OpenCLIP `ViT-B-32` with the `laion2b_s34b_b79k` checkpoint and 512-dimensional normalized embeddings. Both collections use cosine distance.

- **Text Intelligence** searches 200 synthetic scientist records by meaning.
- **Microscopy Intelligence** retrieves similar synthetic microscopy illustrations and joins their batch outcomes.
- **Multimodal Investigation** aligns sensor readings, notes, and image captures to an inclusive event window and shows the final outcome as retrospective context.

The first embedding build downloads public model weights to the local Hugging Face cache. Chroma data is stored under `database/chroma/`, while reusable matrices and the encoder manifest are stored under `data/processed/embeddings/`.

## Multimodal analytics and modeling

**Multimodal Analytics** combines percentile-ranked sensor, text, and image anomaly evidence with the exploratory weights 0.5, 0.2, and 0.3. Missing modality weights are renormalized. This score helps prioritize investigations and is not a calibrated failure probability.

Train the controlled feature-level fusion experiment with:

```powershell
python -m src.models.train_multimodal
```

The experiment applies PCA inside every training fold and compares the same four model families and held-out batches across tabular only, tabular plus text, tabular plus image, and complete fusion. It evaluates 20 text and 20 image principal components as an experiment. One-hot metadata and modality count indicators make the fitted feature totals differ from the conceptual 70-feature example.

With the current 50-batch synthetic dataset, full multimodal Random Forest achieved the best held-out RMSE, but this result is highly uncertain because the sample count is small and notes and images derive from the same simulator state. **Multimodal Predictive Modeling** shows the complete ablation table, regression metrics, residuals, PCA dimensions, and low-titer event metrics.

## FastAPI inference backend

Start the backend from the repository root in a separate terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Verify that both trusted artifact registries load |
| `GET /v1/models` | List model IDs, versions, target units, and cutoff |
| `POST /v1/predict/tabular` | Predict from metadata JSON and a sensor CSV |
| `POST /v1/predict/multimodal` | Add an optional notes JSON file and microscopy images |

Requests supply raw run data; the backend calculates the same features and embeddings used during training. It never accepts uploaded models or filesystem paths. The multimodal response reports supplied and missing modalities, model version, target units, cutoff, low-titer threshold, and one prediction per model family.

Open **New Run Prediction** in Streamlit after the API is running. Set `BIOPROCESS_API_URL` for a remote HTTPS backend. Set the same optional `BIOPROCESS_API_TOKEN` in both environments to require the `X-API-Key` header. See `.env.example` for variable names.

## Scientific copilot

Configure `.env` locally or Streamlit secrets in deployment:

```text
OPEN_ROUTER_API=your-key
OPENROUTER_MODEL=openrouter/free
```

The configured model is tried first, with `openrouter/free` as fallback. **Scientific Copilot** builds a traceable evidence bundle from SQLite, ChromaDB, and calculated cohort comparisons before making the request. The evidence stays visible if the external service is unavailable. The prompt requires measured observations, labels causal interpretations as hypotheses, and states that the demonstration data and microscopy are synthetic.

The integration was live-tested with the configured free-tier model on October 4, 2026. Free model availability and rate limits can change, so `openrouter/free` is used as the portable fallback.

`requirements.txt` contains the full application stack. `requirements-backend.txt` provides the packages needed by the inference service, while `requirements-dev.txt` adds JupyterLab, pytest, HTTP test support, and Ruff.

The dependency stack was smoke-tested on Python 3.12.14 with Streamlit 1.65.0, scikit-learn 1.9.1, XGBoost 3.4.1, ChromaDB 1.5.9, Sentence Transformers 5.7.0, OpenCLIP 3.3.0, and FastAPI 0.142.2. The selected MiniLM and CLIP checkpoints are recorded in the embedding manifest.

The deployed Streamlit app will call the FastAPI service over HTTPS. The API will load versioned, trusted model pipelines created during training; users will upload prediction data, not model files. The backend URL and an application API token will be configured through environment variables or Streamlit secrets.

When the OpenRouter integration is implemented, local credentials will use `.streamlit/secrets.toml`, which is already excluded by `.gitignore`. Hosted credentials will use Streamlit Community Cloud's Secrets settings. See [Streamlit secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Demo data

The demo contains 50 synthetic bioreactor runs, each spanning 10 days with measurements every 4 hours. It includes 3,050 sensor rows, experimental metadata, 200 scientist notes, 150 microscopy-style images, and final outcomes. Five planted abnormal scenarios are stored separately for evaluation.

Regenerate the dataset from the repository root:

```powershell
python -m src.data.generate_data
```

Generated tables are stored under `data/raw/`, scenario truth is stored under `data/evaluation/`, and PNG images are stored under `images/`. The generated `data/raw/manifest.json` records the seed, simulator version, units, assumptions, row counts, and table hashes. The notebook `notebooks/01_generate_data.ipynb` explains and plots the data-generation workflow.

All generated records are synthetic educational data. The microscopy-style images are programmatic illustrations, not real microscopy measurements, and the simulator is not a validated biological process model.

## SQLite database

Build or rebuild the database from the generated CSV files:

```powershell
python -m src.data.build_database
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

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Main Streamlit entry point |
| `data/raw/` | Source synthetic or imported files before cleaning |
| `data/processed/` | Cleaned datasets and prepared feature tables |
| `data/evaluation/` | Planted scenario truth kept separate from normal model inputs |
| `database/` | Versioned SQLite schema and reproducible database snapshot |
| `images/` | Synthetic or sourced microscopy images used by the demo |
| `notebooks/` | Step-by-step experiments before stable code moves into `src/` |
| `src/data/` | Reusable data generation, validation, and loading code |
| `src/models/` | Feature engineering, model training, evaluation, and inference code |
| `src/embeddings/` | Sentence Transformer, CLIP, and ChromaDB integration code |
| `src/visualization/` | Shared Plotly and Streamlit visualization helpers |
| `pages/` | Streamlit pages added during later milestones |
| `docs/` | Project planning and learning documentation |
| `tests/` | Reproducibility, integrity, and later application tests |
| `requirements.txt` | Runtime and application dependencies |
| `requirements-dev.txt` | Runtime dependencies plus notebook and test tools |
