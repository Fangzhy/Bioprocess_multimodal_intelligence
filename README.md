# Bioprocess Multimodal Intelligence Platform

Analyze bioreactor runs by combining process sensor data, experimental metadata, scientist notes, and microscopy images to understand culture behavior, identify anomalous runs, retrieve similar historical experiments, and explain factors associated with final outcomes such as titer or viability.

## Application status

Milestones 1 through 3 are complete. The minimal Streamlit application, project skeleton, reproducible synthetic demonstration dataset, and SQLite database are implemented. Analytics will be added in later milestones; the functions below describe the intended application.

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

`requirements.txt` contains the application, ML, embedding, vector-database, and FastAPI dependencies. `requirements-dev.txt` adds JupyterLab, pytest, HTTP test support, and Ruff. The launch commands for the Streamlit and FastAPI applications will be added after their entry points are implemented.

The dependency stack was smoke-tested on Python 3.12.14 with Streamlit 1.65.0, scikit-learn 1.9.1, XGBoost 3.4.1, ChromaDB 1.5.9, Sentence Transformers 5.7.0, OpenCLIP 3.3.0, and FastAPI 0.142.2. Checkpoints for Sentence Transformers and CLIP have not been downloaded yet; their exact model identifiers will be selected in Milestones 6 and 7.

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
