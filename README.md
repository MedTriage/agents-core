# Agents Core

![](https://raw.githubusercontent.com/MedTriage/agents-core/main/assets/logo.png)

> Agentic AI for clinical triage, drug discovery, and remote digital healthcare, with a graduated autonomy architecture

The core repository housing the agents which constitute the agentic workflow.


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangGraph](https://img.shields.io/badge/langgraph-%231C3C3C.svg?style=for-the-badge&logo=langgraph&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

## Architecture

### Agentic Workflow

The system is built on a multi-agent architecture, built using LangGraph:

```
                           ┌─→ Parametric Agent ─┐
Scribe ─→ Intent Router ───┼─→ KG-RAG Agent ─────┼─→ Evidence Fusion ─→ Orchestrator ─→ Guardian
                           └─→ MCP Agent ────────┘
```

- **Scribe** - Keeps a running medical record for each conversation (conditions, symptoms, medications, allergies), available to every other agent
- **Intent Router** - Separates clinical queries from chitchat; chitchat goes straight to the orchestrator
- **Parametric Agent** - Assesses the case from the model's own clinical knowledge
- **KG-RAG Agent** - Checks candidate diagnoses against the SNOMED CT ontology and finds what tells them apart
- **MCP Agent** - Queries live medical sources (FDA, WHO, PubMed, RxNorm) through the medical-mcp server
- **Evidence Fusion** - Combines the KG-RAG and MCP evidence with Dempster-Shafer theory, independently of any model
- **Orchestrator** - Reconciles the agents, checks for hallucination and emergencies, and writes the response
- **Guardian Agent** - Assigns the risk level, with deterministic rules that a model cannot override downward

Each response also carries a patient-safe explanation of how it was reached: how many sources were consulted and whether they agreed, without naming a disease or showing raw numbers.

### Risk Levels

- **Level-I** - Simple queries and chitchat, directly answered to patient
- **Level-II** - Most clinical queries, where the assessment is sent to a doctor who verifies and approves it and sends it back to the patient
- **Level-III** - Potential Life-threatening scenarios, the agents lock themselves and communicate with local authorities in order to provide assistance to the patient

## Getting Started

### Prerequisites

- Python 3.10 or higher (ideally 3.11)
- Node.js and npm, for the MCP server
- Podman (or Docker), for the local PostgreSQL database
- A SNOMED CT RF2 release, which is free but licence-gated: request it from your National Release Center
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MedTriage/agents-core.git
cd agents-core
```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Build the SNOMED CT knowledge graph for the KG-RAG agent (writes `data/snomed.db`):
```bash
python -m app.rag.snomed_ingest /path/to/SnomedCT_.../Snapshot
```

5. Build the MCP server for the MCP agent:
```bash
cd vendor/medical-mcp && npm install && npm run build
```

### Configuration

Copy `.env.example` to `.env` and fill it in. `OPENROUTER_API_KEY` and the database settings are required. Without the Gmail settings (`SMTP_USER`, `SMTP_APP_PASSWORD`), emails queue in the database and are not sent.

### Database

Accounts, profiles, conversations and the doctor review queue live in PostgreSQL. Start it once (the image is about 300 MB), bound to this machine only:

```bash
set -a; . ./.env; set +a
podman run -d --name medtriage-db --restart unless-stopped \
  -e POSTGRES_USER=medtriage -e POSTGRES_PASSWORD="$MEDTRIAGE_DB_PASSWORD" -e POSTGRES_DB=medtriage \
  -p 127.0.0.1:5432:5432 -v medtriage-pgdata:/var/lib/postgresql/data \
  docker.io/library/postgres:17-alpine
```

Then create the tables, and again after pulling any schema change:

```bash
alembic upgrade head
```

Later, `podman start medtriage-db` brings it back. To remove the database and all its data:

```bash
podman rm -f medtriage-db && podman volume rm medtriage-pgdata && podman rmi postgres:17-alpine
```

### Doctor accounts

Doctors are invited from the command line; there is no public doctor sign-up. The invitation email is sent by the running API.

```bash
python -m app.admin invite-doctor doctor@example.com --name "Dr Rao"
```

### Running the development server

Start the API server with Uvicorn:

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **Local**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)

### Tests

The tests run against a separate `medtriage_test` database in the same container, with the LLM pipeline stubbed, so they cost nothing and need no network:

```bash
podman exec medtriage-db psql -U medtriage -d medtriage -c "create database medtriage_test"   # once
pip install -r requirements-dev.txt
pytest
```


## Contributing

We welcome contributions from the community! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is an active research project under development. Features and documentation will be updated regularly.