# Agents Core

![](https://raw.githubusercontent.com/MedTriage/agents-core/main/assets/logo.png)

> Agentic AI for clinical triage, drug discovery, and remote digital healthcare, with a graduated autonomy architecture

The core repository housing the six different agents which constitute the agentic workflow.


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangGraph](https://img.shields.io/badge/langgraph-%231C3C3C.svg?style=for-the-badge&logo=langgraph&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

## Architecture

### Agentic Workflow

The system is built on a multi-agent architecture, built using LangGraph :

- **Intent Router** - Infers the intent of the query and accordingly routes to the desired agent
- **Vision Agent** - Analyzes medical pictures and sends the findngs to the RAG agent
- **RAG Agent** - Specialised RAG agent which retrieves relevant documents and performs a preliminary analysis
- **Critic Agent** - Critiques the RAG's findings, in order to detect hallucinations and ground evidences and re-performs retrieval if necessary
- **Guardian Agent** - After the critic's analysis, assigns risk scores and accordingly takes actions

### Risk Levels

- **Level-I** - Simple queries and chitchat, directly answered to patient
- **Level-II** - Most clinical queries, where the critic's analysis is sent to a doctor who verifies and approves it and sends it back to the patient
- **Level-III** - Potential Life-threatening scenarios, the agents lock themselves and communicate with local authorities in order to provide assistance to the patient

## Getting Started

### Prerequisites

- Python 3.9 or higher (ideally 3.11)
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

### Configuration

1. Set up your environment variables (create a `.env` file in the project root):
```bash
GEMINI_API_KEY=your_api_key_here
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