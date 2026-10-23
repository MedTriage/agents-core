# Main application entry point
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, conversations, doctor, me, triage
from app.config import EMAIL_WORKER_ENABLED
from app.graph.builder import build_graph
from app.services.mailer import start_worker


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Emails are queued with the events that cause them and sent from here, so a slow or
    # unconfigured SMTP server never holds up a request.
    stop = start_worker() if EMAIL_WORKER_ENABLED else None
    yield
    if stop:
        stop.set()


app = FastAPI(lifespan=lifespan)
app.include_router(auth.router)
app.include_router(me.router)
app.include_router(conversations.router)
app.include_router(triage.router)
app.include_router(doctor.router)
app.include_router(doctor.public)

# Held on the app so tests can swap in a fake: the graph is the one slow, paid,
# external dependency, and everything around it is what the tests protect.
app.state.graph = build_graph()


# Home route
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}
