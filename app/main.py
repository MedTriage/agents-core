# Main application entry point 
from fastapi import FastAPI
from pydantic import BaseModel
from app.graph.builder import build_graph

app = FastAPI()

graph = build_graph()

class InputRequest(BaseModel):
    text: str
    chat_history: list[dict] = []

# Home route
@app.get("/")  
def read_root():
    return {"message": "Welcome to the API!"}

# Input route
@app.post("/process")
def process_input(request: InputRequest):
    result = graph.invoke({
        "user_input": request.text,
        "chat_history": request.chat_history
    })
    return result