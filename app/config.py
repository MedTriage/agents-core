# Configurations for handling environment variables and other settings for the application
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gemini-2.5-flash" 
OPENAI_MODEL_NAME = "gpt-4o-mini"

MAX_RETRIES = 2
REQUEST_TIMEOUT = 15