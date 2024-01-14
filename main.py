from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from starlette.config import Config

# Load configuration from environment variables or .env file
config = Config(".env")

app = FastAPI()

# Function to extract API key from request headers
def get_api_key(request: Request):
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        raise ValueError("Missing API key in request headers")
    return api_key

# Models for request data
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 15
    temperature: float = 0.7

class ChatCompletionRequest(BaseModel):
    messages: list[str]
    max_tokens: int = 15

class EmbeddingRequest(BaseModel):
    input: str

# Endpoints with LangChain integration and port customization
@app.post("/v1/completions")
async def generate_text(completion_request: CompletionRequest, api_key: str = Depends(get_api_key)):
    port = config("PORT", cast=int, default=8000)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    response = llm.invoke(
        prompt=completion_request.prompt,
        max_tokens=completion_request.max_tokens,
        temperature=completion_request.temperature,
    )
    return {"choices": [{"text": choice.content} for choice in response.choices]}

@app.post("/v1/chat/completions")
async def chat_completion(chat_completion_request: ChatCompletionRequest, api_key: str = Depends(get_api_key)):
    port = config("PORT", cast=int, default=8000)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    response = llm.chat(
        messages=chat_completion_request.messages,
        max_tokens=chat_completion_request.max_tokens,
    )
    return {"choices": [{"text": choice.content} for choice in response.choices]}

# @app.post("/v1/embeddings")
# async def create_embeddings(embedding_request: EmbeddingRequest, api_key: str = Depends(get_api_key)):
#     port = config("PORT", cast=int, default=8000)
#     # Adapt this endpoint based on LangChain's capabilities or Gemini Pro's features
