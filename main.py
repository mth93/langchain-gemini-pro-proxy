import logging
import uvicorn  # Import uvicorn for server launch
import uuid
from fastapi import FastAPI, Request, Depends, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from starlette.config import Config
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
import time
# Load configuration from environment variables or .env file
config = Config(".env")

# Set up logging
logging.basicConfig(filename='api_logs.log',level=logging.DEBUG)  # Adjust logging level as needed

app = FastAPI()

# Function to extract API key from request headers
def get_api_key(request: Request):
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        logging.warning("Missing API key in request headers")
        raise HTTPException(status_code=401, detail="Missing API key in request headers")
    return api_key

# Models for request data
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 530
    temperature: float = 0.7

class ChatCompletionRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 530

class EmbeddingRequest(BaseModel):
    input: str


def format_response_completion(input_response):
    formatted_response = {
        "id": str(uuid.uuid4()),  # Generate unique ID
        "object": "text_completion",  # Match object type
        "created": int(time.time()),  # Get current Unix timestamp
        "model": "gpt-3.5-turbo-instruct",  # Hardcode model name (adjust if needed)
        "system_fingerprint": "fp_44709d6fcb",  # Placeholder, update if available
        "choices": [
            {
                "text": input_response.content,  # Extract text from input_response
                "index": 0,
                "logprobs": None,  # Not provided in input_response
                "finish_reason": "length"  # Hardcode finish reason (adjust if needed)
            }
        ],
        "usage":{}
    }
    return formatted_response



def format_response_chat_completion(response):
    formatted_response = {
        "id": str(uuid.uuid4()),  # Generate unique ID
        "object": "chat.completion",
        "created": int(time.time()),  # Get current Unix timestamp
        "model": response.llm_output.get("model", "gemini-pro"),  # Use model name from llm_output if available
        "usage": response.llm_output.get("usage", {}),  # Extract usage data if available
        "choices": [
            {
                "message": {
                    "role": "assistant",  # Consistent role for Gemini responses
                    "content": chat_generation.text  # Access text directly
                },
                "logprobs": None,
                "finish_reason": chat_generation.generation_info.get("finish_reason", "stop"),
                "index": 0,
            }
            for chat_generation in response.generations[0]  # Correctly access generations
        ],
    }
    return formatted_response

def format_embeddings(embeddings_response_example):
  """Formats embeddings to match the OpenAI embeddings schema.

  Args:
    embeddings_response_example: A list of embedding values.

  Returns:
    A dictionary containing the formatted embeddings.
  """
  return {
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding":embeddings_response_example,
      "index": 0  
    }
  ],
  "model": "text-embedding-ada-002",  
  "usage": {
    "prompt_tokens": int(len(embeddings_response_example)/3), 
    "total_tokens": int(len(embeddings_response_example)/3) 
  }
}



# Endpoints with LangChain integration
@app.post("/v1/completions")
async def generate_text(completion_request: CompletionRequest, api_key: str = Depends(get_api_key)):
    logging.info(f"Received text completion request: {completion_request}")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            max_output_tokens=completion_request.max_tokens,
            convert_system_message_to_human=True,
            cache=False
        )
        response = llm.invoke(
            input=completion_request.prompt,
            max_tokens=completion_request.max_tokens,
            temperature=completion_request.temperature,
        )
        formatted_response = format_response_completion(response)
        logging.info(f"Generated text completion response: {response}")
        return formatted_response

    except Exception as e:
        logging.error(f"Error generating text completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating text completion: {str(e)}")


@app.post("/v1/chat/completions")
async def chat_completion(chat_completion_request: ChatCompletionRequest, api_key: str = Depends(get_api_key)):
    messages = []
    for msg in chat_completion_request.messages:
        if msg['role'] == 'system':
            message = SystemMessage(content=msg['content'])  # Create instance
        else:
            message = HumanMessage(content=msg['content'])  # Create instance
        messages.append(message)  # Append the instance to the list

    logging.info(f"Received chat completion request: {messages}")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            max_output_tokens=chat_completion_request.max_tokens,
            convert_system_message_to_human=True,
            cache=False
        )
        response = llm.generate(messages=[messages])
        logging.info(f"Generated chat completion response: {response}")

        # Format response as desired
        formatted_response = {
            "id": str(uuid.uuid4()),  # Generate unique ID
            "object": "chat.completion",
            "created": int(time.time()),  # Get current Unix timestamp
            "model": llm.model,
            "usage": response.llm_output.get("usage", {}),  # Extract usage data if available
            "choices": [
                {"message": choice[0].text, "finish_reason": "stop", "index": 0}  # Convert Choice object to dictionary
                for choice in response.generations
            ],
        }
        return formatted_response

    except Exception as e:
        logging.error(f"Error generating chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chat completion: {str(e)}")


@app.post("/v1/embeddings")
async def create_embeddings(embedding_request: EmbeddingRequest, api_key: str = Depends(get_api_key)):
    """Creates embeddings for the given input text.

    Args:
        embedding_request: The request containing the input text.
        api_key: The Google API key.

    Returns:
        A formatted dictionary containing the embeddings.
    """

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        embeddings = embedding_model.embed_query(embedding_request.input)
        formatted_embeddings = format_embeddings(embeddings)
        return formatted_embeddings

    except Exception as e:
        logging.error(f"Error creating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

# Launch the API using uvicorn, setting the port here
if __name__ == "__main__":
    port = config("GEMINI_API_PORT", cast=int, default=8000)  # Set the port only here
    uvicorn.run(app, host="localhost", port=port)
