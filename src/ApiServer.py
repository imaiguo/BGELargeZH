"""
This script implements an API for the ChatGLM3-6B model,
formatted similarly to OpenAI's API (https://platform.openai.com/docs/api-reference/chat).
It's designed to be run as a web server using FastAPI and uvicorn,
making the ChatGLM3-6B model accessible through OpenAI Client.

Key Components and Features:
- Model and Tokenizer Setup: Configures the model and tokenizer paths and loads them.
- FastAPI Configuration: Sets up a FastAPI application with CORS middleware for handling cross-origin requests.
- API Endpoints:
  - "/v1/models": Lists the available models, specifically ChatGLM3-6B.
  - "/v1/chat/completions": Processes chat completion requests with options for streaming and regular responses.
  - "/v1/embeddings": Processes Embedding request of a list of text inputs.
- Token Limit Caution: In the OpenAI API, 'max_tokens' is equivalent to HuggingFace's 'max_new_tokens', not 'max_length'.
For instance, setting 'max_tokens' to 8192 for a 6b model would result in an error due to the model's inability to output
that many tokens after accounting for the history and prompt tokens.
- Stream Handling and Custom Functions: Manages streaming responses and custom function calls within chat responses.
- Pydantic Models: Defines structured models for requests and responses, enhancing API documentation and type safety.
- Main Execution: Initializes the model and tokenizer, and starts the FastAPI app on the designated host and port.

Note:
    This script doesn't include the setup for special tokens or multi-GPU support by default.
    Users need to configure their special tokens and can enable multi-GPU support as per the provided instructions.
    Embedding Models only support in One GPU.

    Running this script requires 14-15GB of GPU memory. 2 GB for the embedding model and 12-13 GB for the FP16 ChatGLM3 LLM.


"""

import time
import tiktoken
import torch
import uvicorn

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Optional, Union
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

EMBEDDING_PATH = "/opt/Data/ModelWeight/embedding/BAAI/bge-large-zh-v1.5"

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

## for Embedding
class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if isinstance(request.input, str):
        embeddings = [embedding_model.encode(request.input)]
    else:
        embeddings = [embedding_model.encode(text) for text in request.input]
    embeddings = [embedding.tolist() for embedding in embeddings]

    def num_tokens_from_string(string: str) -> int:
        """
        Returns the number of tokens in a text string.
        use cl100k_base tokenizer
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    response = {
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": CompletionUsage(
            prompt_tokens=sum(len(text.split()) for text in request.input),
            completion_tokens=0,
            total_tokens=sum(num_tokens_from_string(text) for text in request.input),
        )
    }
    return response

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(
        id="bge-large-zh-v1.5"
    )
    return ModelList(
        data=[model_card]
    )

if __name__ == "__main__":
    # load Embedding
    embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cpu")
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
