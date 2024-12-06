
import time
import tiktoken
import torch
import typing
import uvicorn
import fastapi
import pydantic
import contextlib
import sentence_transformers

from fastapi.middleware.cors import CORSMiddleware

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

EMBEDDING_PATH = "/opt/Data/ModelWeight/embedding/BAAI/bge-large-zh-v1.5"

@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = fastapi.FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(pydantic.BaseModel):
    id: str
    object: str = "model"
    created: int = pydantic.Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: typing.Optional[str] = None
    parent: typing.Optional[str] = None
    permission: typing.Optional[list] = None


class ModelList(pydantic.BaseModel):
    object: str = "list"
    data: typing.List[ModelCard] = []

## for Embedding
class EmbeddingRequest(pydantic.BaseModel):
    input: typing.Union[typing.List[str], str]
    model: str


class CompletionUsage(pydantic.BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(pydantic.BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage

@app.get("/health")
async def health() -> fastapi.Response:
    """Health check."""
    return fastapi.Response(status_code=200)

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
    # load Embedding cpu cuda
    embedding_model = sentence_transformers.SentenceTransformer(EMBEDDING_PATH, device="cpu")
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
