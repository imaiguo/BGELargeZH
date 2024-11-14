
from openai import OpenAI

OPENAI_SERVER = "127.0.0.1"
OPENAI_PORT = 10000
BaseUrl=f"http://{OPENAI_SERVER}:{OPENAI_PORT}/v1"

def GetEmbedding(msglist:list):
    print(BaseUrl)
    openai_client = OpenAI(base_url=BaseUrl, api_key = "not need key")
    response = openai_client.embeddings.create(
        model="bge-large-zh-1.5",
        input=msglist,
    )
    return response.data

if __name__ == "__main__":
    data = ["你好"]
    result = GetEmbedding(data)
    print(len(result[0].embedding))
