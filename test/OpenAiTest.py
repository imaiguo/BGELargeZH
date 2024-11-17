
import Config
from openai import OpenAI

BaseUrl=f"http://{Config.EbeddingServerIP}:{Config.EbeddingServerPort}/v1"

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
