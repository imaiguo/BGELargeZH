
import openai
import time
import loguru

import Config

BaseUrl=f"http://{Config.EbeddingServerIP}:{Config.EbeddingServerPort}/v1"

def GetEmbedding(msglist:list):
    openai_client = openai.OpenAI(base_url=BaseUrl, api_key = "not need key")
    response = openai_client.embeddings.create(
        model="bge-large-zh-1.5",
        input=msglist,
    )
    return response.data

if __name__ == "__main__":
    data = ["你好"]
    for i in range(10):
        start = time.time()
        result = GetEmbedding(data)
        end = time.time()
        loguru.logger.debug(f"Em length:[{len(result[0].embedding)}] elapsed time:[{end-start}]s")
