import tiktoken
import loguru

if __name__ == "__main__":
    string = "你好"

    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    loguru.logger.debug(f"num_tokens:[{num_tokens}]")