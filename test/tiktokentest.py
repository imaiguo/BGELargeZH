import tiktoken

string = "你好"

encoding = tiktoken.get_encoding('cl100k_base')
num_tokens = len(encoding.encode(string))