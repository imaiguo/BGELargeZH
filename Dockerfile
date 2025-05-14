FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/openai

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p /opt/Data/ModelWeight/embedding/BAAI/bge-large-zh-v1.5

COPY src /opt/openai
COPY requirements.txt /opt/openai

RUN python -m pip install -r requirements.txt

RUN apt update && apt install -y curl

EXPOSE 8000 8000