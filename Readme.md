# BGE-LARGE-ZH 模型

智源研究院研发的中文版文本表示模型，可将任意文本映射为低维稠密向量

## 设置python虚拟环境
```bash
> sudo apt install python3-venv python3-pip
> mkdir /opt/Data/PythonVenv
> cd /opt/Data/PythonVenv
> python3 -m venv BGELargeZH
> source /opt/Data/PythonVenv/BGELargeZH/bin/activate
>
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 服务运行

```bash
> python src/ApiServer.py
>
```

## 4 docker镜像制作

```bash
> sudo docker build -t imaiguo/bgelargezh:1.5 .
```

## 5 修改docker镜像源

进入daemon.json（如果没有手动创建）

```bash
> sudo vim /etc/docker/daemon.json
> #添加
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.1panel.live"
    ]
}
>
> #查看docker 镜像信息
> docker info
```

