# 基础镜像（稳定Python3.10企业版）
FROM python:3.11-slim
# 设置工作目录
WORKDIR /app
ENV PYTHONPATH=/app
# 安装编译依赖（faiss、torch等库需要）
RUN apt update && apt install -y gcc g++ build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 拷贝全部项目代码
COPY . .
# 开放端口
EXPOSE 8000
# 项目启动命令
#CMD ["python", "src/api/agent_api.py"]
CMD ["python", "-m", "uvicorn", "src.api.agent_api:app", "--host", "0.0.0.0", "--port", "8000"]