# 宽带故障智能客服后端服务
## 项目介绍
基于 FastAPI 构建的宽带故障智能对话后端服务，提供会话对话、中断续聊能力，接收前端传入手机号、会话ID、用户问题，处理宽带故障问答逻辑，配套前端单页面对话系统。

## 技术栈
- Python 3.11
- FastAPI 高性能Web框架
- Pydantic 数据模型校验
- langgraph workflow 工作流编排
- Milvus 向量数据库
- docker 容器化部署
- LangSmith 链路追踪

## 接口说明
### 统一入参模型 AgentRequest
所有对话接口共用一套请求体，前端必须携带4个参数：
```python
class AgentRequest(BaseModel):
    user_id: str        # 用户全局唯一标识
    session_id: str     # 单次对话会话唯一ID（前端uuid生成）
    query: str          # 用户输入的宽带故障问题
    phone: str          # 当前会话绑定的11位手机号
1. 对话主入口接口
POST /agent/invoke
功能：新建会话 / 正常发起对话，初次对话会返回中断开场白
装饰器：链路追踪 @traceable(run_type="chain", name="对话入口")
返回模型：AgentResponse
特殊逻辑：返回 status="interrupted" 时代表会话中断，前端下次发送自动调用续聊接口
2. 中断续聊接口
POST /agent/resume
功能：上一轮对话返回中断状态后，用户再次提问时调用，接续中断会话继续问答
入参与 /agent/invoke 完全一致，复用 AgentRequest
返回数据结构示例（中断状态）
json
{
  "session_id": "52a6e44e-9f07-44fe-9d4c-408801b5c8e4",
  "status": "interrupted",
  "timestamp": 1784517859.3994102,
  "message": null,
  "result": null,
  "interrupt_data": {
    "interrupt_type": "user_input",
    "content": "您好，请问您的宽带遇到什么问题了吗？",
    "payload": {}
  }
}
interrupt_data.content：机器人开场白 / 中断提示文本，前端读取展示
status="interrupted"：标记会话中断，前端切换调用 /agent/resume

环境部署步骤
1. 安装依赖
创建虚拟环境（推荐）
bash
# 创建虚拟环境
python -m venv venv
# Windows激活
venv\Scripts\activate
# Mac/Linux激活
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
2. requirements.txt 参考内容
3. 启动服务
bash
# 开发热重载模式
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
服务地址：http://127.0.0.1:8012
接口文档地址：
Swagger：http://127.0.0.1:8012/docs
ReDoc：http://127.0.0.1:8012/redoc
