import llmopts
import fastapi
from pydantic import BaseModel, Field

class Api(fastapi.FastAPI):
    """
    API类，作为uvicorn入口类，继承FastAPI并在初始化时加载LLM模型
    self.llm: llmopts.LLM 提供增强生成功能
    """

    def __init__(self):
        super().__init__()
        self.llm = llmopts.LLM()

api = Api()



class Query(BaseModel):
    """
    定义查询请求的结构
    question: 用户提出的问题
    history: 历史对话记录，包含之前的问答内容
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )
    history: str = Field(
        ...,
        min_length=0,
        max_length=8000,
    )



@api.post("/query")
def query(
    body: Query,
):
    """
    查询函数，执行一次完整的查询流程，接受问题，返回查询结果

    Args:
        body (Query): 包含用户问题和历史对话记录的请求体
    
    Returns:
        dict: 包含查询结果的响应字典，格式为 {"response": response}
    """
    response = api.llm.query(
        question=body.question,
        history=body.history,
        n_results=3,
    )
    return {"response": response}