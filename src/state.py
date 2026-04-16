from typing import List, TypedDict, Optional, Annotated
import operator

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
class AgentState(TypedDict):
    topic: str
    depth_level: str  # 入门 / 进阶 / 专家
    search_queries: dict  # {"papers": [...], "github": [...], "youtube": [...]}
    paper_results: Annotated[List[dict], operator.add]
    github_results: Annotated[List[dict], operator.add]
    youtube_results: Annotated[List[dict], operator.add]
    final_report: Optional[str]
    errors: Annotated[List[str], operator.add]
    loop_count: int = 0
    synthesis_error: str#记录综合错误信息

    messages: Annotated[list[AnyMessage], add_messages]
    # 假设之前的节点下载了 PDF，并将本地路径存在这里
    downloaded_pdfs: List[str] 
    
    # RAG 节点提取出的精准片段，使用 operator.add 累加
    pdf_context: Annotated[List[dict], operator.add]
    
    next_step: str
    user_feedback: str     # 👈 新增：用来存储用户的修改要求
    summary: str#汇总会话内容
    
