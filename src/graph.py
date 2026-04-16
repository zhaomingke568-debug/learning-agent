import src.rag_pdf
from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.rag_pdf import pdf_rag_node
# import sqlite3
# from langgraph.checkpoint import SqliteSaver
from src.nodes import (
    plan_node,
    paper_agent,
    code_agent,
    video_agent,
    review_node,
    synthesis_node,
    #reduce_data_node,
    adjust_node,
    get_feedback_node,
)
from src.router import(
    should_we_synthesize,
    check_synthesis_status,
    intelligent_router)
def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("paper_agent", paper_agent)
    workflow.add_node("code_agent", code_agent)
    workflow.add_node("video_agent", video_agent)
    workflow.add_node("review", review_node)
    workflow.add_node("synthesis", synthesis_node)
    #workflow.add_node("reduce_data_node", reduce_data_node)
    workflow.add_node("pdf_rag_node", pdf_rag_node)
    workflow.add_node("adjust_node", adjust_node)
    workflow.add_node("get_feedback", get_feedback_node)
    
    # Set Edges
    workflow.add_edge(START, "plan")
    
    # Parallel dispatch from plan
    workflow.add_edge("plan", "paper_agent")
    workflow.add_edge("plan", "code_agent")
    workflow.add_edge("plan", "video_agent")
    workflow.add_edge("paper_agent", "pdf_rag_node")
    

    # Fan-in to review
    workflow.add_edge("paper_agent", "review")
    workflow.add_edge("code_agent", "review")
    workflow.add_edge("video_agent", "review")

        # Conditional routing from review
    workflow.add_conditional_edges(
    "review",                       # 起点
    should_we_synthesize,           # 判断逻辑 (不用写 lambda 也行，直接传函数名)
    {
        "synthesis": "synthesis",   # 如果返回 "synthesis"，去总结
        "plan": "plan"              # 如果返回 "plan"，回退去重新计划
    }
)
#     # 新增一条条件路由
#     workflow.add_conditional_edges(
#         "synthesis", 
#         check_synthesis_status,
#     {
#         "reduce_data_node": "reduce_data_node", # 把数据砍掉一半
#         END: END                           # 成功则结束
#     }
# )

# 当 reduce_data 节点把数据删减后，再指回 synthesis 重试
 #   workflow.add_edge("reduce_data_node", "synthesis")

    workflow.add_edge("synthesis", "get_feedback")
    
    # conn = sqlite3.connect("checkpointer.sqlite", check_same_thread=False)
    # memory = SqliteSaver(conn)
   

    workflow.add_conditional_edges(
    "get_feedback",        # 起点：用户输入完反馈后
    intelligent_router,    # 路由函数：调用我们刚写的那个智能判断器
    {
        "end": END,        # 如果返回 "end"，流程彻底结束
        "plan": "plan",    # 如果返回 "plan"，回退到最开始的 plan_node，重新检索
        "revise": "adjust_node" # 如果返回 "revise"，走向单纯改写文本的 revise_node
    }
)
    return workflow.compile() 