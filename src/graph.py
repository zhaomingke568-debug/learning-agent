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
    reduce_data_node,
)
def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("paper_agent", paper_agent)
    workflow.add_node("code_agent", code_agent)
    workflow.add_node("video_agent", video_agent)
    workflow.add_node("review", review_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("reduce_data_node", reduce_data_node)
    workflow.add_node("pdf_rag_node", pdf_rag_node)
    
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
            "review",
            lambda x: review_node(x),
            {
                "synthesis": "synthesis",
                "plan": "plan"
            }
        )
    # 新增一条条件路由
    workflow.add_conditional_edges(
        "synthesis", 
        check_synthesis_status,
    {
        "reduce_data_node": "reduce_data_node", # 把数据砍掉一半
        END: END                           # 成功则结束
    }
)

# 当 reduce_data 节点把数据删减后，再指回 synthesis 重试
    workflow.add_edge("reduce_data_node", "synthesis")

    workflow.add_edge("synthesis", END)
    # conn = sqlite3.connect("checkpointer.sqlite", check_same_thread=False)
    # memory = SqliteSaver(conn)
    return workflow.compile() # checkpointer=memory)

def check_synthesis_status(state: AgentState) -> str:
    if state.get("synthesis_error"):
        # 如果发现错误，路由到一个专门用来“精简数据”的节点，然后再去生成
        return "reduce_data_node" 

    return END
