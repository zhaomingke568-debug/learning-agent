import os
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.tools.rag.rag_pdf import pdf_rag_node
from src.reduce_data import reduce_data_node
# import sqlite3
# from langgraph.checkpoint import SqliteSaver
from src.nodes import (
    plan_node,
    paper_agent,
    code_agent,
    video_agent,
    review_node,
    synthesis_node,
    summarize_node,
    adjust_node,
    get_feedback_node,
)
from src.router import (
    should_we_synthesize,
    check_synthesis_status,
    feedback_router,
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
    workflow.add_node("pdf_rag_node", pdf_rag_node)
    workflow.add_node("reduce_data_node", reduce_data_node)
    workflow.add_node("adjust_node", adjust_node)
    workflow.add_node("get_feedback", get_feedback_node)
    workflow.add_node("summarize", summarize_node)

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
        should_we_synthesize,
        {
            "synthesis": "synthesis",
            "plan": "plan"
        }
    )

    # After synthesis, check for errors
    workflow.add_conditional_edges(
        "synthesis",
        check_synthesis_status,
        {
            "reduce_data_node": "reduce_data_node",
            END: END,
        }
    )

    # reduce_data_node -> synthesis to retry
    workflow.add_edge("reduce_data_node", "synthesis")

    # Normal flow: synthesis -> get_feedback
    workflow.add_edge("synthesis", "get_feedback")

    # get_feedback -> feedback_router handles both summarization check and final routing
    workflow.add_conditional_edges(
        "get_feedback",
        feedback_router,
        {
            "summarize": "summarize",
            "end": END,
            "plan": "plan",
            "revise": "adjust_node",
        }
    )

    # After summarization, re-check feedback routing
    workflow.add_edge("summarize", "get_feedback")

    return workflow.compile() 