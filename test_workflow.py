import os
os.environ["OPENAI_API_KEY"] = "sk-dummy"
os.environ["GITHUB_TOKEN"] = "dummy"
os.environ["YOUTUBE_API_KEY"] = "dummy"

from src.state import AgentState
from src.graph import build_graph
from unittest.mock import MagicMock
import src.nodes as nodes

# Mock the tools
nodes.search_arxiv = MagicMock(return_value=[{"title": "Test Paper", "summary": "...", "url": "..."}])
nodes.search_github = MagicMock(return_value=[{"name": "test-repo", "description": "...", "url": "..."}])
nodes.search_youtube = MagicMock(return_value=[{"title": "Test Video", "url": "...", "description": "..."}])

# Mock the LLM calls in nodes
nodes.llm.invoke = MagicMock()
# First call (plan_node)
nodes.llm.invoke.side_effect = [
    MagicMock(content='{"papers": ["query1"], "github": ["query1"], "youtube": ["query1"]}'), # plan_node
    MagicMock(content='# Final Report\nThis is a test report.') # synthesis_node
]

def test_workflow():
    initial_state = {
        "topic": "Test Topic",
        "depth_level": "入门",
        "search_queries": {},
        "paper_results": [],
        "github_results": [],
        "youtube_results": [],
        "final_report": None,
        "errors": []
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    print("\n--- Final State ---")
    print(f"Topic: {final_state['topic']}")
    print(f"Papers found: {len(final_state['paper_results'])}")
    print(f"Repos found: {len(final_state['github_results'])}")
    print(f"Videos found: {len(final_state['youtube_results'])}")
    print(f"Report: {final_state['final_report'][:50]}...")

if __name__ == "__main__":
    test_workflow()
