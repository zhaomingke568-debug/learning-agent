from mailbox import Message
import os
from dotenv import load_dotenv
from src.graph import build_graph

# Load environment variables
load_dotenv(dotenv_path='.env')

def main():
    # Verify API Keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        return

    # User Input
    topic = input("请输入学习主题 (Enter learning topic): ")
    depth = input("请输入深度等级 (入门 / 进阶 / 专家): ") or "进阶"

    # Initialize State
    initial_state = {
        "topic": topic,
        "depth_level": depth,

        "search_queries": {},

        "paper_results": [],
        "github_results": [],
        "youtube_results": [],

        "final_report": None,

        "errors": [],
        "synthesis_error": None,
        "reduce_data_error": None,
        "next_step": None,

        "loop_count": 0,#搜索循环次数
        "feedback": None,
        "summary": None,



    }

    # Build and Run Graph
    app = build_graph()
    
    print(f"\n--- Starting Learning Agent for: {topic} ---\n")
    config = {"configurable": {"thread_id": "user_123"}}
    #messages={"messages": [Message(content=user_query)]}
    # Run the workflow
    final_state = app.invoke(initial_state, config=config, messages=Message)

    # Output Report
    if final_state.get("final_report"):
        print("\n" + "="*50)
        print("FINAL LEARNING GUIDE")
        print("="*50 + "\n")
        print(final_state["final_report"])
        
        # Optionally save to file
        filename = f"learning_guide_{topic.replace(' ', '_')}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_state["final_report"])
        print(f"\nReport saved to {filename}")
    else:
        print("\nFailed to generate report.")
        if final_state.get("synthesis_error"):
            print("Synthesis Error:", final_state["synthesis_error"])
        if final_state.get("reduce_data_error"):
            print("Reduce Data Error:", final_state["reduce_data_error"])
        if final_state.get("errors"):
            print("Errors:", final_state["errors"])

if __name__ == "__main__":
    main()
