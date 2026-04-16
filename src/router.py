import os
import json
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# Ensure environment variables are loaded before initializing LLM
load_dotenv(dotenv_path='.env')

from src.state import AgentState
from langchain_anthropic import ChatAnthropic

llm_router = ChatAnthropic(
    model="MiniMax-M2.5-highspeed",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    max_retries=5
)
def check_synthesis_status(state: AgentState) -> str:
    if state.get("synthesis_error"):
        # 如果发现错误，路由到一个专门用来“精简数据”的节点，然后再去生成
        return "reduce_data_node" 

    return END


def should_we_synthesize(state):
    if state.get("next_step", "") == "synthesis" :
        return "synthesis"
    else:
        return "plan"

def intelligent_router(state: AgentState) -> str:
    """
    智能路由：根据用户的反馈，判断是走大循环（重规划）还是小循环（纯修改）
    """
    feedback = state.get("user_feedback", "").strip()
    
    # 1. 极速通道：如果没有反馈或确认满意，直接结束，不消耗 LLM token
    if not feedback or feedback.lower() in ["ok", "无", "没有", "满意", "不需要"]:
        print("\n--- 🎉 用户无修改要求，学习路径生成完毕！ ---")
        return "end"

    print(f"\n--- 🧠 AI 正在分析你的修改诉求... ---")
    
    # 2. 智能分类：让 LLM 判断反馈的深度
    prompt = f"""
    你是一个智能工作流路由助手。请分析用户对生成的文章提出的修改要求。
    
    用户修改要求："{feedback}"
    
    请严格根据以下标准进行分类：
    1. 选【REVISE】：如果用户的要求仅仅是改短一点、换个语气、增加表格、或者基于文章现有的内容做删减改写（即：不需要额外去搜索引擎或知识库查资料）。
    2. 选【REPLAN】：如果用户的要求提出了文中没有提到的新概念、要求深挖某个具体技术、或者明显需要补充新知识（即：必须重新去查资料）。
    
    请只输出 "REVISE" 或 "REPLAN"，不要输出任何其他标点或解释。
    """
    
    chain = llm_router | StrOutputParser()
    decision = chain.invoke(prompt).strip().upper()
    
    # 3. 结果分发
    if "REPLAN" in decision:
        print("👉 诊断：【需要补充新知识】，启动大循环：返回计划节点 (Plan)...")
        return "plan"
    else:
        print("👉 诊断：【仅需修改文本格式】，启动小循环：进入修改节点 (Revise)...")
        return "revise"
