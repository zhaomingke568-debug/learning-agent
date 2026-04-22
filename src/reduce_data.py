import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from src.state import AgentState

small_llm = ChatAnthropic(
    model="MiniMax-M2.5-highspeed",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL")
)
def reduce_data_node(state: AgentState) -> dict:
    """使用小模型对超长上下文进行高密度压缩"""
    print("--- 🗜️ TRIGGERING SMALL MODEL FOR CONTEXT COMPRESSION ---")
    
    # 1. 收集所有原始长数据
    papers = state.get("paper_results", [])
    github = state.get("github_results", [])
    youtube = state.get("youtube_results", [])
    
    raw_data_str = f"""
    [论文]: {json.dumps(papers, ensure_ascii=False)}
    [代码]: {json.dumps(github, ensure_ascii=False)}
    [视频]: {json.dumps(youtube, ensure_ascii=False)}
    """
    
    # 2. 针对小模型设计的“高密度压缩 Prompt”
    # 秘诀：要求它提取核心实体、结论和链接，坚决去除废话。
    compress_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个无情的信息压缩机器。你的任务是将用户提供的超长 JSON 数据压缩成高密度的知识摘要。
【压缩规则】
1. 坚决保留所有 URL 链接、作者、项目名称。
2. 将长段落摘要(Abstract/Readme)缩写为不超过20个字的核心技术点。
3. 去除所有寒暄、格式化字符、冗余的形容词。
4. 必须输出极简的 Markdown 列表。"""),
        ("human", "原始数据如下，请立即开始压缩：\n{raw_data}")
    ])
    
    # 3. 组装链并调用小模型
    compress_chain = compress_prompt | small_llm | StrOutputParser()
    
    try:
        compressed_text = compress_chain.invoke({"raw_data": raw_data_str})
        print("--- ✅ COMPRESSION SUCCESSFUL ---")
        
        # 4. 返回压缩后的数据，并清空报错状态，以便主节点重试
        return {
            "compressed_context": compressed_text,
            "synthesis_error": "" 
        }
    except Exception as e:
        print(f"--- ❌ COMPRESSION FAILED: {e} ---")
        # 如果小模型也崩了（说明数据大得离谱），这里可以再嵌套一层暴力字符串截断
        return {"compressed_context": raw_data_str[:3000]}
