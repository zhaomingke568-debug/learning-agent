import os
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded before initializing LLM
load_dotenv(dotenv_path='.env')

from src.state import AgentState
from src.tools.arxiv_tool import search_arxiv
from src.tools.github_tool import search_github
from src.tools.youtube_tool import search_youtube
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# Initialize LLM (Minimax via Anthropic-compatible API)
llm = ChatAnthropic(
    model="MiniMax-M2.7",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL")
    
)
class SearchQueries(BaseModel):
    topic_analysis: str = Field(description="简要分析用户主题的核心技术实体是什么，以及入门与进阶用户的不同需求。")
    papers: list[str] = Field(description="ArXiv 论文搜索关键词列表")
    github: list[str] = Field(description="GitHub 开源项目搜索关键词列表")
    youtube: list[str] = Field(description="YouTube 视频搜索关键词列表")

def plan_node(state: AgentState) -> dict:
    """Analyze the topic and generate search queries."""
    print(f"--- PLANNING for topic: {state['topic']} (Level: {state['depth_level']}) ---")
    history = state.get("messages", [])
    # 2. 初始化 JSON 解析器
    parser = JsonOutputParser(pydantic_object=SearchQueries)
    
    # 3. 修复 Prompt：使用 from_messages，并注入 parser 的格式说明
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个顶级的 AI 技术研究规划师与搜索引擎优化专家（SEO）。
你的任务是根据深度级别,搜索历史，将用户的学习主题，精准翻译为 ArXiv、GitHub 和 YouTube 这三个平台的底层搜索查询词。

【平台搜索法则】
1. 学术论文 (ArXiv): 必须是纯英文，使用学术界标准术语，切忌口语化。
2. 代码仓库 (GitHub): 侧重于寻找可运行的系统、库或框架。
3. 教学视频 (YouTube): 侧重于直观的解释、实战教程或架构解析。

{format_instructions}
【Few-Shot 案例】
主题：LangChain Agent入门
深度级别：Beginner
你的输出结构应类似：
- topic_analysis: "用户想了解LangChain框架中Agent的基础概念和搭建方法。属于初级应用层需求。"
- paper_query: "LLM Agent architecture AND tool use" (注：寻找底层的工具调用原理论文)
- github_query: "LangChain Agent tutorial examples language:python"
- youtube_query: "LangChain Agents explained for beginners step by step"

"""),
        ("human", "当前任务 —— 主题：{topic}，深度级别：{depth_level}。请开始分析并生成查询词。"),
        ("placeholder", "{chat_history}"),
    ])
    
    # 4. 组装 LCEL 链：Prompt -> LLM -> JSON解析器
    chain = prompt | llm | parser
    
    try:
        # invoke 时，除了传入 topic 和 depth_level，必须传入 format_instructions
        queries = chain.invoke({
            "topic": state["topic"],
            "depth_level": state["depth_level"],
            "format_instructions": parser.get_format_instructions(),
            "chat_history": history

        })
    except Exception as e:
        print(f"Warning: JSON parsing failed, using fallback queries. Error: {e}")
        # 5. 依然保留你的 Fallback 机制以防万一
        queries = {
            "topic_analysis": f"用户想了解{state['topic']}。属于{state['depth_level']}需求。",
            "papers": [state["topic"]],
            "github": [state["topic"]],
            "youtube": [state["topic"]],
            "rag_context": [state['topic']]
        }
    
    return {"search_queries": queries,
    "loop_count": state["loop_count"] + 1 }

def paper_agent(state: AgentState) -> dict:
    """Execute paper search."""
    print("--- SEARCHING PAPERS ---")
    queries = state.get("search_queries", {}).get("papers", [state["topic"]])
    results = []
    seen_titles = set()
    
    for q in queries:
        try:
            papers = search_arxiv(q, max_results=2)
            for paper in papers:
                if paper["title"] not in seen_titles:
                    seen_titles.add(paper["title"])
                    results.append(paper)
        except Exception as e:
            print(f"  ❌ Error searching ArXiv for '{q}': {e}")
            
    print(f"--- ✅ Found {len(results)} unique papers ---")
    return {"paper_results": results}

def code_agent(state: AgentState) -> dict:
    """Execute code search."""
    print("--- SEARCHING CODE ---")
    queries = state.get("search_queries", {}).get("github", [state["topic"]])
    results = []
    seen_repos = set()
    
    for q in queries:
        try:
            repos = search_github(q, max_results=2)
            for repo in repos:
                if repo["name"] not in seen_repos:
                    seen_repos.add(repo["name"])
                    results.append(repo)
        except Exception as e:
            print(f"  ❌ Error searching GitHub for '{q}': {e}")

    print(f"--- ✅ Found {len(results)} unique repos ---")
    return {"github_results": results}

def video_agent(state: AgentState) -> dict:
    """Execute video search."""
    print("--- SEARCHING VIDEOS ---")
    queries = state.get("search_queries", {}).get("youtube", [state["topic"]])
    results = []
    seen_videos = set()
    
    for q in queries:
        try:
            videos = search_youtube(q, max_results=2)
            for video in videos:
                if video["id"] not in seen_videos:
                    seen_videos.add(video["id"])
                    results.append(video)
        except Exception as e:
            print(f"  ❌ Error searching YouTube for '{q}': {e}")
            
    print(f"--- ✅ Found {len(results)} unique videos ---")
    return {"youtube_results": results}

def review_node(state: AgentState) -> str:
    """Evaluate if information is sufficient to proceed to synthesis."""
    print("--- 🧐 EVALUATING SEARCH RESULTS ---")
    
    papers = state.get("paper_results", [])
    github = state.get("github_results", [])
    youtube = state.get("youtube_results", [])
    loop_count = state.get("loop_count", 0)
    
    # 1. 检查是否三大源都拿到了数据
    if len(papers) > 0 and len(github) > 0 and len(youtube) > 0:
        print("--- ✅ INFO SUFFICIENT: Moving to Synthesis ---")
        return "synthesis"
    
    # 2. 防死循环：如果已经重试了 2 次（总共执行了 3 次），强制进入总结
    if loop_count >= 3:
        print("--- ⚠️ MAX RETRIES REACHED: Forcing Synthesis ---")
        return "synthesis"
        
    # 3. 触发重试逻辑：回到 Plan 节点重新生成查询词
    print("--- 🔄 INFO INSUFFICIENT: Retrying Plan Node ---")
    return "plan"

def synthesis_node(state: AgentState) -> dict:
    """Generate final Markdown report."""
    print("--- SYNTHESIZING REPORT ---")
    
    from langchain_core.prompts import ChatPromptTemplate

    synthesis_prompt = ChatPromptTemplate.from_template("""
你是一个资深的技术布道师和全栈工程师，负责创建一个丰富、现代化的单页面网页（SPA），用于指定主题的详细学习和研究。

【任务信息】
主题：{topic}
目标级别：{depth_level}

【原始数据上下文】
以下是未经处理的检索数据（JSON格式），请仔细阅读并提取有价值的信息：
[论文数据]：{papers}
[GitHub数据]：{github}
[YouTube数据]：{youtube}

【网页结构与内容要求】
请生成一个完整的 HTML5 文档。页面必须包含以下部分：
1. 页面头部：优雅的标题和 1-2 句话的核心概念总结。
2. 理论基础 (Theory)：挑选最相关的 3 篇论文。使用卡片式布局，包含论文标题（带超链接）、作者和 1-2 句话的核心创新点总结。
3. 实践实现 (Practice)：挑选 3 个优质 GitHub 项目。突出显示 Stars 数量，并用 1-2 句话解释其用途及仓库链接。
4. 视频教程 (Tutorials)：挑选 3 个 YouTube 视频。重点：请直接使用 <iframe> 标签将视频嵌入到网页中（利用 videoId），并在旁边配上 1-2 句话的精华总结。

【UI/UX 设计规范】
1. 必须使用内联 CSS (或引入 Tailwind/Bootstrap CDN) 来确保页面美观。
2. 风格要求：现代、极简、护眼模式（浅色背景，深灰字体）、使用卡片（Cards）展示各项资源、加入微小的悬浮阴影效果。
3. 字体建议：系统默认的无衬线字体 (system-ui, -apple-system, sans-serif)。

【输出严格约束】
- 仅输出纯 HTML 代码！
- 绝对不要在开头或结尾添加 ```html 或 ``` 这样的 Markdown 标记，直接以 <!DOCTYPE html> 开头。
""")
    
    
    chain = synthesis_prompt | llm | StrOutputParser()
   
    # 将字典列表转化为格式化的 JSON 字符串喂给 LLM
    html_content = chain.invoke({
        "topic": state["topic"],
        "depth_level": state["depth_level"],
        "papers": json.dumps(state.get("paper_results", []), ensure_ascii=False),
        "github": json.dumps(state.get("github_results", []), ensure_ascii=False),
        "youtube": json.dumps(state.get("youtube_results", []), ensure_ascii=False)
    })
    
    # 防御性处理：万一大模型还是不听话加了 ```html，手动剥离它
    if html_content.startswith("```html"):
        html_content = html_content[7:]
    if html_content.endswith("```"):
        html_content = html_content[:-3]
    try:
        # 正常生成逻辑...
        return {"final_report": html_content}
    except Exception as e:          
        # 发生错误时，将错误信息写入 state
        return {"synthesis_error": f"synthesis_error: {e}"}

def reduce_data_node(state: AgentState) -> dict:
    """
    Reduces the amount of data in the state if synthesis failed,
    to attempt regeneration with less information.
    """
    print("--- REDUCING DATA FOR RETRY ---")
    
    # Clear some results to force a retry with less data
    # For example, keep only the top 1 result from each category
    reduced_state = state.copy()
    reduced_state["paper_results"] = reduced_state.get("paper_results", [])[:1]
    reduced_state["github_results"] = reduced_state.get("github_results", [])[:1]
    reduced_state["youtube_results"] = reduced_state.get("youtube_results", [])[:1]
    reduced_state["synthesis_error"] = "" # Clear the error for retry
    
    return reduced_state