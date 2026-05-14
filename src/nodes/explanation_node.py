import os
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from src.state import AgentState
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from src.tools.rag.enhanced_rag import get_official_docs_rag


llm = ChatAnthropic(
    model="MiniMax-M2.7",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    max_retries=5
)


def explanation_node(state: AgentState) -> dict:
    """
    讲解当前里程碑的知识点。
    结合用户 level、weak_points、pdf_context 和 official_docs_context 进行 RAG 增强。
    """
    topic = state["topic"]
    depth_level = state.get("depth_level", "入门")
    user_profile = state.get("user_profile", {})
    learning_path = state.get("current_learning_path", {})
    milestone_index = state.get("current_milestone_index", 0)
    assessment_result = state.get("assessment_result", {})

    # 获取当前里程碑
    milestones = learning_path.get("milestones", [])
    if milestone_index >= len(milestones):
        current_milestone = milestones[-1] if milestones else {"topic": topic, "description": ""}
    else:
        current_milestone = milestones[milestone_index]

    current_topic = current_milestone.get("topic", topic)

    print(f"\n--- 📖 知识点讲解：{current_topic} ---")

    # 获取 RAG 上下文（PDF + 官方文档检索）
    pdf_context = state.get("pdf_context", [])

    # 从官方文档 RAG 检索相关片段
    official_docs_rag = get_official_docs_rag()
    official_docs = official_docs_rag.retrieve(current_topic, k=3)

    # 构建 RAG 上下文字符串
    rag_context = ""
    if pdf_context:
        rag_context += "\n【PDF 文档内容】\n" + "\n".join([f"- {c.get('content', '')[:200]}" for c in pdf_context[:3]])
    if official_docs:
        rag_context += "\n【官方文档内容】\n" + "\n".join([f"- {c.get('content', '')[:200]}" for c in official_docs[:3]])

    # 生成讲解
    explanation = _generate_explanation(
        topic=current_topic,
        user_level=user_profile.get("capability_level", depth_level),
        weak_points=assessment_result.get("weaknesses", []),
        description=current_milestone.get("description", ""),
        rag_context=rag_context,
        official_docs=official_docs
    )

    print(f"\n--- ✅ 讲解生成完毕 ---\n")

    return {
        "current_explanation": explanation,
        "explanation_references": pdf_context[:3] + official_docs,
        "official_docs_context": official_docs,
        "review_type": "explanation",
        "next_step": "review"
    }


def _generate_explanation(topic: str, user_level: str, weak_points: List[str],
                          description: str, rag_context: str, official_docs: List[Dict]) -> str:
    """使用 LLM 生成知识点讲解"""
    weak_points_str = ", ".join(weak_points) if weak_points else "无"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个资深的技术讲师，负责讲解编程知识点。

你的讲解应该：
1. 清晰、准确、有条理
2. 包含核心概念解释和代码示例
3. 针对用户的薄弱点重点讲解
4. 难易度匹配用户级别（{user_level}）
5. 使用 Markdown 格式输出

用户薄弱点：{weak_points_str}

{rag_context}
{official_docs}
"""),
        ("human", """讲解主题：{topic}
主题描述：{description}

请生成一份详细的知识点讲解，包含：
1. 核心概念（用简短的话解释清楚）
2. 常见用法（代码示例）
3. 注意事项（新手容易犯的错误）
4. 练习建议

格式要求：使用 Markdown，包含代码块示例。
""")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "topic": topic,
            "description": description,
            "rag_context": rag_context,
            "official_docs": official_docs,
            "weak_points": weak_points_str,
            "user_level": user_level    
        })
        return result
    except Exception as e:
        print(f"生成讲解失败: {e}")
        return f"# {topic}\n\n讲解内容生成失败，请稍后重试。\n\n错误: {e}"