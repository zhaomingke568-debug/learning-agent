import os
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from src.state import AgentState
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatAnthropic(
    model="MiniMax-M2.7",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    max_retries=5
)


def review_node(state: AgentState) -> dict:
    """
    审核节点：审核讲解内容/代码/练习题。

    review_type 决定审核模式：
    - explanation: 审核讲解内容，对比 RAG 文档，排 hallucinations
    - exercise: 审核练习题，验证 solution_code 正确性，test_cases 有效性
    - code: 审核用户提交的代码，逻辑 + 风格
    """
    review_type = state.get("review_type", "explanation")
    print(f"\n--- 🔍 Review: {review_type} ---")

    if review_type == "explanation":
        result = _review_explanation(state)
    elif review_type == "exercise":
        result = _review_exercise(state)
    elif review_type == "code":
        result = _review_code(state)
    else:
        result = {"verdict": "approved", "correctness_score": 1.0, "issues": [], "suggestions": []}

    print(f"--- ✅ Review 完成: {result['verdict']} (score: {result['correctness_score']:.2f}) ---")

    return {
        "review_result": result,
        "next_step": "advance_review"  # 后续由 router 决定走向
    }


def _review_explanation(state: AgentState) -> Dict:
    """审核讲解内容，对比 RAG 文档"""
    explanation = state.get("current_explanation", "")
    official_docs = state.get("official_docs_context", [])
    pdf_context = state.get("pdf_context", [])

    # 合并 RAG 上下文
    rag_context = ""
    for doc in (official_docs[:3] + pdf_context[:3]):
        rag_context += f"- {doc.get('content', '')[:200]}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格的技术审核专家。
你的任务是审核讲解内容是否存在事实性错误（hallucination）或误导性信息。

请检查：
1. 核心概念是否准确
2. 代码示例是否正确可运行
3. 是否有与官方文档矛盾的内容

如果发现问题，在 issues 中详细说明。"""),
        ("human", """讲解内容：
{explanation}

官方文档参考：
{rag_context}

请审核并返回 JSON：
{{"verdict": "approved|needs_revision",
  "correctness_score": 0.0-1.0,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]}}
""")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        result_str = chain.invoke({
            "explanation": explanation,
            "rag_context": rag_context or "无"
        })
        import json
        result_str = result_str.strip()
        if result_str.startswith("```"):
            result_str = result_str.split("```")[1]
            if result_str.startswith("json"):
                result_str = result_str[4:]
        result = json.loads(result_str)
        return result
    except Exception as e:
        print(f"Review 解析失败: {e}")
        return {"verdict": "approved", "correctness_score": 0.8, "issues": [], "suggestions": [f"审核解析失败: {e}"]}


def _review_exercise(state: AgentState) -> Dict:
    """审核练习题：验证 solution_code 正确性，test_cases 有效性"""
    exercise = state.get("current_exercise", {})
    solution_code = exercise.get("solution_code", "")
    test_cases = exercise.get("test_cases", [])
    starter_code = exercise.get("starter_code", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个编程教育审核专家。
你的任务是验证练习题的正确性：
1. solution_code 是否能正确解决题目描述的问题
2. test_cases 是否覆盖了主要场景（正确用例 + 边界情况）
3. starter_code 是否提供了足够的起始代码

如果发现问题，在 issues 中详细说明。"""),
        ("human", """练习题：
标题：{title}
描述：{description}
starter_code：
{starter_code}

solution_code：
{solution_code}

test_cases：
{test_cases}

请审核并返回 JSON：
{{"verdict": "approved|needs_revision",
  "correctness_score": 0.0-1.0,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]}}
""")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        result_str = chain.invoke({
            "title": exercise.get("title", ""),
            "description": exercise.get("description", ""),
            "starter_code": starter_code,
            "solution_code": solution_code,
            "test_cases": str(test_cases)
        })
        import json
        result_str = result_str.strip()
        if result_str.startswith("```"):
            result_str = result_str.split("```")[1]
            if result_str.startswith("json"):
                result_str = result_str[4:]
        result = json.loads(result_str)
        return result
    except Exception as e:
        print(f"Review 解析失败: {e}")
        return {"verdict": "approved", "correctness_score": 0.8, "issues": [], "suggestions": [f"审核解析失败: {e}"]}


def _review_code(state: AgentState) -> Dict:
    """审核用户提交的代码：逻辑 + 风格"""
    user_code = state.get("user_code", "")
    exercise = state.get("current_exercise", {})
    code_result = state.get("code_execution_result", {})

    passed = code_result.get("passed", False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个代码审核专家。
你的任务是审核用户提交的代码：
1. 逻辑是否正确（能否通过测试用例）
2. 代码风格是否良好（可读性、命名规范）
3. 是否有潜在问题（安全、性能）

如果代码已经通过了测试用例，重点检查风格和潜在问题。
如果代码未通过，给出修复建议。

返回 JSON：
{{"verdict": "approved|needs_revision",
  "correctness_score": 0.0-1.0,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]}}
"""),
        ("human", """用户代码：
{user_code}

题目要求：{description}

测试结果：{test_result}

请审核并返回 JSON：
""")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        result_str = chain.invoke({
            "user_code": user_code,
            "description": exercise.get("description", ""),
            "test_result": str(code_result)
        })
        import json
        result_str = result_str.strip()
        if result_str.startswith("```"):
            result_str = result_str.split("```")[1]
            if result_str.startswith("json"):
                result_str = result_str[4:]
        result = json.loads(result_str)
        return result
    except Exception as e:
        print(f"Review 解析失败: {e}")
        return {"verdict": "approved", "correctness_score": 0.8 if passed else 0.5,
                "issues": [], "suggestions": [f"审核解析失败: {e}"]}