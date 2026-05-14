import os
import uuid
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from src.state import AgentState, Exercise
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


llm = ChatAnthropic(
    model="MiniMax-M2.7",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    max_retries=5
)


def exercise_node(state: AgentState) -> dict:
    """
    根据当前里程碑生成练习题。
    包含 starter_code、test_cases（可见 + 隐藏）、hints。
    """
    topic = state["topic"]
    user_profile = state.get("user_profile", {})
    learning_path = state.get("current_learning_path", {})
    milestone_index = state.get("current_milestone_index", 0)
    current_explanation = state.get("current_explanation", "")

    # 获取当前里程碑
    milestones = learning_path.get("milestones", [])
    if milestone_index >= len(milestones):
        current_milestone = milestones[-1] if milestones else {"topic": topic, "description": ""}
    else:
        current_milestone = milestones[milestone_index]

    current_topic = current_milestone.get("topic", topic)

    print(f"\n--- 📝 生成练习题：{current_topic} ---")

    user_level = user_profile.get("capability_level", state.get("depth_level", "入门"))

    # 生成练习题
    exercise = _generate_exercise(
        topic=current_topic,
        user_level=user_level,
        explanation=current_explanation
    )

    print(f"\n--- ✅ 练习题已生成 ---")
    print(f"标题：{exercise['title']}")
    print(f"难度：{exercise['difficulty']}")
    print(f"测试用例数：{len(exercise['test_cases'])}（其中 {sum(1 for tc in exercise['test_cases'] if tc.get('hidden'))} 个隐藏）")

    return {
        "current_exercise": exercise,
        "review_type": "exercise",
        "next_step": "review"
    }


def _generate_exercise(topic: str, user_level: str, explanation: str) -> Dict:
    """使用 LLM 生成练习题"""
    system_prompt = """你是一个专业的编程练习题生成专家。根据知识点和用户级别，生成一道高质量的编程练习。

主题：{topic}
用户级别：{user_level}

要求：
1. 题目要具体、可执行、有明确答案
2. starter_code 是用户看到的起始代码（包含函数签名，不包含完整实现）
3. solution_code 是完整的正确答案
4. test_cases 包含输入输出对，hidden=True 的用例对用户隐藏
5. hints 提供 2-3 个渐进式提示
6. difficulty 根据级别设定：入门-easy，进阶-medium，专家-hard

输出格式（严格 JSON）：
{{
  "exercise_id": "uuid",
  "title": "练习标题",
  "description": "题目描述",
  "starter_code": "def solution():\\n    pass",
  "solution_code": "def solution():\\n    return 42",
  "test_cases": [
    {{"input": "test_input", "expected": "42", "hidden": false}},
    {{"input": "test_input2", "expected": "43", "hidden": true}}
  ],
  "hints": ["提示1", "提示2", "提示3"],
  "difficulty": "easy|medium|hard",
  "topic_tags": ["{topic}"]
}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "为主题「{topic}」生成练习题。\n\n知识点讲解内容：\n{explanation}")
    ])

    parser = JsonOutputParser(pydantic_object=dict)

    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "topic": topic,
            "user_level": user_level,
            "explanation": explanation[:500] if explanation else ""
        })
        # 确保有 exercise_id
        if "exercise_id" not in result:
            result["exercise_id"] = str(uuid.uuid4())
        return result
    except Exception as e:
        print(f"生成练习题失败，使用默认练习: {e}")
        # 返回默认练习
        return {
            "exercise_id": str(uuid.uuid4()),
            "title": f"{topic} 练习",
            "description": f"请完成以下{topic}相关的练习",
            "starter_code": f"def solution():\n    # 请在此实现你的代码\n    pass",
            "solution_code": f"def solution():\n    return True",
            "test_cases": [
                {"input": "", "expected": "True", "hidden": False}
            ],
            "hints": ["仔细阅读题目要求", "注意边界情况"],
            "difficulty": "easy",
            "topic_tags": [topic]
        }