import os
import uuid
from typing import Dict

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from src.state import AgentState
from src.sandbox.code_sandbox import get_sandbox
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatAnthropic(
    model="MiniMax-M2.5-highspeed",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    max_retries=5
)


def code_execution_node(state: AgentState) -> dict:
    """
    接收用户代码，调用沙箱执行，返回结果。
    如果执行失败，生成修复建议。
    """
    topic = state["topic"]
    current_exercise = state.get("current_exercise", {})
    user_code = state.get("user_code", "")

    print(f"\n--- ⚡ 代码执行 ---")
    print(f"当前练习：{current_exercise.get('title', '未知')}")

    # 如果没有用户代码，让用户输入
    if not user_code:
        print("\n请输入你的代码（输入完成后按回车，单独输入 'done' 结束输入）：\n")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "done":
                break
            lines.append(line)
        user_code = "\n".join(lines)

    print(f"\n--- 🔍 正在执行代码... ---\n")

    sandbox = get_sandbox()
    test_cases = current_exercise.get("test_cases", [])

    if test_cases:
        # 有测试用例，执行并验证
        result = sandbox.execute_with_test_cases(user_code, test_cases)
    else:
        # 无测试用例，只执行
        result = sandbox.execute(user_code)
        result["all_passed"] = result.get("passed", False)
        result["summary"] = "passed" if result["passed"] else "failed"

    execution_id = str(uuid.uuid4())

    print(f"\n--- 📊 执行结果 ---")
    print(f"执行 ID：{execution_id}")
    print(f"结果：{'✅ 通过' if result.get('all_passed', False) else '❌ 未通过'}")
    print(f"摘要：{result.get('summary', 'N/A')}")

    if not result.get("all_passed", False):
        error_msg = result.get("main_error", "") or result.get("error_message", "")
        if error_msg:
            print(f"错误信息：{error_msg}")

        # 生成修复建议
        fix_suggestion = _generate_fix_suggestion(user_code, error_msg, current_exercise)
        result["fix_suggestions"] = fix_suggestion
        print(f"\n💡 修复建议：{fix_suggestion}")

    return {
        "code_execution_result": result,
        "user_code": user_code,
        "review_type": "code",
        "next_step": "review"
    }


def _generate_fix_suggestion(code: str, error_msg: str, exercise: Dict) -> str:
    """使用 LLM 生成代码修复建议"""
    if not error_msg:
        return "代码执行通过，无需修复。"

    exercise_title = exercise.get("title", "")
    exercise_description = exercise.get("description", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个 Python 调试专家。根据错误信息和题目要求，给出简洁的修复建议。

规则：
1. 指出错误类型和具体位置
2. 解释为什么出错
3. 给出修复方向（不要直接给完整代码）
4. 保持简洁，3句话以内
"""),
        ("human", """题目：{title}
描述：{description}

错误信息：
{error_msg}

用户代码：
{code}

请给出修复建议：
""")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        return chain.invoke({
            "title": exercise_title,
            "description": exercise_description,
            "error_msg": error_msg,
            "code": code
        })
    except Exception as e:
        print(f"生成修复建议失败: {e}")
        return f"代码执行出错，请检查代码逻辑。错误信息: {error_msg[:100]}"