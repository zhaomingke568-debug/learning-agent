import os
import uuid
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from src.state import AgentState
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


class Question(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str = ""


class QuestionsList(BaseModel):
    questions: List[Question]


llm = ChatAnthropic(
    model="MiniMax-M2.7",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    max_retries=5
)


class AssessmentResult(BaseModel):
    level: str = Field(description="用户的能力级别：入门/进阶/专家")
    strengths: List[str] = Field(description="用户的强项")
    weaknesses: List[str] = Field(description="用户的薄弱点")
    confidence: float = Field(description="评估置信度 0.0-1.0")


def assessment_node(state: AgentState) -> dict:
    """
    评估用户当前水平的节点。

    流程：
    1. 根据 topic 生成 3-5 道诊断题
    2. 让用户回答（通过 CLI）
    3. 分析回答，判断 level
    4. 更新 user_profile
    """
    topic = state["topic"]
    depth_level = state.get("depth_level", "入门")

    print(f"\n--- 📝 能力评估：{topic} ---")
    print(f"目标级别：{depth_level}\n")

    # 生成诊断题
    questions = _generate_questions(topic, depth_level)

    print("【诊断题目】请回答以下问题（输入选项字母，如 A/B/C/D）：\n")
    for i, q in enumerate(questions, 1):
        print(f"Q{i}. {q['question']}")
        for opt in q.get("options", []):
            print(f"   {opt}")
        print()

    # 收集用户回答
    answers = []
    for i, q in enumerate(questions, 1):
        user_answer = input(f"Q{i} 你的答案: ").strip().upper()
        answers.append({"question_id": f"q{i}", "question": q["question"], "answer": user_answer})

    # 分析结果（传入题目和回答，计算正确率）
    result = _analyze_answers(topic, questions, answers)

    print(f"\n--- ✅ 评估完成 ---")
    print(f"你的水平：**{result['level']}**")
    print(f"强项：{', '.join(result['strengths'])}")
    print(f"薄弱点：{', '.join(result['weaknesses'])}")

    # 构建 user_profile
    user_id = state.get("session_id", str(uuid.uuid4()))
    profile = {
        "user_id": user_id,
        "capability_level": result["level"],
        "weak_points": result["weaknesses"],
        "strong_points": result["strengths"],
        "learning_history": [],
        "preferences": {}
    }

    return {
        "assessment_result": result,
        "assessment_answers": answers,
        "user_profile": profile,
        "next_step": "learning_path_planning"
    }


def _generate_questions(topic: str, depth_level: str) -> List[Dict]:
    """使用 LLM 生成诊断题"""
    system_prompt = """你是一个专业的编程教育评估专家。根据用户想学习的主题和目标级别，生成 3-5 道诊断题。

主题：{topic}
目标级别：{depth_level}

要求：
1. 题目应该覆盖该主题的核心概念、常见用法和易错点
2. 选项应该有一定迷惑性，区分不同水平的学习者
3. 题目难易度应该与目标级别匹配

请以 JSON 格式输出，格式如下：
{{"questions": [
  {{"question": "题目内容", "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"], "correct_answer": "B", "explanation": "解释"}}
]}}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "为主题「{topic}」（{depth_level}）生成诊断题目")
    ])
    
    try:
        structured_llm = llm.with_structured_output(QuestionsList)
        chain = prompt | structured_llm

        result = chain.invoke({
            "topic": topic,
            "depth_level": depth_level
        })

        print(f"Debug: structured invoke succeeded, result type={type(result)}")

        if not result.questions:
            raise ValueError("questions 为空")

        return [q.model_dump() for q in result.questions]
    
        
    except Exception as e:
        print(f"生成题目失败，使用默认题目: {e}")
        # 返回默认题目
        return [
            {
                "question": f"关于「{topic}」，你之前有过相关学习或实践经验吗？",
                "options": ["A. 完全没接触过", "B. 看过一些基础概念", "C. 有过实际项目经验", "D. 非常熟悉"],
                "correct_answer": "A",
                "explanation": "了解用户的背景"
            }
        ]


def _analyze_answers(topic: str, questions: List[Dict], answers: List[Dict]) -> Dict:
    """分析用户回答，根据正确率判断能力级别"""

    # 计算正确率
    correct_count = 0
    total_count = len(questions)

    # 建立题号到正确答案的映射
    question_correct_map = {q.get("question", ""): q.get("correct_answer", "").upper() for q in questions}

    for answer in answers:
        q_text = answer.get("question", "")
        user_ans = answer.get("answer", "").upper()
        correct_ans = question_correct_map.get(q_text, "").upper()

        if user_ans == correct_ans:
            correct_count += 1

    correct_rate = correct_count / total_count if total_count > 0 else 0

    # 根据正确率判断级别
    # 0-40%: 入门, 41-70%: 进阶, 71-100%: 专家
    if correct_rate <= 0.4:
        level = "入门"
    elif correct_rate <= 0.7:
        level = "进阶"
    else:
        level = "专家"

    # 分析薄弱点和强项（根据答错的题推断）
    wrong_questions = []
    correct_questions = []
    for answer in answers:
        q_text = answer.get("question", "")
        user_ans = answer.get("answer", "").upper()
        correct_ans = question_correct_map.get(q_text, "").upper()
        if user_ans != correct_ans:
            wrong_questions.append(q_text)
        else:
            correct_questions.append(q_text)

    # 返回结果
    return {
        "level": level,
        "correct_rate": correct_rate,
        "correct_count": correct_count,
        "total_count": total_count,
        "strengths": correct_questions[:2],  # 答对的题目作为强项参考
        "weaknesses": wrong_questions[:2],     # 答错的题目作为薄弱点参考
        "confidence": 0.8 if total_count >= 3 else 0.5
    }

