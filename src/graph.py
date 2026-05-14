import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

from langgraph.graph import StateGraph, START, END
from src.state import AgentState


def build_graph():
    """
    构建 Phase 2 的 StateGraph（含 RAG + Review）。

    流程：
    START → assess_capability → learning_path_planning
          → knowledge_explanation → review(explanation)
          → exercise_generation → review(exercise)
          → code_execution → review(code)
          → report_generation → get_feedback
          → (继续? advance_milestone → explanation : END)
    """
    workflow = StateGraph(AgentState)

    # === 导入节点 ===
    from src.nodes.assessment_node import assessment_node
    from src.nodes.learning_path_node import learning_path_node
    from src.nodes.explanation_node import explanation_node
    from src.nodes.exercise_node import exercise_node
    from src.nodes.code_execution_node import code_execution_node
    from src.nodes.feedback_node import get_feedback_node
    from src.nodes.advance_node import advance_milestone_node
    from src.nodes.review_node import review_node
    from src.nodes.report_node import report_node

    # === 注册节点 ===
    workflow.add_node("assess_capability", assessment_node)
    workflow.add_node("learning_path_planning", learning_path_node)
    workflow.add_node("knowledge_explanation", explanation_node)
    workflow.add_node("exercise_generation", exercise_node)
    workflow.add_node("code_execution", code_execution_node)
    workflow.add_node("get_feedback", get_feedback_node)
    workflow.add_node("advance_milestone", advance_milestone_node)
    workflow.add_node("review", review_node)
    workflow.add_node("report_generation", report_node)

    # === 设置边 ===
    workflow.add_edge(START, "assess_capability")
    workflow.add_edge("assess_capability", "learning_path_planning")
    workflow.add_edge("learning_path_planning", "knowledge_explanation")

    # === 知识讲解 → review(explanation) ===
    workflow.add_edge("knowledge_explanation", "review")

    # === exercise_generation → review(exercise) ===
    workflow.add_edge("exercise_generation", "review")

    # === code_execution → review(code) ===
    workflow.add_edge("code_execution", "review")

    # === review → 根据 verdict 决定后续节点 ===
    workflow.add_conditional_edges(
        "review",
        _review_router,
        {
            "continue_explanation": "knowledge_explanation",  # 讲解不通过，重新生成
            "continue_exercise": "exercise_generation",        # 练习不通过，重新生成
            "continue_code": "code_execution",               # 代码不通过，重新提交
            "continue_report": "report_generation",          # 审核通过，生成报告
            "continue_feedback": "get_feedback"              # 代码审核通过，进入反馈
        }
    )

    workflow.add_edge("report_generation", "get_feedback")

    # === 反馈循环 ===
    workflow.add_conditional_edges(
        "get_feedback",
        _feedback_router,
        {
            "continue": "advance_milestone",
            "end": END
        }
    )
    workflow.add_edge("advance_milestone", "knowledge_explanation")

    return workflow.compile()


def _review_router(state: AgentState) -> str:
    """
    Review 路由：根据 review_type 和 verdict 决定后续节点。
    """
    review_type = state.get("review_type", "explanation")
    review_result = state.get("review_result", {})
    verdict = review_result.get("verdict", "approved")

    # 如果审核不通过，返回到对应节点重新生成
    if verdict == "needs_revision":
        if review_type == "explanation":
            return "continue_explanation"
        elif review_type == "exercise":
            return "continue_exercise"
        elif review_type == "code":
            return "continue_code"

    # 审核通过，根据 review_type 决定后续
    if review_type == "explanation":
        return "continue_exercise"  # 讲解通过，进入练习生成
    elif review_type == "exercise":
        return "continue_code"      # 练习通过，进入代码执行
    elif review_type == "code":
        return "continue_feedback" # 代码通过，进入反馈（或报告）
    else:
        return "continue_feedback"


def _feedback_router(state: AgentState) -> str:
    """
    反馈路由：
    - 用户满意或完成所有 milestone → END
    - 用户想继续 → advance_milestone
    """
    user_feedback = state.get("user_feedback", "").strip().lower()
    learning_path = state.get("current_learning_path", {})
    milestone_index = state.get("current_milestone_index", 0)
    milestones = learning_path.get("milestones", [])

    # 检查是否还有下一个 milestone
    has_next = (milestone_index + 1) < len(milestones)

    if not user_feedback or user_feedback in ["ok", "满意", "继续", "next", "y", "yes"]:
        if has_next:
            return "continue"
        else:
            return "end"

    if user_feedback in ["退出", "quit", "exit", "q"]:
        return "end"

    # 默认继续
    if has_next:
        return "continue"
    return "end"