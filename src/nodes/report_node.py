import os
from typing import Dict

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from src.state import AgentState


def report_node(state: AgentState) -> dict:
    """
    生成 HTML 格式的学习进度报告。
    """
    topic = state.get("topic", "未知")
    learning_path = state.get("current_learning_path", {})
    milestone_index = state.get("current_milestone_index", 0)
    assessment_result = state.get("assessment_result", {})
    code_execution_result = state.get("code_execution_result", {})
    user_profile = state.get("user_profile", {})

    milestones = learning_path.get("milestones", [])
    completed_milestones = milestones[:milestone_index + 1] if milestone_index > 0 else [milestones[0]] if milestones else []

    # 构建 HTML 报告
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>学习进度报告 - {topic}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; }}
        .milestone {{ padding: 10px; margin: 5px 0; background: #fff; border-left: 4px solid #4CAF50; }}
        .result {{ color: #4CAF50; font-weight: bold; }}
        .weak-points {{ color: #f44336; }}
    </style>
</head>
<body>
    <h1>📚 学习进度报告</h1>
    <div class="section">
        <h2>学习主题</h2>
        <p>{topic}</p>
    </div>
    <div class="section">
        <h2>能力评估</h2>
        <p>级别：<strong>{assessment_result.get('level', '未知')}</strong></p>
        <p>正确率：{assessment_result.get('correct_rate', 0) * 100:.0f}%</p>
    </div>
    <div class="section">
        <h2>学习路径</h2>
        <p>总里程碑数：{len(milestones)}</p>
        <p>已完成：{milestone_index + 1} / {len(milestones)}</p>
        <h3>已完成里程碑：</h3>
        {"".join([f'<div class="milestone">✅ {m.get("topic", "")}</div>' for m in completed_milestones])}
    </div>
    <div class="section">
        <h2>代码执行结果</h2>
        <p class="result">{"✅ 通过" if code_execution_result.get("all_passed", False) else "❌ 未通过"}</p>
        <p>摘要：{code_execution_result.get("summary", "N/A")}</p>
    </div>
    <div class="section">
        <h2>用户画像</h2>
        <p>强项：{", ".join(user_profile.get("strong_points", [])) or "暂无"}</p>
        <p class="weak-points">薄弱点：{", ".join(user_profile.get("weak_points", [])) or "暂无"}</p>
    </div>
</body>
</html>"""

    print(f"\n--- 📊 生成 HTML 报告 ---")
    print(f"报告长度: {len(html)} 字符")

    return {
        "html_report": html,
        "next_step": "get_feedback"
    }