import sys
import os

# 获取当前 test_rag.py 所在的绝对路径（也就是你的项目根目录 E:\learning-agent）
project_root = "E:\\learning-agent"

# 强行把根目录塞进 Python 的环境变量里
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from src.tools.rag.rag_pdf import pdf_rag_node
from src.state import AgentState

# 加载环境变量
load_dotenv()

def test_retrieval(pdf_path: str, test_topic: str):
    """
    测试 PDF RAG 节点的检索效果
    """
    print(f"\n--- Testing RAG for topic: '{test_topic}' ---")
    
    if not os.path.exists(pdf_path):
        print(f"  ❌ Error: Test PDF not found at {pdf_path}")
        print("  Please provide a valid PDF path to run the test.")
        return

    # 构造模拟状态
    state: AgentState = {
        "topic": test_topic,
        "downloaded_pdfs": [pdf_path],
        "pdf_context": []
    }
    
    # 执行 RAG 节点
    result = pdf_rag_node(state)
    
    # 打印检索到的内容
    retrieved_context = result.get("pdf_context", [])
    if not retrieved_context:
        print("  ⚠️ No context retrieved.")
    else:
        print(f"  ✅ Successfully retrieved {len(retrieved_context)} chunks:")
        for i, ctx in enumerate(retrieved_context):
            print(f"\n[Chunk {i+1}] (Source: {ctx['source']}, Page: {ctx['page']})")
            print("-" * 40)
            print(ctx['content'][:500] + "..." if len(ctx['content']) > 500 else ctx['content'])
            print("-" * 40)

if __name__ == "__main__":
    # 你可以修改这里的路径和主题进行测试
    # 假设你当前目录下有一个名为 'sample.pdf' 的文件
    TEST_PDF = "E:\\learning-agent\\src\\download_pdfs\\LN02.pdf" 
    TEST_TOPIC = "control system 是什么" # 替换为 PDF 中可能出现的主题关键词
    
    test_retrieval(TEST_PDF, TEST_TOPIC)
