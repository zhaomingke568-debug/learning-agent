import os
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class OfficialDocsRAG:
    """
    官方文档 RAG 类，使用 Chroma + BAAI/bge-m3 embeddings。
    支持添加官方文档片段并检索。
    """

    def __init__(self, persist_directory: str = "./chroma_official_docs"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.vectorstore = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        """初始化向量数据库"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except Exception:
            self.vectorstore = Chroma.from_documents(
                documents=[],
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """
        添加文档片段到向量库。

        Args:
            texts: 文档文本列表
            metadatas: 元数据列表（如 {"source": "python官方文档", "url": "..."}）
        """
        if not texts:
            return

        if metadatas is None:
            metadatas = [{"source": "official_docs"} for _ in texts]

        ids = [f"doc_{i}_{hash(t[:100])}" for i, t in enumerate(texts)]

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        检索与查询相关的文档片段。

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            List[Dict] - 包含 content, source, url 的列表
        """
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "official_docs"),
                    "url": doc.metadata.get("url", "")
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"RAG 检索失败: {e}")
            return []

    def clear(self):
        """清空向量库"""
        try:
            self.vectorstore.delete_collection()
            self._init_vectorstore()
        except Exception:
            pass


# 全局单例
_official_docs_rag = None


def get_official_docs_rag() -> OfficialDocsRAG:
    """获取全局 OfficialDocsRAG 实例"""
    global _official_docs_rag
    if _official_docs_rag is None:
        _official_docs_rag = OfficialDocsRAG()
    return _official_docs_rag


def init_official_docs_rAG():
    """
    初始化官方文档 RAG，添加默认的 Python 官方文档片段。
    在实际使用时，应该从官方文档网站抓取内容。
    这里用内置的 Python 知识作为演示。
    """
    rag = get_official_docs_rag()

    # 内置的 Python 官方文档片段（实际应用中应从网站抓取）
    default_docs = [
        {
            "text": "Python 装饰器是用于修改函数或类行为的函数。装饰器使用 @decorator_name 语法 적용。",
            "metadata": {"source": "Python 官方文档 - 函数", "url": "https://docs.python.org/3/tutorial/classes.html#decorators"}
        },
        {
            "text": "装饰器可以接受参数，也可以是类。类装饰器必须返回可调用对象。",
            "metadata": {"source": "Python 官方文档 - 函数", "url": "https://docs.python.org/3/tutorial/classes.html#decorators"}
        },
        {
            "text": "functools.wraps 用于保留被装饰函数的元信息（name, docstring 等）。",
            "metadata": {"source": "Python functools 模块", "url": "https://docs.python.org/3/library/functools.html"}
        },
        {
            "text": "装饰器执行顺序：从下到上。先应用 @a，再应用 @b，结果是 b(a(original_func))。",
            "metadata": {"source": "Python 官方文档 - 函数", "url": "https://docs.python.org/3/tutorial/classes.html#decorators"}
        },
        {
            "text": "staticmethod 和 classmethod 是内置装饰器，用于定义类方法。",
            "metadata": {"source": "Python 内置函数", "url": "https://docs.python.org/3/library/functions.html"}
        },
    ]

    for doc in default_docs:
        rag.add_documents([doc["text"]], [doc["metadata"]])

    print(f"✅ 官方文档 RAG 初始化完成，已加载 {len(default_docs)} 个文档片段")
    return rag


# 测试
if __name__ == "__main__":
    rag = init_official_docs_rAG()
    results = rag.retrieve("装饰器", k=3)
    print("\n检索结果:")
    for r in results:
        print(f"  - [{r['source']}] {r['content'][:50]}...")