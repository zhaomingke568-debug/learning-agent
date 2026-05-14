from fastmcp import FastMCP
import llama_parse
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")

# 初始化 MCP
mcp = FastMCP("PDF-Multimodal-RAG")
LLAMA_CLOUD_API_KEY="llx-AWm2cdNrGCtWwC47FudLQvVLvUFdYzLVZl1JBNPknVVvAUgU"
# 初始化解析器和向量库
parser = llama_parse.LlamaParse(result_type="markdown", api_key=LLAMA_CLOUD_API_KEY) 
db = PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="pdf_content", embedding_function=ef)

@mcp.tool()
def ingest_pdf(path: str):
    """解析 PDF 并将图文内容存入向量库"""
    # 1. 解析 PDF (包含图片转描述)
    documents = parser.load_data(path)
    
    # 2. 存入向量数据库
    for i, doc in enumerate(documents):
        collection.add(
            documents=[doc.text],
            ids=[f"{path}_{i}"],
            metadatas=[{"source": path, "type": "text_or_image_desc"}]
        )
    print(f"已完成 {path} 的索引，共处理 {len(documents)} 个区块。")

@mcp.tool()
def query_pdf(question: str):
    """在已索引的 PDF 中检索相关信息"""
    results = collection.query(query_texts=[question], n_results=3)
    print(results["documents"])

if __name__ == "__main__":
    #ingest_pdf("E://learning-agent//src//download_pdfs//LN02.pdf")
    query_pdf("control systems 架构图？")