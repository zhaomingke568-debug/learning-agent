import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # 如果用 Minimax，请替换为其专属 Embedding
from langchain_community.vectorstores import FAISS
from src.state import AgentState
def pdf_rag_node(state: AgentState) -> dict:
    """Extract precise context from downloaded PDFs."""
    print("--- EXECUTING DEEP READ (PDF RAG) ---")
    
    pdf_paths = state.get("downloaded_pdfs", [])
    topic = state.get("topic", "")
    
    if not pdf_paths:
        print("  -> No PDFs to process. Skipping.")
        return {"pdf_context": []}

    all_docs = []
    
    # ==========================================
    # 步骤 1：加载与解析 PDF
    # ==========================================
    for path in pdf_paths:
        try:
            print(f"  -> Loading {path}...")
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ❌ Error loading {path}: {e}")
            continue

    if not all_docs:
        return {"pdf_context": []}

    # ==========================================
    # 步骤 2：智能文本切块 (Chunking)
    # ==========================================
    # 设置 chunk_size 为 1000 字符，重叠 200 字符以保持上下文连贯
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"  -> Split into {len(splits)} chunks.")

    # ==========================================
    # 步骤 3：构建内存级向量数据库 (FAISS)
    # ==========================================
    try:
        print("  -> Building temporary Vector Store...")
        # ⚠️ 注意：这里使用的是 OpenAI 的 Embedding 模型
        # 如果你用的是 MiniMax，请替换为对应的 Embedding 接口
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 
        
        # 在内存中瞬间建立 FAISS 索引
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # ==========================================
        # 步骤 4：基于用户主题进行精准检索
        # ==========================================
        print(f"  -> Retrieving top chunks for topic: '{topic}'")
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3} # 只取最相关的 3 个文本块
        )
        
        retrieved_docs = retriever.invoke(topic)
        
        # 提取有用信息，格式化为字典列表
        extracted_info = []
        for i, doc in enumerate(retrieved_docs):
            extracted_info.append({
                "source": doc.metadata.get("source", "Unknown PDF"),
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content
            })
            
        print("--- ✅ DEEP READ COMPLETE ---")
        return {"pdf_context": extracted_info}
        
    except Exception as e:
        print(f"--- ❌ RAG PROCESS FAILED: {e} ---")
        # 容错：如果向量化失败，返回空列表，不阻断 Graph 流转
        return {"pdf_context": []}