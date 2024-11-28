import hashlib
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 初始化 Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# 初始化 Ollama LLM
llm = Ollama(
    model="kenneth85/llama-3-taiwan:8b-instruct",
    callback_manager=callback_manager
)

# 初始化 Ollama Embeddings
embeddings = OllamaEmbeddings(model="kenneth85/llama-3-taiwan:8b-instruct")

# 向量資料庫存儲目錄
persist_directory = "chroma_vectorstore"

# PDF 文件路徑
pdf_path = "pdf_test.pdf"

# 計算 PDF 文件哈希
def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# 哈希文件存儲路徑
hash_file_path = os.path.join(persist_directory, "pdf_hash.txt")

# 檢查是否需要更新數據庫
def needs_update(pdf_path, hash_file_path):
    new_hash = calculate_file_hash(pdf_path)
    if not os.path.exists(hash_file_path):
        return True, new_hash
    with open(hash_file_path, "r") as f:
        existing_hash = f.read().strip()
    return new_hash != existing_hash, new_hash

# 檢查是否需要更新向量數據庫
update_required, new_hash = needs_update(pdf_path, hash_file_path)

if os.path.exists(persist_directory) and not update_required:
    print("---正在加載現有的向量數據庫...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("---PDF 文件已更新，正在重新生成數據庫...")

    # 清理舊數據庫
    if os.path.exists(persist_directory):
        print("---正在刪除舊的向量數據庫...")
        import shutil
        shutil.rmtree(persist_directory)

    # 步驟 1：載入 PDF
    print("---正在載入文件，請稍候...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 驗證文件內容
    print(f"---PDF 加載完成，共 {len(documents)} 頁")
    for i, doc in enumerate(documents[:3]):  # 顯示前3頁
        print(f"第 {i+1} 頁內容（部分）:\n{doc.page_content[:50]}")

    # 步驟 2：分割文本
    print("---正在分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 檢查分割後的文檔
    print(f"---分割後的文檔數量: {len(docs)}")
    print(f"第一段內容（部分）:\n{docs[0].page_content[:50]}")

    # 步驟 3：生成向量數據庫並持久化
    print("---正在生成向量數據庫...")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    vectorstore.persist()

    # 保存新的 PDF 哈希值
    with open(hash_file_path, "w") as f:
        f.write(new_hash)
    print("---向量數據庫已保存！")


# 步驟 4：建立檢索問答鏈
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 提供查詢功能
while True:
    query = input("請輸入查詢內容（輸入 'bye' 退出）：")
    if query.lower() == "bye":
        print("已退出查詢系統。")
        break
    result = qa_chain.invoke(query)
    print(f"查詢結果:\n{result}")
