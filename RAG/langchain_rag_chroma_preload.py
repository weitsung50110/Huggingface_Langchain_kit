import hashlib
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document

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
persist_directory = "chroma_vectorstore_nba"

# 範例 NBA 數據
nba_data = [
    {"player": "好崴寶Weibert", "points_per_game": 30.1, "assists_per_game": 5.7, "team": "Weibert"},
    {"player": "孟孟", "points_per_game": 29.7, "assists_per_game": 8.7, "team": "Mengbert"},
    {"player": "崴崴Weiberson", "points_per_game": 25.1, "assists_per_game": 10.5, "team": "Weiberson"}
]

# 計算數據哈希
def calculate_data_hash(data):
    hasher = hashlib.md5()
    hasher.update(str(data).encode('utf-8'))
    return hasher.hexdigest()

# 哈希文件存儲路徑
hash_file_path = os.path.join(persist_directory, "data_hash.txt")

# 檢查是否需要更新數據庫
def needs_update(data, hash_file_path):
    new_hash = calculate_data_hash(data)
    if not os.path.exists(hash_file_path):
        return True, new_hash
    with open(hash_file_path, "r") as f:
        existing_hash = f.read().strip()
    return new_hash != existing_hash, new_hash

# 檢查是否需要更新向量數據庫
update_required, new_hash = needs_update(nba_data, hash_file_path)

if os.path.exists(persist_directory) and not update_required:
    print("---正在加載現有的向量數據庫...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("---數據已更新，正在重新生成數據庫...")

    # 清理舊數據庫
    if os.path.exists(persist_directory):
        print("---正在刪除舊的向量數據庫...")
        import shutil
        shutil.rmtree(persist_directory)

    # 步驟 1：生成嵌入並存儲數據
    print("---正在生成向量數據庫...")

    # 將數據轉換為 Document 對象
    documents = [
        Document(page_content=data["player"], metadata=data) for data in nba_data
    ]

    # 創建 Chroma 向量數據庫
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory  # 啟用持久化
    )

    # 保存新的數據哈希值
    with open(hash_file_path, "w") as f:
        f.write(new_hash)
    print("---向量數據庫已保存！")

# 步驟 2：建立檢索問答鏈
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 提供查詢功能
def query_system():
    while True:
        query = input("請輸入查詢內容（輸入 'bye' 退出）：")
        if query.lower() == "bye":
            print("已退出查詢系統。")
            break

        # 檢索相關數據
        results = retriever.get_relevant_documents(query)
        if results:
            print("---檢索到的相關數據：")
            # 整理上下文
            context = "\n".join(
                [f"球員: {res.metadata['player']}, 場均得分: {res.metadata['points_per_game']}, 助攻: {res.metadata['assists_per_game']}, 球隊: {res.metadata['team']}" for res in results]
            )
            print(context)  # 檢查上下文
            # 傳遞給 LLM
            prompt = f"以下是相關數據：\n{context}\n\n根據以上數據，回答以下問題：{query}"
            response = qa_chain.invoke(prompt)
            print(f"回答：{response}")
        else:
            print("未找到相關數據，請嘗試其他查詢。")

query_system()
