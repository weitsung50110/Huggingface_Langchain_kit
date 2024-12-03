import os
import shutil
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings

# 向量資料庫存儲目錄
persist_directory = "chroma_vectorstore_nba"

# 初始化嵌入模型
embeddings = OllamaEmbeddings(model="kenneth85/llama-3-taiwan:8b-instruct")

# 初始化向量數據庫
if os.path.exists(persist_directory):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    os.makedirs(persist_directory)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 查看所有數據
def view_all():
    documents = vectorstore._collection.get(include=["metadatas", "documents"])
    if not documents["documents"]:
        print("資料庫為空！")
    else:
        for doc, metadata in zip(documents["documents"], documents["metadatas"]):
            print(f"內容: {doc}, 元數據: {metadata}")

# 新增數據
def add_data():
    print("輸入新增的數據 (格式: 玩家名, 場均得分, 助攻, 球隊)")
    data = input("輸入數據: ").split(",")
    if len(data) != 4:
        print("輸入格式錯誤！請重新嘗試。")
        return
    player, points_per_game, assists_per_game, team = data
    document = Document(
        page_content=player.strip(),
        metadata={
            "player": player.strip(),
            "points_per_game": float(points_per_game.strip()),
            "assists_per_game": float(assists_per_game.strip()),
            "team": team.strip(),
        },
    )
    vectorstore.add_documents([document])
    print(f"新增數據成功！玩家: {player.strip()}")

# 刪除所有數據
def delete_all():
    confirm = input("確定要刪除所有數據嗎？(yes/no): ")
    if confirm.lower() == "yes":
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory)
        global vectorstore
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("所有數據已刪除！")
    else:
        print("操作已取消。")

# 查詢相關數據
def query_data():
    query = input("輸入查詢內容: ")
    retriever = vectorstore.as_retriever()
    
    # 限制返回的最大結果數量
    results = retriever.get_relevant_documents(query)[:2]  # 最多返回 2 筆
    if results:
        print("---檢索到的相關數據：")
        for res in results:
            print(f"球員: {res.metadata['player']}, 場均得分: {res.metadata['points_per_game']}, 助攻: {res.metadata['assists_per_game']}, 球隊: {res.metadata['team']}")
    else:
        print("未找到相關數據！")

# 主程序
def main():
    print("Chroma 向量數據庫管理系統")
    print("功能列表：")
    print("1. 查看所有數據")
    print("2. 新增數據")
    print("3. 刪除所有數據")
    print("4. 查詢相關數據")
    print("5. 退出系統")
    
    while True:
        command = input("輸入功能編號 (1-5): ").strip()
        if command == "5":
            print("退出系統。")
            break
        elif command == "1":
            view_all()
        elif command == "2":
            add_data()
        elif command == "3":
            delete_all()
        elif command == "4":
            query_data()
        else:
            print("未知指令！請輸入 1-5 的功能編號。")

if __name__ == "__main__":
    main()
