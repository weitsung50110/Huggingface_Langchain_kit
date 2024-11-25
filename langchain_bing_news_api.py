from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests

# 初始化 LLM
llm = Ollama(
    model="kenneth85/llama-3-taiwan:8b-instruct",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
embeddings = OllamaEmbeddings(model="kenneth85/llama-3-taiwan:8b-instruct")

# 全局變量
search_results = ""

# 定義搜尋工具
def simple_search(query, api_key):
    try:
        endpoint = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {"q": query, "count": 5}
        
        response = requests.get(endpoint, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("webPages", {}).get("value", []):
            result = {
                "title": item.get("name"),
                "link": item.get("url"),
                "snippet": item.get("snippet", "無摘要")
            }
            results.append(result)
        
        global search_results
        search_results = "\n\n".join(
            f"標題: {res['title']}\n連結: {res['link']}\n摘要: {res['snippet']}"
            for res in results
        )
        return search_results
    except Exception as e:
        global search_results
        search_results = f"發生錯誤：{e}"
        return search_results

# 摘要工具
def summarize_content():
    global search_results
    if not search_results.strip():
        return "無內容可進行摘要。"

    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="請對以下內容進行重點摘要，今年是2024年：\n\n{content}"
    )
    prompt = prompt_template.format(content=search_results)
    response = llm.invoke(prompt)
    return response.strip()

# 主程式
if __name__ == "__main__":
    try:
        BING_API_KEY = "你的 Bing API 金鑰"
        user_query = input("請輸入您的問題（可用中文）：")
        
        print("\n正在執行搜尋...")
        simple_search(user_query, BING_API_KEY)
        print("\n搜尋結果：")
        print(search_results)
        
        print("\n正在進行摘要...")
        summary = summarize_content()
        print("\n摘要結果：")
        print(summary)
    except Exception as e:
        print(f"執行時發生錯誤：{e}")
