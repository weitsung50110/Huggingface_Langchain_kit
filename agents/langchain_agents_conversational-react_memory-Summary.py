from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryMemory

# 初始化 Ollama LLM
llm = Ollama(
    model="kenneth85/llama-3-taiwan:8b-instruct",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# 工具: 日期工具
def get_today_date(_):
    print(datetime.now().strftime("今天是 %Y 年 %m 月 %d 日，星期%a。"))
    return datetime.now().strftime("今天是 %Y 年 %m 月 %d 日，星期%a。")

# 定義工具
tools = [
    Tool(
        name="get today date", 
        func=get_today_date, 
        description="回答今天的日期和星期幾。"
    )
]

# 自定義 Prompt
custom_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""\
你是一個智慧型 AI 助手，能回答問題並記住上下文。
可以使用以下工具：
1. get today date: 提供今天的日期和星期幾。

當問題可以直接回答時，請直接回答。只有在必須使用工具時才使用它們。

以下是目前的對話摘要：
{chat_history}

使用者的最新問題是：{input}
請根據摘要和上下文回答問題，並僅在必要時使用工具。
"""
)

# 初始化記憶體（使用 ConversationSummaryMemory）
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",  # 使用對話型代理
    verbose=False,
    agent_prompt=custom_prompt,
    handle_parsing_errors=True,
    memory=memory
)

# 主程序
def main():
    print("歡迎使用智慧型 AI 助手！可以進行多輪對話，並記住上下文。")
    print("隨時輸入問題，例如：『今天是幾號？』，或『你剛剛問了什麼？』")
    print("如果想結束對話，請輸入 'bye', 'exit' 或 'quit'。\n")

    while True:
        user_query = input("請輸入您的問題：")
        if user_query.lower() in ["bye", "exit", "quit"]:
            print("感謝您的使用，再見！")
            break

        try:
            response = agent.invoke({"input": user_query})
            print("\nAI 助手的回答紀錄：")
            print(response)
        except Exception as e:
            print(f"抱歉，處理你的問題時出現了一些錯誤：{e}")

# 執行主程序
if __name__ == "__main__":
    main()
