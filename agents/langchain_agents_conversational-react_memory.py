from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

# 初始化 Ollama LLM
llm = Ollama(model="kenneth85/llama-3-taiwan:8b-instruct", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# 工具: 日期工具
def get_today_date(_):
    print(datetime.now().strftime("今天是 %Y 年 %m 月 %d 日，星期%a。"))
    return datetime.now().strftime("今天是 %Y 年 %m 月 %d 日，星期%a。")

# 定義工具
tools = [
    Tool(
        name="get today date", 
        func=get_today_date, 
        description="回答今天的日期和星期幾。")
]

# 自定義 Prompt
custom_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""\
你是一個智慧型 AI 助手，能回答問題並記住上下文。
可以使用以下工具：
1. get today date: 提供今天的日期和星期幾。

當問題可以直接回答時，請直接回答。只有在必須使用工具時才使用它們。

以下是目前的對話記錄：
{chat_history}

使用者的最新問題是：{input}
請根據上下文回答問題，並僅在必要時使用工具。
"""
)

# 初始化記憶體
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

# conversational-react-description
# 設計目標：適用於需要上下文的多輪對話。

# 特點：
# 上下文支持：
# 支持記憶功能（如 ConversationBufferMemory），能記住之前的對話記錄，適合多輪交互。
# 行為模式：
# 代理會嘗試根據上下文回答問題，只有當問題需要工具時才使用工具。
# 在多輪對話中，會自動參考對話記憶，生成更連貫的回答。
# 適用場景：
# 用於聊天機器人、虛擬助理或需要持續上下文的應用。


# zero-shot-react-description
# 設計目標：適合一次性問題的快速回答。

# 特點：
# 無上下文記憶：
# 代理不會參考過去的對話記錄，即使有記憶功能，也僅在工具使用中保留臨時上下文。
# 行為模式：
# 嘗試對每個問題做一次性的推理（Zero-Shot），並直接產生回答或選擇工具。
# 適用場景：
# 適合單次操作或無需記住上下文的問題（如數學計算、數據查詢）。


# 如何選擇適合的代理？
# 如果需要多輪對話和記憶功能：conversational-react-description。
# 如果需要快速單次回答：zero-shot-react-description。
# 如果需要精確工具映射：react-description。
# 如果需要實時搜索：self-ask-with-search。
# 如果使用 OpenAI 插件：openai-functions。
