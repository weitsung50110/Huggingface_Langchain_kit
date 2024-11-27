#這個只能各別問日期和數學，如果一起問會失去記憶
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 初始化 Ollama LLM
llm = Ollama(model="kenneth85/llama-3-taiwan:8b-instruct", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# 工具: 日期工具
def get_today_date(_):
    """
    返回今天的日期和星期幾
    """
    return datetime.now().strftime("今天是 %Y 年 %m 月 %d 日，星期%a。")

# 工具: 數學計算工具
def calculate_math(expression):
    """
    計算數學表達式，添加預處理和檢查
    """
    try:
        # 清理輸入表達式：去掉單引號和多餘空白
        expression = expression.strip().strip("'").strip('"')
        
        # 檢查表達式是否包含非法字符
        allowed_chars = "0123456789+-*/(). "  # 允許的字符
        if not all(char in allowed_chars for char in expression):
            return "數學表達式包含無效字符，請檢查輸入。"
        
        # 計算表達式
        result = eval(expression)
        return f"計算結果是: {result}"
    except Exception as e:
        return f"計算過程中發生錯誤: {e}"



# 定義工具
get_today_date_tool = Tool(
    name="get today date(_)",
    func=get_today_date,
    description="回答今天的日期和星期幾。"
)

math_tool = Tool(
    name="Python Calculator",
    func=calculate_math,
    description="可以計算數學表達式的工具。輸入應該是有效的 Python 數學表達式，例如 '2+2' 或 'math.sqrt(16)'。"
)

tools = [
    get_today_date_tool,  # 日期工具
    math_tool             # 數學計算工具
]

# 自定義 Prompt
custom_prompt = PromptTemplate(
    input_variables=["input"],
    template="""\
你是一個智慧型 AI 助手，可以使用以下工具來完成任務：
- get today date(_): 回答今天是幾月幾號以及星期幾。
- Python Calculator: 計算數學表達式的工具。

請根據使用者的問題，選擇最適合的工具並提供準確、簡潔的回答。

使用者問題：{input}
"""
)

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    agent_prompt=custom_prompt,
    handle_parsing_errors=True  # 啟用解析錯誤處理
)

# 主程序
if __name__ == "__main__":
    try:
        while True:
            user_query = input("請輸入您的問題：")
            if user_query.lower() in ["bye", "exit", "quit"]:
                print("感謝您的使用，再見！")
                break

            # 直接執行並返回結果
            response = agent.run({"input": user_query})
            print("\nAI 助手的回答：")
            print(response)
    except Exception as e:
        print(f"執行時發生錯誤：{e}")
