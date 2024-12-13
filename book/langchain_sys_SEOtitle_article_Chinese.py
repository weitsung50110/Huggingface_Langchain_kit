# 用LangChain使mistral:7b-instruct-q2_K藉由對話生成文章和SEO標題，把語言模型變成愛情作家之教學

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 如果加上了CallbackManager就可以即時看到llm生成的文字，
# 若沒加CallbackManager則是要等到llm把文字全部生成完成後 才會顯示出來

llm = Ollama(model='kenneth85/llama-3-taiwan:8b-instruct-dpo-q4_K_M', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位專業的愛情故事作家，擅長創作感人的短篇故事，並且熟悉 SEO 技術。你的任務是：以繁體中文撰寫一篇約 50 字的浪漫愛情故事，並創建一個吸引人且包含關鍵字的 SEO 標題。輸出格式為：\n1. 故事內容：\n2. SEO 標題："),
    ("user", "{input}"),
])




# print(prompt)

chain = prompt | llm
chain.invoke({"input": input('>>> ')})

#輸入文字後，會依據文字生成SEO標題 和 文章內容