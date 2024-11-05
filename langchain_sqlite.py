from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy.exc import OperationalError

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


llm = ChatOllama(model='kenneth85/llama-3-taiwan:8b-instruct', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


db = SQLDatabase.from_uri("sqlite:///./weibert.sqlite3")


def get_db_schema(_):
    return db.get_table_info()


def run_query(query):
    try:
        return db.run(query)
    except (OperationalError, Exception) as e:
        return str(e)


gen_sql_prompt = ChatPromptTemplate.from_messages([
    ('system', '根據下面的資料庫結構，編寫一個 SQL 查詢來回答使用者的問題：{db_schema}'),
    ('user', '請為以下問題生成一個 SQL 查詢："{input}"。\
     查詢應該格式化為以下方式，並且不附加任何額外的解釋：\
     SQL> <sql_query>\
    '),
])


class SqlQueryParser(StrOutputParser):
    def parse(self, s):
        r = s.split('SQL> ')
        if len(r) > 0:
            return r[1]
        return s


gen_query_chain = (
    RunnablePassthrough.assign(db_schema=get_db_schema)
    | gen_sql_prompt
    | llm
    | SqlQueryParser()
)
# print(gen_query_chain)


gen_answer_prompt = ChatPromptTemplate.from_template("""
根據提供的問題、SQL 查詢和查詢結果，撰寫一個自然語言的回答。
不應包含任何額外的解釋。


回答應該格式化為以下形式：
'''
問題: {input}
執行: {query}
查詢結果: {result}
回答: <answer>
'''
""")


chain = (
    RunnablePassthrough.assign(query=gen_query_chain).assign(
        result=lambda x: run_query(x["query"]),
    )
    | gen_answer_prompt
    | llm
)
# print(chain)
input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        response = chain.invoke({
            'input': input_text,
        })
        # print(response)
    input_text = input('>>> ')