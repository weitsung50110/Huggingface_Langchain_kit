from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 初始化語言模型和嵌入模型
llm = Ollama(model='llama3')
embeddings = OllamaEmbeddings()

# 建立一個向量存儲，從文本生成嵌入
vector = FAISS.from_texts(['My name is Weiberson, I\'m 25\'years old. '], embeddings)
# 將向量存儲轉換為檢索器
retriever = vector.as_retriever()

# 建立生成搜尋查詢的提示模板
prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
# 建立帶有歷史紀錄感知的檢索器鏈
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

# 建立回答使用者問題的提示模板
prompt_get_answer = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('user', '{input}'),
])
# 建立文件處理鏈
document_chain = create_stuff_documents_chain(llm, prompt_get_answer)

# 結合檢索器鏈和文件處理鏈建立檢索鏈
retrieval_chain_combine = create_retrieval_chain(retriever_chain, document_chain)

# 初始化聊天歷史紀錄
chat_history = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        # 使用檢索鏈處理輸入，並生成回應
        response = retrieval_chain_combine.invoke({
            'input': input_text,
            'chat_history': chat_history,
        })
        # 輸出回應
        print(response['answer'])
        # 更新聊天歷史紀錄
        chat_history.append(HumanMessage(content=input_text))
        chat_history.append(AIMessage(content=response['answer']))
        print("--------------------------")
        print(chat_history)
    input_text = input('>>> ')


# https://www.linkedin.com/pulse/beginners-guide-conversational-retrieval-chain-using-langchain-pxhjc