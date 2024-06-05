from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import WebBaseLoader

# 使用 WebBaseLoader 從指定網址加載文檔
loader = WebBaseLoader("https://www.ollama.com/")
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 使用 RecursiveCharacterTextSplitter 將文檔分割成較小的塊
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# 初始化 Ollama 語言模型
llm = Ollama(model='llama3')
# 初始化 Ollama 嵌入模型
embeddings = OllamaEmbeddings()

# 使用 FAISS 創建向量數據庫並加載分割後的文檔
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 創建用於生成搜索查詢的提示模板
prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
# 創建帶有歷史上下文的檢索器
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

# 創建用於獲取答案的提示模板
prompt_get_answer = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('user', '{input}'),
])
# 創建文檔鏈以生成答案
document_chain = create_stuff_documents_chain(llm, prompt_get_answer)

# 結合檢索器和文檔鏈，創建檢索鏈
retrieval_chain_combine = create_retrieval_chain(retriever_chain, document_chain)

# 初始化聊天歷史記錄
chat_history = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        # 調用檢索鏈並獲取回答
        response = retrieval_chain_combine.invoke({
            'input': input_text,
            'chat_history': chat_history,
        })
        # 打印回答
        print(response['answer'])
        # print(response)
        # 將用戶輸入和 AI 回答添加到聊天歷史中
        chat_history.append(HumanMessage(content=input_text))
        chat_history.append(AIMessage(content=response['answer']))

    input_text = input('>>> ')

# https://www.linkedin.com/pulse/beginners-guide-conversational-retrieval-chain-using-langchain-pxhjc
