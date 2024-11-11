from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain.text_splitter import CharacterTextSplitter

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 初始化Ollama模型
llm = Ollama(model='kenneth85/llama-3-taiwan:8b-instruct', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# 建立文件列表，每個文件包含一段文字內容
docs = [
    Document(page_content='曼德珍珠奶茶草：這種植物具有強大的魔法屬性，常用於恢復被石化的受害者。'),
    Document(page_content='山羊可愛蓮花石 ：是一種從山羊胃中取出的石頭，可以解百毒。在緊急情況下，它被認為是最有效的解毒劑。'),
    Document(page_content='日本小可愛佐籐鱗片：這些鱗片具有強大的治愈能力，常用於製作治療藥水，特別是用於治療深層傷口。'),
]

# 設定文本分割器，chunk_size是分割的大小，chunk_overlap是重疊的部分
text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
documents = text_splitter.split_documents(docs)  # 將文件分割成更小的部分

# 初始化嵌入模型
embeddings = OllamaEmbeddings(model="llama3")

# 使用FAISS建立向量資料庫
# vectordb = FAISS.from_documents(docs, embeddings)

db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="./knowledge-base"
)
# 將向量資料庫設為檢索器
retriever = db.as_retriever()

# 設定提示模板，將系統和使用者的提示組合
prompt = ChatPromptTemplate.from_messages([
    ('system', '根據以下上下文，用中文回答使用者的問題：:\n\n{context}'),
    ('user', '問題: {input}'),
])

# 創建文件鏈，將llm和提示模板結合
document_chain = create_stuff_documents_chain(llm, prompt)

# 創建檢索鏈，將檢索器和文件鏈結合
retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        'context': context
    })
    print(response['answer'])
    # context = response['context']
    print("-------------------")
    print(response)
    print("-------------------")
    print(response['context'])
    input_text = input('>>> ')

# https://myapollo.com.tw/blog/langchain-tutorial-retrieval/
# https://huggingface.co/learn/cookbook/zh-CN/advanced_rag
# https://chatgpt.com/share/e0f169d7-8620-4468-ba0a-581e7d9f5676
# https://medium.com/@jackcheang5/%E5%BB%BA%E6%A7%8B%E7%B0%A1%E6%98%93rag%E7%B3%BB%E7%B5%B1-ca4e593f3fed