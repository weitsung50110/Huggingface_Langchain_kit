from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 初始化Ollama模型
llm = Ollama(model='wangshenzhi/llama3-8b-chinese-chat-ollama-q8', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# 載入並分割PDF文件
loader = PyPDFLoader("pdf_test.pdf")
docs = loader.load_and_split()

# 設定文本分割器，chunk_size是分割的大小，chunk_overlap是重疊的部分
text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
documents = text_splitter.split_documents(docs)

# 初始化嵌入模型
embeddings = OllamaEmbeddings(model="wangshenzhi/llama3-8b-chinese-chat-ollama-q8")

# 使用FAISS建立向量資料庫
vectordb = FAISS.from_documents(documents, embeddings)
# 將向量資料庫設為檢索器
retriever = vectordb.as_retriever()

# 設定提示模板，將系統和使用者的提示組合
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
    ('user', 'Question: {input}'),
])

# 創建文件鏈，將llm和提示模板結合
document_chain = create_stuff_documents_chain(llm, prompt)

# 創建檢索鏈，將檢索器和文件鏈結合
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        # 'context': context
    })
    # print(response['answer'])
    # context = response['context']

    input_text = input('>>> ')

