from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


from langchain.text_splitter import CharacterTextSplitter


llm = Ollama(model='llama3')

loader = PyPDFLoader("weibert.pdf")
docs = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5) #通過設置塊重疊部分，我們可以確保每個分割後的文本塊仍然包含足夠的上下文信息
documents = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings()

vectordb = FAISS.from_documents(docs, embeddings)
retriever = vectordb.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
    ('user', 'Question: {input}'),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        'context': context
    })
    print(response['answer'])
    context = response['context']
    input_text = input('>>> ')

# https://myapollo.com.tw/blog/langchain-tutorial-retrieval/
# https://huggingface.co/learn/cookbook/zh-CN/advanced_rag
# https://chatgpt.com/share/e0f169d7-8620-4468-ba0a-581e7d9f5676
# https://medium.com/@jackcheang5/%E5%BB%BA%E6%A7%8B%E7%B0%A1%E6%98%93rag%E7%B3%BB%E7%B5%B1-ca4e593f3fed