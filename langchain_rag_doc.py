from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain.text_splitter import CharacterTextSplitter


llm = Ollama(model='llama3')

docs = [
    Document(page_content='變身水（Polyjuice Potion）可變成其他人的樣貌。不可拿來變身成動物，也對動物產生不了效果（包括半人半動物的生物），誤用動物毛髮的話，則會變成動物的容貌。'),
    Document(page_content='吐真劑（Veritaserum）出自《火盃的考驗》，特徵為像水一樣清澈無味，使用者只要加入三滴，就能強迫飲用者說出真相。它是現存最強大的吐實魔藥，在《哈利波特》的虛構世界觀中受英國魔法部嚴格控管。J·K·羅琳表示，吐真劑最適合用在毫無戒心、易受傷害、缺乏自保技能的人身上，有些巫師能使用鎖心術等方式保護自己免受吐真劑影響。'),
    Document(page_content='福來福喜（Felix Felicix）出自《混血王子》，是一種稀有而且難以調製的金色魔藥，能夠給予飲用者好運。魔藥的效果消失之前，飲用者的所有努力都會成功。假如飲用過量，會導致頭暈、魯莽和危險的過度自信，甚至成為劇毒。'),
]

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