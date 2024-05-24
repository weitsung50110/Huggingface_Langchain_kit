from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


llm = Ollama(model='llama3')
embeddings = OllamaEmbeddings()

vector = FAISS.from_texts(['My name is Amo.'], embeddings)
retriever = vector.as_retriever()

prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

prompt_get_answer = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('user', '{input}'),
])
document_chain = create_stuff_documents_chain(llm, prompt_get_answer)

retrieval_chain_combine = create_retrieval_chain(retriever_chain, document_chain)

chat_history = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        response = retrieval_chain_combine.invoke({
            'input': input_text,
            'chat_history': chat_history,
        })
        print(response['answer'])
        chat_history.append(HumanMessage(content=input_text))
        chat_history.append(AIMessage(content=response['answer']))
        print("--------------------------")
        print(chat_history)
    input_text = input('>>> ')

# https://www.linkedin.com/pulse/beginners-guide-conversational-retrieval-chain-using-langchain-pxhjc