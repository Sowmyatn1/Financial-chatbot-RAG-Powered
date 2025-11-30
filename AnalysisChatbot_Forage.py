from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from google.colab import drive

     

os.environ["OPENAI_API_KEY"] = "Enter Key here"
drive.mount('/content/drive')
     
def extract_htmlcode(path):
  with open(path,"r", encoding="utf-8") as file:
    soup =BeautifulSoup(file,"html.parser")
    return soup.get_text()
     

document=extract_htmlcode('/content/drive/My Drive/GenAI10K/10KAnalysis.html')

#create embeddings
Embeddings = OpenAIEmbeddings()
#we are using FIASS vector store here
vector_store = FAISS.from_texts([document], Embeddings)
#create retriver.you will be using this retriver in line number 82 while you create_history_aware_retriever
retriever=vector_store.as_retriever()  
#create instance of LLM
llm=ChatOpenAI( model_name="gpt-3.5-turbo",temperature=0)

#-------------------------------------------------------------
# this code is using conversation buffer which will just store the previous queationa and pass it LLM when a new question is asked it will append the chat history and send it to LLM.
#disadvantage is If the chat history grows too long, the context may exceed token limits.
# Alternatively u can use create_history_aware_retriever  : This it will Reframes the current query by incorporating chat history before retrieving relevant documents.

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#rag_chain = ConversationalRetrievalChain.from_llm(
    #llm, retriever=retriever, memory=memory
#)
#query = "what is the revenue of Apple in 2022."
#response = rag_chain.run(query)
#print(response)
#-------------------------------------------------------------



#Try the same with create_history_aware_retriever it Provides instructions to the model on how to rephrase the query based on past interactions.
# here we are doing the query rewriting technique . where we are using an LLM to re write the query for us(wait we havent retrived any documents yet from vector DB. we are rewriting the query
#that we are going to send to the vector DB to fecth the docs) . we are saying LLM using the instruction provided int the contextualize_q_system_prompt and rewite the query for me

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

#("human", "{input}") ->Represents the current user query that needs to be rephrased.

# below lines create the prompt properly that can be understood by LLM its atemplete to fill in our instructions for the prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# below code does multiple thing . it first rewrites the query uignt he LLM and the using the retriever it will fetch the relavanet docsuments
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# now retrrival is done .lets move on to generation phase of the RAG
#below code we are generating human understandalbe answers for the user question with the documents we retrived
#for that first lets prepare the prompt

system_prompt = (
    "You are an AI assistant specializing in financial insights. "
    "Explain financial concepts simply and concisely. "
    "When relevant, describe how a chart or graph could help visualize trends "
    "or comparisons (e.g., a line graph for revenue trends, a bar chart for assets vs. liabilities). "
    "\n\n"
    "{context}"
)

# put the prompt in a template to make sure LLM to understand it 
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
     
#Below code Generates the answer using:
#1. The retrieved documents (from the retriever)
#2. The user query (or rewritten query)

# below line is Question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#below line is full rag chain(chain here means not maintaining any chain of thoughts r anything it a name representing its works like a pipe line of retrival->generatior)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
     

store = {}
     

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
     

def chatbot(input_text, session_id="default"):
    # Run the input through the RAG pipeline
    #Calls conversational_rag_chain.invoke() with session ID to maintain history.
    response = conversational_rag_chain.invoke(
    {"input":input_text },
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)


    answer=response["answer"]
    follow_up_suggestions = [
        "Would you like to compare this year's data with previous years?",
        "Do you want insights on a specific company's financial health?",
        "Would you like a breakdown of key financial metrics?"
    ]

    # Format response with a follow-up question
    final_response = f"{answer}\n\n💡 {follow_up_suggestions[0]}"
    return final_response

     

print(chatbot("How did Tesla’s revenue change over the last three years?", session_id="user1"))

     
