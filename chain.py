from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from utils.citations import format_docs_with_citations

def get_conversational_chain(retriever):
    """
    Builds an LCEL chain for conversational retrieval.
    
    Args:
        retriever: The LangChain retriever object.
        
    Returns:
        Runnable: The LCEL chain ready to be invoked.
    """
    # Use Gemini 1.5 Pro for chat generation
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    # 1. Prompt to contextualize the user's question based on history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # 2. Prompt for answering the question using retrieved context
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer concise but thorough enough to be helpful.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # 3. Build the full chain
    def get_contextualized_question(input_dict):
        if input_dict.get("chat_history"):
            return contextualize_q_chain.invoke({
                "chat_history": input_dict["chat_history"],
                "question": input_dict["question"]
            })
        return input_dict["question"]

    # Retrieve docs
    retrieval = RunnablePassthrough.assign(
        docs=lambda x: retriever.invoke(get_contextualized_question(x))
    )
    
    # Format docs for the prompt
    format_docs = RunnablePassthrough.assign(
        context=lambda x: format_docs_with_citations(x["docs"])
    )
    
    # Answer generation
    answer_chain = (
        qa_prompt
        | llm
        | StrOutputParser()
    )
    
    # Final chain returns a dict with 'docs' and 'answer'
    conversational_rag_chain = retrieval | format_docs | RunnablePassthrough.assign(answer=answer_chain)
    
    return conversational_rag_chain
