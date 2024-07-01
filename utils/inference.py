from utils.llm import LLM
from utils.build_rag import RAG
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from transformers import T5Tokenizer, T5ForConditionalGeneration


def predict_rag(qns:str,history=None)->str:
    llm = LLM().get_llm_hf()
    retriever = RAG().get_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(qns)
    return result


def predict_rag_cot(qns:str,history=None)->str:
    llm = LLM().get_llm_hf()
    # llm = LLM().get_llm()
    retriever = RAG().get_retriever()
   # Prefix of the prompt
    prefix = "You are a helpful chatbot and answer questions based on provided context only. If the answer to the question is not there in the context, you can politely say that you do not have the answer"

    # Examples of chain of thought to be included in the prompt
    EXAMPLES=["""Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be based on {context}
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    # Final Answer: the final answer to the ORIGINAL input question"""]

    # Template to be used
    example_template ="""
    Context: {context}
    User: {query}
    AI: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables = ["context","query","answer"],
        template = example_template
    )

    suffix = """
    Context: {context}
    User: {query}
    AI:
    """

    CHAT_PROMPT = PromptTemplate.from_examples(
        examples = EXAMPLES, suffix=suffix, input_variables=["context","query"],
        prefix=prefix,verbose=True
    )
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | CHAT_PROMPT
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(qns)
    return result
