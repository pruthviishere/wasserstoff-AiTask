# Example to show chain of thought
import os

from dotenv import load_dotenv
from langchain import PromptTemplate, FewShotPromptTemplate, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

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
Final Answer: the final answer to the original input question"""]

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
    examples = EXAMPLES, suffix=suffix, input_variables=["context","query"],prefix=prefix
)

# query = "I want to buy stocks of Google. Can I buy through your bank"
# context = "Bank customers will not be able to trade in shares and mutual funds through their bank account. " \
#           "They will need to open a trading account for trading in the market"

#print(CHAT_PROMPT.format(query=query,context=context))

#query = "I want to buy stocks of Google. Can I buy through your bank"
#query = "Is the bank open on 25th December"
# context = "Langchain in a python based llm framework. It was created in 2023 by Harrison chase"
#print(TEXTWORLD_PROMPT.format(query=query,context=context))

 

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForCausalLM.from_pretrained("google/flan-t5-small")

chain = load_qa_chain(llm,chain_type="stuff",prompt=CHAT_PROMPT,verbose=False)
docs = [Document(page_content="Bank customers will not be able to trade in shares and mutual funds through their bank "
                              "account. They will need to open and trading account for trading in the market",
                 metadata={}),
        Document(page_content="Bank customers can open trading account by logging into the banks portal",
                 metadata={})
        ]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
   
embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")
    
persist_directory ="db"
CHROMA_SETTINGS = Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False
)
chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, 
            client_settings=CHROMA_SETTINGS, client=chroma_client)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

qa_chain = RetrievalQA.from_chain_type(   
llm=llm,   
chain_type="stuff",   
retriever=retriever ,
chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
callbacks=callbacks,
    return_source_documents= not args.hide_source

while True:
    query = input("What is your question:\n")
    response = chain.run(input_documents=docs,query=query)
    print(response)