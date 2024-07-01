
import os,time
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
# from langchain_together import Together
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from langchain import HuggingFacePipeline
load_dotenv()

class LLM:
    def __init__(self) -> None:
        self.local_model_path = os.getenv('MODEL_PATH')

    def get_llm(self) -> CTransformers:
        
        start = time.time()

        llm = CTransformers(model=self.local_model_path,
                            config={'max_new_tokens': 4096,
                                'temperature': 0.00,
                                'context_length': 4096})
        end = time.time()
        print('Time to load the model:',end-start)
        return llm

    def get_llm_hf(self) :

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="mps")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        llm = HuggingFacePipeline(
            pipeline = pipe,
            model_kwargs={"temperature": 0, "max_length": 512},
        )
        return llm


