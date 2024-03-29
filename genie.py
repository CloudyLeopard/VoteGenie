import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_core.vectorstores import VectorStore
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import PydanticOutputParser

# from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback

# retrievers
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)

from langchain.chains import LLMChain

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_relevancy,
    context_recall,
    AnswerSimilarity,
    AnswerCorrectness
)
from ragas import evaluate

# type declaration
from langchain_core.documents.base import Document
from pydantic import BaseModel, Field
from typing import List

import logging
import uuid

import json
from datetime import datetime
from pytz import timezone

from datasets import Dataset


class Genie:
    def __init__(
        self,
        politician_name: str,
        vectorstore: VectorStore,
        model_name: str = "gpt-3.5-turbo-1106",
        use_parser=True,
        debug=False
    ):
        self.politician_name = politician_name
        self.model_name = self.validate_model_name(model_name)
        self.vectorstore = vectorstore
        self.retriever = self._multiquery_retriever(self.vectorstore, politician_name)

        if use_parser:
            output_parser = JsonOutputParser(
                pydantic_object=ResponseWithReasoningAndEvidence
            )
            format_instruction = output_parser.get_format_instructions()
        else:
            output_parser = StrOutputParser()
            format_instruction = ""

        self.prompt_template = self.prompt_generator(politician_name, format_instruction)

        # the magic starts here
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def print_prompt(prompt):
            if debug:
                print(prompt.to_messages())
            return prompt

        llm = ChatOpenAI(model_name=model_name, temperature=0)
        # llm = RunnableLambda(lambda x: json.dumps({"fake": "llm"})) # for testing

        llm_generate = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | self.prompt_template
            | RunnableLambda(print_prompt)
            | llm
            | output_parser
        )

        self.rag_chain_base = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(result=llm_generate)

        self.rag_chain = (
            self.rag_chain_base
            | RunnablePassthrough.assign(
                source_documents=(lambda x: self._parse_documents(x["context"]))
            )
            | RunnableLambda(
                lambda output: [output.pop("context"), output][1]
            )  # drop the key "context"
        )

        # Note: figure how out runnable works and add a chain to parse source_documents here

        # logic: rag_chain.invoke(xxx)
        # --> xxx gets passed into self.retriever and RunnablePassthrough()
        # returns {"context": retriever(xxx), "question": xxx}
        # --> this is input into rag_chain_from_docs, and the output of that is appended
        #   --> the RunnablePassthrough creates a {"context"=format_docs(retriever(xxx)),
        #       "source_documents":"retriver(xxx)", "question'":xxx}
        #   --> passthrough to prompt, a chat prompt template that needs "context" and "question" as variables
        #       returns the entire prompt which is passed into llm, etc.
        #   --> append result to the {"context": ..., "question": ...} dict under the key "result"
        # --> parse the "context" item (which is a list of Document objects), store under "source_documents"
        # --> remove "context" key

        self.logger = logging.getLogger("genie")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler("./api_calls.log")
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def __str__(self):
        return f"<Genie name={self.politician_name} model={self.model_name}>"

    @staticmethod
    def validate_model_name(name):
        if name == "gpt-4" or name == "gpt-4-1106-preview":
            return name
        return "gpt-3.5-turbo-1106"

    @staticmethod
    def generate_random_id():
        random_uuid = uuid.uuid4()
        return str(random_uuid)[:8]

    @staticmethod
    def prompt_generator(name, output_format_instruction: str = ""):
        # template = "You determine a person's response to a given question based on their statements"
        template = '''You will be provided with different pieces of context delimited by triple quotes and a question. \
The context are either statements made by the person of interest, or statements describing the person of interest. \
Your task is to answer the question using only the provided context, then support the answer with evidence and reasoning. \
If the document does not contain the information needed to answer this question, simply write “unknown”.
'''
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        prompt_template = '''
"""
{context}
"""

Question: For {name}, {question}

{format_instructions}
'''
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "context",
                "question",
            ],  # i'm guessing retriever outputs these
            partial_variables={
                "format_instructions": output_format_instruction,
                "name": name,
            },
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        return chat_prompt

    @staticmethod
    def _retriever(vectorstore: VectorStore, name: str, k: int = 4):
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"name": name}, "k": 4},
        )

    @staticmethod
    def _multiquery_retriever(vectorstore: VectorStore, name: str, model_name="gpt-3.5-turbo-1106"):

        # logging.basicConfig()
        # logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        base_retriever = Genie._retriever(vectorstore, name)
        llm = ChatOpenAI(model=model_name, temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )
        return retriever_from_llm

    @staticmethod
    def _contextual_compression_retriever(vectorstore: VectorStore, name: str):
        base_retriever = Genie._retriever(vectorstore, name)
        embeddings = OpenAIEmbeddings()

        splitter = CharacterTextSplitter(
            chunk_size=300, chunk_overlap=0, separator=". "
        )
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(
            embeddings=embeddings, similarity_threshold=0.76
        )
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return compression_retriever
    
    @staticmethod
    def _hyde_retriever(vectorstore: VectorStore, name: str, model_name="gpt-3.5-turbo-1106"):
        # Output parser will split the LLM result into a list of queries

        prompt = ChatPromptTemplate.from_messages(
            [("human", """What are all the possible answers to the following question? \
Separate the answer by new lines. The answers should be two sentences long. 

Question: {question}

Here's an example with the correct format for the answer
Question: Should the government ban women's access to abortion?

It is a woman's right to make decisions about her own body and health.
The government should ban abortion as some view it as the taking of an innocent life and believe the state has a responsibility to protect that life.
Access to safe abortions is a matter of public health and human rights.
A ban on abortion could disproportionately affect those from lower socioeconomic backgrounds who may not have the means to access safe alternatives. Health care decisions should be private and accessible to all women.
Life begins at conception, and the government has a duty to preserve it
"""),
("system", "You are an expert on politics")]
)


        llm = ChatOpenAI(temperature=0, model=model_name)

        parser = LineListOutputParser()
        llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

        base_retriever = Genie._retriever(vectorstore, name)
        hyde_retriever = MultiQueryRetriever(
            retriever=base_retriever, llm_chain=llm_chain, parser="lines"
        )
        return hyde_retriever
        

    @staticmethod
    def _parse_documents(source_docs: List[Document]):
        for i in range(len(source_docs)):
            source_docs[i] = {
                "source_content": source_docs[i].page_content,
                "source_category": source_docs[i].metadata.get("parent_question", ""),
                "source_sub_category": source_docs[i].metadata.get("question", ""),
            }
            # source_docs[i] = source_docs[i].dict() # langchain has a method to parse document to dict
        return source_docs

    def _get_rag_chain(self):
        return self.rag_chain
    
    def _get_base_rag_chain(self):
        return self.rag_chain_base


    def get_prompt_template(self):
        return self.prompt_template
    
    def get_relevant_documents(self, question: str):
        return self.retriever.get_relevant_documents(question)
    
    def base_ask(self, question: str):
        return self.rag_chain_base.invoke(question)
    
    def base_batch_ask(self, questions: List[str]):
        return self.rag_chain_base.batch(questions)
    
    async def async_base_batch_ask(self, questions: List[str]):
        return await self.rag_chain_base.abatch(questions)

    def ask(self, question: str):
        with get_openai_callback() as cb:
            result = self.rag_chain.invoke(question)

            result["total_tokens"] = cb.total_tokens
            result["total_cost"] = cb.total_cost

            # logging
            # Note: move log into the chain later
            tz = timezone("US/Eastern")
            self.logger.info(
                "Genie Response:id=%s name=%s model=%s time=%s response=%s",
                id,
                self.politician_name,
                self.model_name,
                datetime.now(tz),
                json.dumps(result),
            )
        return result

    def batch_ask(self, questions: List[str]):
        with get_openai_callback() as cb:
            predictions = self.rag_chain.batch(questions)

            results = {
                "results": predictions,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost
            }

        return results

    async def async_ask(self, question: str):
        with get_openai_callback() as cb:
            result = await self.rag_chain.ainvoke(question)
            result["total_tokens"] = cb.total_tokens
            result["total_cost"] = cb.total_cost

            # logging
            tz = timezone("US/Eastern")
            self.logger.info(
                "Async Genie Response:id=%s name=%s model=%s time=%s response=%s",
                id,
                self.politician_name,
                self.model_name,
                datetime.now(tz),
                json.dumps(result),
            )
        return result
    
    async def async_batch_ask(self, questions: List[str]):
        with get_openai_callback() as cb:
            predictions = await self.rag_chain.abatch(questions)

            results = {
                "results": predictions,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost
            }

        return results
    
    def evaluate(self, questions: List[str], ground_truths: List[List[str]] | None = None):
        print("Getting LLM response...")
        results = self.base_batch_ask(questions)

        answers = [json.dumps(entry['result']) for entry in results]
        contexts = [[doc.page_content for doc in entry['context']] for entry in results]

        data_samples = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,            
        }
        if ground_truths:
            data_samples['ground_truths'] = ground_truths
        
        dataset = Dataset.from_dict(data_samples)

        metrics = [
            faithfulness,
            answer_relevancy,
            context_relevancy,
        ]
        if ground_truths:
            metrics.extend([
                context_recall,
                AnswerCorrectness(weights=[0.4,0.6])
            ]) 

        result = evaluate(dataset, metrics = metrics)

        return result


# parser classes
class BaseResponse(BaseModel):
    answer: str = Field(
        description="The response to the question. Accepted values are: yes, no, unknown"
    )

class ResponseWithReasoning(BaseResponse):
    reasoning: str = Field(description="The reasoning behind the response")

class ResponseWithReasoningAndEvidence(ResponseWithReasoning):
    evidence: List[str] = Field(
        description="List all segments of the context that supports the answer. Leave the list empty if answer is unknown"
    )

class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)