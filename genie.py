import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain.vectorstores.base import VectorStore
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback

# retrievers
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter

import logging

logging.basicConfig(filename="./api_calls.log", level=logging.INFO)
import json
from datetime import datetime
from pytz import timezone


class Genie:
    def __init__(
        self,
        politician_name: str,
        vectorstore: VectorStore,
        model_name: str = "gpt-3.5-turbo",
        use_parser = True
    ):
        self.politician_name = politician_name
        self.model_name = self.validate_model_name(model_name)
        self.vectorstore = vectorstore
        self.retriever = self._retriever(self.vectorstore, politician_name)
        
        self.output_parser = None
        if use_parser:
            self.output_parser = PydanticOutputParser(pydantic_object=self.ResponseWithReasoningAndEvidence)
            format_instruction = self.output_parser.get_format_instructions()
        else:
            format_instruction = ""
        self.prompt = self.prompt_generator(politician_name, format_instruction)

        self.genie = RetrievalQA.from_chain_type(
            ChatOpenAI(model_name=model_name, temperature=0),
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )
    @staticmethod
    def validate_model_name(name):
        if name == "gpt-4":
            return name
        return "gpt-3.5-turbo"

    @staticmethod
    def prompt_generator(name, output_format_instruction: str = ""):
        # template = "You determine a person's response to a given question based on their statements"
        template = "You determine how a person will respond to a given question based on that person's quotes"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        prompt_template = """Use the following pieces of context to answer the question at the end, and support it with evidence. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{format_instructions}

{context}

Question: For {name}, {question}
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "context",
                "question",
            ],  # i'm guessing retriever outputs these
            partial_variables={"format_instructions": output_format_instruction, "name": name},
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
    def _multiquery_retriever(vectorstore: VectorStore, name: str):
        import logging

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        base_retriever = Genie._retriever(vectorstore, name)
        llm = ChatOpenAI(temperature=0)
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

    def get_relevant_documents(self, question: str):
        return self.retriever.get_relevant_documents(question)

    def ask(self, question: str):
        with get_openai_callback() as cb:
            result = self.genie({"query": question})
            # parse result into desired format (e.g. dict) if a parser is passed in
            if self.output_parser:
                result["result"] = self.output_parser.parse(result["result"]).dict()
            source_docs = result["source_documents"]
            for i in range(len(source_docs)):
                source_docs[i] = {
                    "source_content": source_docs[i].page_content,
                    "source_category": source_docs[i].metadata.get(
                        "parent_question", ""
                    ),
                    "source_sub_category": source_docs[i].metadata.get("question", ""),
                }
            result["total_tokens"] = cb.total_tokens
            result["total_cost"] = cb.total_cost
            tz = timezone("US/Eastern")
            logging.info(
                "Genie Response:name=%s model=%s time=%s response=%s",
                self.politician_name,
                self.model_name,
                datetime.now(tz),
                json.dumps(result),
            )
        return result

    async def async_ask(self, question: str):
        with get_openai_callback() as cb:
            result = await self.genie.acall({"query": question})
            # parse result into desired format (e.g. dict) if a parser is passed in
            if self.output_parser:
                result["result"] = self.output_parser.parse(result["result"]).dict()
            source_docs = result["source_documents"]
            for i in range(len(source_docs)):
                source_docs[i] = {
                    "source_content": source_docs[i].page_content,
                    "source_category": source_docs[i].metadata.get(
                        "parent_question", ""
                    ),
                    "source_sub_category": source_docs[i].metadata.get("question", ""),
                }
            result["total_tokens"] = cb.total_tokens
            result["total_cost"] = cb.total_cost
            tz = timezone("US/Eastern")
            logging.info(
                "Async Genie Response:name=%s model=%s time=%s response=%s",
                self.politician_name,
                self.model_name,
                datetime.now(tz),
                json.dumps(result),
            )
        return result
    
    # parser classes
    class BaseResponse(BaseModel):
        answer: str = Field(
            description="The response to the question. Accepted values are: yes, no, unknown"
        )

    class ReasponseWithReasoning(BaseResponse):
        reasoning: str = Field(description="The reasoning behind the response")
    
    class ResponseWithReasoningAndEvidence(ReasponseWithReasoning):
        evidence: List[str] = Field(description="List the quotes from the context that supports your answer")

