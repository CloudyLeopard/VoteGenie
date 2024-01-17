import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_core.vectorstores import VectorStore
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
# from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback

# retrievers
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel


import logging
import uuid

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
            self.output_parser = JsonOutputParser(pydantic_object=self.ResponseWithReasoningAndEvidence)
            format_instruction = self.output_parser.get_format_instructions()
        else:
            format_instruction = ""
        self.prompt = self.prompt_generator(politician_name, format_instruction)


        # the magic starts here
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(model_name=model_name, temperature=0)
        llm = lambda x: json.dumps({"fake": "llm"})
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["source_documents"])))
            | self.prompt
            | llm
            | self.output_parser
        )

        # chain to return context
        self.rag_chain = RunnableParallel(
            {"source_documents": self.retriever, "question": RunnablePassthrough()}
        ).assign(result=rag_chain_from_docs)
        # Note: figure how out runnable works and add a chain to parse source_documents here
        
        # logic: rag_chain.invoke(xxx)
        # --> xxx gets passed into self.retriever and RunnablePassthrough()
        # returns {"context": retriever(xxx), "question": xxx}
        # --> this is input into rag_chain_from_docs, and the output of that is appended
        # to the dict under the key "result"

        self.logger = logging.getLogger('genie')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('./api_calls.log')
        self.logger.addHandler(handler)
        self.logger.propagate = False

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
    
    def _parse_context(source_docs: dict):
        for i in range(len(source_docs)):
            source_docs[i] = {
                "source_content": source_docs[i].page_content,
                "source_category": source_docs[i].metadata.get(
                    "parent_question", ""
                ),
                "source_sub_category": source_docs[i].metadata.get("question", ""),
            }
        return source_docs

    def get_relevant_documents(self, question: str):
        return self.retriever.get_relevant_documents(question)

    def ask(self, question: str):
        with get_openai_callback() as cb:
            result = self.rag_chain.invoke(question)

            result["total_tokens"] = cb.total_tokens
            result["total_cost"] = cb.total_cost
            result["source_documents"] = self._parse_context(result["source_documents"])

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
        return

    async def async_ask(self, question: str):
        with get_openai_callback() as cb:
            result = await self.genie.acall({"query": question})
            result = self._parse_result(result, cb.total_tokens, cb.total_cost)

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
    
    # parser classes
    class BaseResponse(BaseModel):
        answer: str = Field(
            description="The response to the question. Accepted values are: yes, no, unknown"
        )

    class ReasponseWithReasoning(BaseResponse):
        reasoning: str = Field(description="The reasoning behind the response")
    
    class ResponseWithReasoningAndEvidence(ReasponseWithReasoning):
        evidence: List[str] = Field(description="List the quotes from the context that supports your answer")

