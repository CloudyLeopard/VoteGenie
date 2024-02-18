from typing import List

from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from langchain.retrievers.multi_query import MultiQueryRetriever
from genie_master import GenieMaster

genie_db_path = "./chroma_qadata_db/"
gm = GenieMaster(db_path=genie_db_path)


import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


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
("system", "You are a helpful assistant")]
)


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

parser = LineListOutputParser()
llm_chain = prompt | llm | parser

print(llm_chain.invoke({"question":"Should the U.S. increase restrictions on its current border security policy?"}))

retriever = MultiQueryRetriever(
    retriever=gm.vectorstore.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)

retriever.get_relevant_documents("Should a photo ID be required to vote?")