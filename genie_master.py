from genie import Genie
import uuid
import pandas as pd

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb.config import Settings
from datetime import datetime
from pytz import timezone

# type declaration
from typing import List


class GenieMaster:
    _EMBEDDING_FUNCTION = OpenAIEmbeddings()

    def __init__(self, db_path="./chroma_db", collection_name="langchain"):
        tz = timezone("US/Eastern")
        self.init_time = datetime.now(tz)
        print(f"Genie Master initialized at: {self.init_time}")

        self.db_path = db_path

        # init chromadb vector storage
        # https://docs.trychroma.com/usage-guide
        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )
        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self._EMBEDDING_FUNCTION,
        )

    def _document_count(self):
        return self.get_collection().count()

    def model_is_ready(self):
        return self._document_count() > 0

    def get_collection(self):
        return self.vectorstore._collection

    def transform_and_add_data(self, df: pd.DataFrame, page_content_column):
        """Transforms the data in the inputted database into vector and stored
        in a vector db. Note: If data being added is already in the database (word for word),
        then this entry will be skipped.
        """
        if not "name" in df.columns:
            raise Exception("Dataframe must have a name column")
        if df[df.isna().any(axis=1)].shape[0] > 0:
            raise Exception("Dataframe must not have any NA values")

        # 1. Load in text as Documents
        loader = DataFrameLoader(df, page_content_column)
        data = loader.load()

        print("Number of documents loaded:", len(data))

        # 2. Transform
        # split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

        # create a list of unique ids for each document based on the content
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]

        print(f"Documents split into {len(docs)} chunks")

        # 3. Embed and Store
        print("Begin embedding and storage")

        # vectorstore collection
        collection = self.get_collection()
        for id, langchain_doc in zip(ids, docs):
            # ids are unique in chromadb, so duplicate ids will be skipped -> ensures no duplicating document
            content = langchain_doc.page_content
            metadata = langchain_doc.metadata

            collection.add(ids=id, metadatas=metadata, documents=content)

        print("Documents successfully embedded and stored into vectorbase")
        print("Number of documents:", self._document_count())

        return True

    def get_genie(
        self,
        name: str,
        model_name: str = "gpt-3.5-turbo-1106",
        use_parser=True,
    ) -> Genie:
        if not self.model_is_ready():
            raise Exception("Model is not ready: please add data first")

        return Genie(
            name,
            vectorstore=self.vectorstore,
            model_name=model_name,
            use_parser=use_parser,
        )

    def get_genies(
        self, names: List[str], model_name: str = "gpt-3.5-turbo-1106", use_parser=True
    ) -> dict[str, Genie]:
        genies = {
            name: self.get_genie(
                name=name, model_name=model_name, use_parser=use_parser
            )
            for name in names
        }
        return genies
    
    def evaluate(self, df: pd.DataFrame, genie_model: str = "gpt-3.5-turbo-1106"):
        if not set(['name', 'question']).issubset(df.columns):
            raise Exception("Make sure dataframe contains columns 'question', 'answer' and 'contexts'")
        
        # ground truths value should be list
        has_ground_truths = 'ground_truths' in df.columns

        df_eval = pd.DataFrame()

        for name in df['name'].unique():
            df_data = df.loc[df['name'] == name]
            genie = self.get_genie(name, model_name=genie_model)
            print("\033[1mEvaluating", genie, '\033[0m')

            questions = df_data['question'].unique().tolist()
            ground_truths = df_data['ground_truths'].to_list() if has_ground_truths else None

            eval_result = genie.evaluate(questions, ground_truths)

            # produce dataframe w score
            df_eval_result = eval_result.to_pandas()
            df_eval_result.insert(0, 'name', '')
            df = df.assign(name=name)
            df_eval = pd.concat([df_eval, df_eval_result], axis=0, ignore_index=True)

        return df_eval
        

if __name__ == "__main__":
    import re

    def preprocess_quote(quote):
        # Replace missing spaces with space
        preprocessed_quote = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", quote)

        # ...
        # add more if necessary

        return preprocessed_quote

    df = pd.read_excel("data/qadata.xlsx")
    df[["name", "party"]] = df.username.str.split(" - ", expand=True).apply(
        lambda x: x.str.strip()
    )
    df = df.drop(["username"], axis=1)

    # taking only a portion of the data for now
    category_list = [
        "Abortion, Pro-Life & Genetic Engineering",
        "Crime, Police & Imprisonment",
        "Environment & Climate Change",
        "Gun & Property Rights",
        "Immigration, Border Security, Terrorism & Homeland Security",
        "Jobs, Economy, Trade, Business, Industry & Agriculture",
        "Education & Schools",
    ]
    df = df.loc[df.parent_question.isin(category_list)]
    df["answer"] = df["answer"].apply(preprocess_quote)

    # removing None values in "party"
    df["party"] = df["party"].apply(lambda p: p if p else "Other")

    handler = GenieMaster()
    # handler.transform_and_add_data(df, page_content_column="answer")
    genie = handler.get_genie("Joe Biden")
    print(genie.ask("Do you believe labor unions help the economy?"))
