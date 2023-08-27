from genie_master import GenieMaster
import pandas as pd
import asyncio
import aiohttp
import time
import tqdm.asyncio
from tqdm import tqdm

async def process_row(session, genie, row, output_path, container):
    # ask genie with the question and get a result
    output = await genie.async_ask(row["question"])

    # parsing the result, and storing it into into the row that will be returned
    result = output["result"]
    row['answer'] = result.get('answer', "")
    row['reasoning'] = result.get('reasoning', "")
    row['evidence'] = result.get('evidence', "")
    
    if output["source_documents"]:
        source_doc = output["source_documents"][0]
        row['source_content'] = source_doc['source_content']
        row['source_category'] = source_doc['source_category']
        row['source_sub_category'] = source_doc['source_sub_category']
    
    row['cost'] = output["total_cost"]
    new_row_df = pd.DataFrame([row])
    new_row_df.to_csv(output_path, mode='a', header=False, index=False)

    # track total cost in current "session".
    container[0] += output["total_cost"]

    return row

async def process_rows(df, genies, output_path, include_header=False):
    # keep tracks of cost in this "session". container is a list so that it can be passed throughout function calls
    container = [0] # cost

    # async stuff
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in df.iterrows():
            genie = genies[row["name"]]
            task = process_row(session, genie, row.copy(), output_path, container)
            tasks.append(task)
        processed_rows = await asyncio.gather(*tasks)
    
    processed_df = pd.DataFrame(processed_rows)
    # processed_df.to_csv(output_path, index=False)  # Save as CSV

    return processed_df

class VoteEasyGenieWrapper:
    US_STATES = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
        "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
        "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
        "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
        "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
        "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
    ]

    def __init__(self, db_path):
        self.gm = GenieMaster(db_path = db_path)

    async def process_df(self, df_people: pd.DataFrame, questions: list[str], output_file: str, clear_file = True, model_name = "gpt-3.5-turbo"):
        columns_to_check = ["name", "party", "usertitle"]
        for column_name in columns_to_check:
            if not column_name in df_people.columns:
                print("Column not in dataframe:", column_name)
        
        # make all questions lower case
        questions = [q.lower() for q in questions]

        # generate dataframe that will be used as "template" to get the results
        people_tuples = df_people.values.tolist()
        combinations = [tup + [q] for tup in people_tuples for q in questions]
        df_prompts = pd.DataFrame(combinations, columns=['name', 'party', 'usertitle', 'question'])

        # get a genie for each person that we will be finding data on
        genies = {name: self.gm.get_genie(name, model_name=model_name) for name in df_people.name.unique()}
        if clear_file:
            open(output_file, 'w').close() # erase content inside csv file

        # limit number of requests per minute when processing
        RPM = 10
        # rate limit for gpt-4: 40,000 tokens/min, 200 req/min. For this purpose, 40K TPM / 750 Tokens per Requests =  ~50 RPM < 200 RPM.
        # So let's limit it to 50 RPM
        if model_name == "gpt-4":
            RPM = 50
        # rate limit for gpt-3.5: 90,000 TPM, 3500 req/min. For this purpose, 90K/750 = 120.
        # So let's limit it to 120
        if model_name == "gpt-3.5-turbo":
            RPM = 120
        
        # number of rows to be processed each minute
        batch_size = round(RPM * 0.9)

        # track if the cost has reached a certain point
        cost_benchmarks = [0.01, 0.1, 1, 5] + list(range(10, 110, 10))
        curr_bm = 0
        curr_cost = 0

        # stores all the processed rows, to be concatenated in the end and returned
        processed_rows_list = []

        # process the dataframe by using the name and question, asking llm, get result, store in new dataframe, and return it
        num_rows = df_prompts.shape[0]
        for batch_start in tqdm(range(0, num_rows, batch_size), desc="Processing rows", total=num_rows):
            batch_end = min(batch_start + batch_size, num_rows)
            df_batch = df_prompts.iloc[batch_start:batch_end]

            processed_rows = await process_rows(df_batch, genies, output_file)
            processed_rows_list.append(processed_rows)

            curr_cost += processed_rows['cost'].sum()
            if curr_cost.sum() > cost_benchmarks[curr_bm]:
                print(f"Cost exceeded {cost_benchmarks[curr_bm]}: {round(curr_cost, 4)}")
                curr_bm += 1

            time.sleep(60)
        df_new = pd.concat(processed_rows_list)

        print("Total cost:", df_new['cost'].sum())
        return df_new
