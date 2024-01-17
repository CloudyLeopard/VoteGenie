from genie_master import GenieMaster
import pandas as pd
import asyncio
import aiohttp
import time
from tqdm import tqdm

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

INPUT_DF_COLUMNS = ["name", "question"]
COLUMN_NAMES = ["name", "party", "usertitle", "question", "answer", "reasoning",
                "evidence", "source_content", "source_category", "source_sub_category", "cost"]

AVG_TOKENS_PER_REQUEST = 800.0

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

async def process_df(df_prompts: pd.DataFrame, vectorstore_path: str, output_file: str, model_name = "gpt-3.5-turbo"):
    # CREATE GENIES
    gm = GenieMaster(db_path = vectorstore_path)
    names = df_prompts.name.unique()
    genies = gm.get_genies(names, model_name=model_name)

    # CALCULATE REQUESTS PER MINUTE LIMIT
    rpm = calculate_requests_per_min(model_name)
    
    # CALCULATE BATCH_SIZE
    BATCH_SIZE_PERCENTAGE = 0.9
    batch_size = round(rpm * BATCH_SIZE_PERCENTAGE)
    SECONDS_PER_BATCH = 60

    # SET UP COST BENCHMARKING
    # this tracks if the cost has reached a certain point
    cost_benchmarks = [0.01, 0.1, 1, 5] + list(range(10, 210, 10))
    curr_bm = 0
    curr_cost = 0

    # PROCESS DTF
    # stores all the processed rows, to be concatenated in the end and returned
    processed_rows_list = []

    # process the dataframe by using the name and question, asking llm, get result, store in new dataframe, and return it
    num_rows = df_prompts.shape[0]

    estimated_time = num_rows / batch_size * SECONDS_PER_BATCH
    print("Estimated time (in seconds):", estimated_time)
    
    print(estimated_time)
    for batch_start in tqdm(range(0, num_rows, batch_size), desc=f"Processing batches ({batch_size} rows per batch)"):
        batch_end = min(batch_start + batch_size, num_rows)
        df_batch = df_prompts.iloc[batch_start:batch_end]

        start_time = time.time()

        processed_rows = await process_rows(df_batch, genies, output_file)
        processed_rows_list.append(processed_rows)

        curr_cost += processed_rows['cost'].sum()
        if curr_cost.sum() > cost_benchmarks[curr_bm]:
            print(f"Cost exceeded {cost_benchmarks[curr_bm]}: {round(curr_cost, 4)}")
            curr_bm += 1
        
        elapsed_time = time.time() - start_time
        sleep_time = SECONDS_PER_BATCH - elapsed_time
        # if its not the end, rest for 60 sec to prevent reaching rate limit
        if sleep_time > 0:
            time.sleep(sleep_time)
    df_new = pd.concat(processed_rows_list)

    print("Total cost:", df_new['cost'].sum())
    return df_new

def check_columns(df_people: pd.DataFrame, columns_to_check):
    for column_name in columns_to_check:
        if not column_name in df_people.columns:
            print("Column not in dataframe:", column_name)
            return False
        
    return True

def calculate_requests_per_min(model_name: str):
    # limit number of requests per minute when processing    
    if model_name == "gpt-4": # rate limit for gpt-4
        max_tpm = 40000 # tokens per min
        max_rpm = 200 # requests per min
    elif model_name == "gpt-3.5-turbo": # rate limit for gpt-3.5
        max_tpm = 90000
        max_rpm = 3500
    else: # for just random cases, only 10 requests per batch
        max_tpm = AVG_TOKENS_PER_REQUEST * 10
        max_rpm = 10
    
    rpm = min(max_rpm, int(max_tpm / AVG_TOKENS_PER_REQUEST))
    return rpm

def estimate_cost(model_name: str, tokens):
    if model_name == "gpt-4":
        return round(tokens * 0.05 / 1000, 4)
    if model_name == "gpt-3.5-turbo":
        return round(tokens * 0.002 / 1000, 4)
    return 0

# ***********
# THE ACTUALLY IMPORTANT FUNCTIONS
# ***********

def get_result_column_names():
    return COLUMN_NAMES

def get_df_prompts(df_people: pd.DataFrame, questions: list[str]):
    # make all questions lower case
    questions = [q.lower() for q in questions]

    # generate dataframe that will be used as "template" to get the results
    people_tuples = df_people.values.tolist()
    combinations = [tup + [q] for tup in people_tuples for q in questions]
    df_prompts = pd.DataFrame(combinations, columns=['name', 'party', 'usertitle', 'question'])
    return df_prompts

def get_df_remaining_prompts(df_original, df_finished):
    # CHECK COLUMNS
    columns_to_check = ["name", "party", "usertitle", "question"]
    if not check_columns(df_original, columns_to_check):
        return
    if not check_columns(df_finished, columns_to_check):
        return
    
    # Merge the two DataFrames and mark the rows using an indicator
    merged = df_original.merge(df_finished[["name", "party", "usertitle", "question"]], on=['name', "party", "usertitle", 'question'], how='left', indicator=True)

    # Filter the rows to keep only those not present in both DataFrames
    result_df = merged[merged['_merge'] == 'left_only']

    # Drop the '_merge' column, which was used for indication
    result_df = result_df.drop(columns=['_merge'])

    return result_df

async def get_df_results(df_prompts: pd.DataFrame, vectorstore_path: str,
                         output_csv: str, output_xlsx: str, model_name="gpt-3.5-turbo",
                         verbose=True):
    # CHECK COLUMNS
    if not check_columns(df_prompts, INPUT_DF_COLUMNS):
        return
    
    if verbose:
        # prompt user to make sure they want to continue
        estimated_tokens = df_prompts.shape[0] * AVG_TOKENS_PER_REQUEST
        
        print("Number of rows to be processed:", df_prompts.shape[0])

        print("Estimated cost: $"+ str(estimate_cost(model_name, estimated_tokens)))
        if not input("Continue (y/n)? ") == 'y' :
            return

        # ERASE OUTPUT FILE IF CALLED FOR
        if input(f"Clear output file ({output_csv}) content (y/n)? ") == 'y':
            open(output_csv, 'w').close()

    # PROCESS DATAFRAME
    df_new = await process_df(df_prompts, vectorstore_path, output_file=output_csv, model_name=model_name)
    df_new.to_excel(output_xlsx, index=False)
    return df_new

# ********
# MAIN
# ********
async def main():
    df_people = pd.read_pickle("./data/all_people").sample(7)
    questions = [
        "Should the government reduce spending on medicare or social security programs?", # social security and medicare
        "Should the government forgive student loan?", # education
        "Should the government regulate what is being taught in school?", # education
        "Should abortion be legal?", # abortion
        "Should there be stricter gun control laws?", # gun control
        "Should the U.S. government maintain and possibly expand spending on foreign aid?", # foreign policy
        "Should there be stricter border control measures?", # immigration
        "Should LGBTQ issues be included in school curricula?", # LGBTQ, education
        "Do you support transgender individuals' access to gender-affirming healthcare?", # LGBTQ
        "Do you support qualified immunity for police officers?", # crime
    ][:2]
    vectorstore_path = './chroma_qadata_db'
    model_name = "gpt-3.5-turbo"
    output_csv = "./result/test.csv"
    output_xlsx = "./result/text.xlsx"

    df_new = await get_df_results(df_people=df_people, questions=questions, vectorstore_path=vectorstore_path, output_csv=output_csv, output_xlsx=output_xlsx, model_name=model_name)

if __name__ == "__main__":
    asyncio.run(main())