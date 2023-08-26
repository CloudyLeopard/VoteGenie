from genie_master import GenieMaster
gm = GenieMaster(db_path="./chroma_qadata_db")

name = input("Input name: ")
if name == "":
    name = "Joe Biden"

question = input("What do you want to ask: ")
if question == "":
    question = "should the U.S. raise taxes on the rich?"

genie = gm.get_genie(name)

# test prompt
# print(genie.prompt.format(context="CONTEXT", question="QUESTION"))
# exit()

response = genie.ask(question)
print("\033[1mANSWER:\033[0m")
print(response["result"].get("answer", ""))
print("\033[1mREASONING:\033[0m")
print(response["result"].get("reasoning", ""))
print("\033[1mEVIDENCE:\033[0m")
print(response["result"].get("evidence", ""))
print("\033[1mCOST:\033[0m", response["total_cost"])
print("="*20)

# for doc in response["source_documents"]:
#     print("\033[1mSource:\033[0m")
#     print("Category: " + doc["source_category"])
#     print("Category: " + doc["source_sub_category"])
#     print(doc["source_content"])
#     print("-"*20)
