from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import pandas as pd
from vote_easy_genie_lib import *


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
AVG_TOKENS_PER_REQUEST = 800

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
]
model_name = "gpt-3.5-turbo"
vectorstore_path = "./chroma_qadata_db"

# Configure the upload folder
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
            return redirect(request.url)
    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)
        session["filepath"] = filepath


        df_people = pd.read_excel(uploaded_file)
        session["df_people"] = df_people
        row_nums = df_people.shape[0]
        table_html = df_people.iloc[:5].to_html(classes='table table-striped', index=False)
        # Process the uploaded file (you can add your processing logic here)
        return render_template('index.html', politician_nums=row_nums, table=table_html, questions=questions)
    return redirect(url_for('index'))

@app.route('/prompts', methods=['GET'])
def prompts():
    if not "df_people" in session:
          return redirect(url_for('index'))
    df_people = session["df_people"]
    df_prompts = get_df_prompts(df_people, questions)
    session["df_prompts"] = df_prompts
    row_nums = df_prompts.shape[0]
    # prompt user to make sure they want to continue
    estimated_tokens = row_nums * AVG_TOKENS_PER_REQUEST
    estimated_cost = estimate_cost(model_name, estimated_tokens)
    table_html = df_prompts.iloc[:5].to_html(classes='table table-striped', index=False)
    
    return render_template('index.html', prompt_nums=row_nums, estimated_cost=estimated_cost, table=table_html)

@app.route('/process', methods=['GET'])
async def process():
    if not "df_prompts" in session:
        return redirect(url_for('index'))
    df_prompts = session["df_prompts"]

    output_csv = "./temp/temp.csv"
    output_xlsx = "./temp/temp.xlsx"
    df_results = await get_df_results(df_prompts, vectorstore_path, output_csv, output_xlsx, model_name, verbose=False)
    session["df_results"] = df_results

    total_cost = df_results['cost'].sum()
    table_html = df_results.to_html(classes='table table-striped', index=False)
    return render_template('index.html', total_cost=total_cost, table=table_html)

if __name__ == '__main__':
    app.run(debug=True)
