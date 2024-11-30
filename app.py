#Flask Imports & File imports
from flask import Flask, render_template, request, jsonify, session, send_file
import os
import pickle
import io
import base64
import pdfkit
import re
import PIL
from PIL import Image

#Data Libraries
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from datetime import datetime

#Machine Leanring imports
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

#AI Imports
import openai
import google.generativeai as genai



#Database
from pymongo import MongoClient





#AI Keys
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY= os.getenv("GEM_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")  # MongoDB URI stored as environment variable
client = MongoClient(MONGO_URI)
db = client['data_analysis']  # Define database name
collection = db['uploaded_files']  # Define collection name for uploaded files


# Constants
UPLOAD_FOLDER = 'uploads'
MAX_SESSION_SIZE = 4093  

# Initialize Flask app
application = Flask(__name__)
application.secret_key = 'your_secret_key_here'  # Required for session management
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#To keep track of all users chat's
lista = []
prompts = []

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists("/tmp/saved_tables"):
    os.makedirs("/tmp/saved_tables")

if not os.path.exists("/tmp/saved_graphs"):
    os.makedirs("/tmp/saved_graphs")

@application.route('/')
def home():

    session['chat_history'] = [
    {
        'role': 'system', 
        'message': '''
        <div style="font-family: 'Arial', sans-serif; color: #fff; line-height: 1.6;">
            <p>Welcome to the <strong>Data Analysis Chatbot</strong>!</p>
            <p>Upload a CSV file to start analyzing your data and ask questions to get insights, generate plots, or summarize information from your dataset.</p>
            
            <p><strong>Here are some commands to get started:</strong></p>
            <ul style="margin-left: 20px;">
                <li><strong>!info</strong>: Learn more about the features and capabilities of this chatbot.</li>
                <li><strong>!help</strong>: Get a list of example prompts you can use to interact with the chatbot.</li>
            </ul>

            <p>Once your data is uploaded, you can begin by asking questions like:</p>
            <ul style="margin-left: 20px;">
                <li>"Show me the first 10 rows of the dataframe."</li>
                <li>"Generate a bar plot of 'Product Category' vs 'Total Sales'.</li>
                <li>"Train a machine learning model to predict 'Price' based on 'Size' and 'Location'.</li>
            </ul>
        </div>
        '''
    }
]


    session['df'] = None  # Ensure DataFrame is reset for a new session
    global lista
    lista.clear()

    global prompts
    prompts.clear()

    session['prompts'] = []  # Initialize prompts in session
    return render_template('chat_ui.html')



def save_df_to_mongo(df, file_id):
    """Stores a DataFrame as JSON in MongoDB with an identifier."""
    data_json = df.to_dict(orient='records')  # Convert DataFrame to JSON
    collection.replace_one({'file_id': file_id}, {'file_id': file_id, 'data': data_json}, upsert=True)



def load_df_from_mongo(file_id):
    """Loads a DataFrame from MongoDB based on a file identifier."""
    document = collection.find_one({'file_id': file_id})
    if document:
        return pd.DataFrame(document['data'])
    return None



@application.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        df.dropna(inplace=True)  # Clean DataFrame by removing rows with NaN values

        # Check DataFrame size and decide storage location
        serialized_df = pickle.dumps(df)
        if len(serialized_df) <= MAX_SESSION_SIZE:
            session['df'] = serialized_df
            storage_location = 'session'
            session['chat_history'].append({'role': 'system', 'message': 'File uploaded and stored in session successfully!'})
        else:
            file_id = session.get('file_id', file.filename)  # Use filename as unique identifier
            save_df_to_mongo(df, file_id)
            session['df'] = None
            storage_location = 'mongodb'
            session['file_id'] = file_id
            session['chat_history'].append({'role': 'system', 'message': 'Data too large for session; stored in MongoDB.'})

        return jsonify({'message': f'File uploaded and stored in {storage_location}.'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@application.route('/ask_openai', methods=['POST'])
def ask_openai():
    user_input = request.form.get('user_input')

    check = False

    if user_input == '!help':
        return jsonify({'message' : help()})

    if user_input == '!info':
        return jsonify({'message' : info()})

    try:

        # Load DataFrame from session or MongoDB
        if session.get('df') is not None:
            df = pickle.loads(session['df'])  # Load DataFrame from session data
            check = True
        else:
            file_id = session.get('file_id')
            if not file_id:
                return jsonify({'error': 'No DataFrame found. Please upload a CSV file first.'}), 400

            df = load_df_from_mongo(file_id)  # Load DataFrame from MongoDB by file_id
            if df is None:
                return jsonify({'error': 'Data could not be loaded from MongoDB.'}), 500

        

        # Convert df.info() and df.describe() to string
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        df_info = info_buffer.getvalue()
        
        describe_buffer = io.StringIO()
        df.describe().to_string(buf=describe_buffer)
        df_describe = describe_buffer.getvalue()

        # Create messages for the OpenAI API with updated prompt
        messages = [
            {"role": "user", "content": (
                f"ONLY OUTPUT THE PYTHON CODE NOTHING ELSE. IN YOUR CODE DON'T HAVE IMPORT STATEMENTS. The result of the script will be placed inside of a variable called result. The LAST LINE OF EVERY SCRIPT MUST BE ASSIGNED TO RESULT"
                f"The following libraries are available for your use: Pandas, Seaborn, Matplotlib, and scikit-learn. "
                f"Use scikit-learn if machine learning is relevant to the prompt (MAKE SURE THE RESULT IS EITHER A TABLE OR GRAPH OR NUMERICAL ). If the prompt doesn't make sense to make python, then don't. "
                f"Before each answer you make, double check for syntax. Here is the DataFrame info:\n{df_info}\n\nHere is the DataFrame description:\n{df_describe}\n\n{user_input}"
            )}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        try:
            message = response['choices'][0]['message']['content']
            message = re.sub(r'^\s*from\s+[^\n]*\n|^\s*import\s+[^\n]*\n', '', message, flags=re.MULTILINE)
            print(message)
            local_vars = {
                'df': df, 'pd': pd, 'plt': plt, 'seaborn': sns, 'train_test_split': train_test_split, 'LinearRegression': LinearRegression, 
                'LogisticRegression': LogisticRegression, 'DecisionTreeClassifier': DecisionTreeClassifier,  'RandomForestClassifier': RandomForestClassifier, 
                'KMeans': KMeans, 'openai': openai, 'pickle': pickle, 'base64': base64, 'pdfkit': pdfkit, 'np': np, 'sklearn': sklearn
            } 
        
            exec(message, {"__builtins__": None} , local_vars)
        except Exception as e:
            print(f"Error during execution: {str(e)}")
            return jsonify({'message' : 'Please provide more details or try again. It might help to use key vocab. Try !help for more details'})

        result = local_vars.get('result', None)
        print(result)

        if isinstance(result, pd.DataFrame):
            result_html = result.to_html(classes='table table-striped', index=False)



            table_filename = f"/tmp/saved_tables/table_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
            with open(table_filename, 'w') as f:
                f.write(result_html)

            if(check): #If table exceeds pdf size, don't add it to the pdf tracker
                 lista.append(result_html)
                 prompts.append(table_filename)

            # Store results and prompts in session-specific variables
            session['prompts'].append(table_filename)
            session['chat_history'].append({'role': 'system', 'message': result_html})
            
            return jsonify({'html': result_html})  # Return as JSON

        if isinstance(result, plt.Axes):
            result = result.figure  # Get the figure from Axes

        if isinstance(result, plt.Figure):
            img_buffer = io.BytesIO()
            result.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            img_filename = f"/tmp/saved_graphs/graph_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            with open(img_filename, 'wb') as img_file:
                img_file.write(img_buffer.getvalue())
            
            plt.close(result)  # Close the figure to free up memory
            session['prompts'].append(img_filename)
            prompts.append(img_filename)

            # Create an HTML img tag and append to lista
            img_tag = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/>' 
            lista.append(img_tag)

            return jsonify({'image': f'data:image/png;base64,{img_base64}'})  # Return image as base64

        return jsonify({'message': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500




@application.route('/get_chat_history')
def get_chat_history():
    return jsonify(session.get('chat_history', []))



@application.route('/download_pdf', methods=['GET'])
def download_pdf():
    # Create HTML content for the PDF using only `lista`
    html_content = '<h1>Data Report</h1>'
    
    # Loop through each item in the session['lista'] to generate the report
    for idx, item in enumerate(lista):
        if item.startswith('<img src="data:image/png;base64'):  # It's an image (graph)
            img_tag = item
            img_analysis = analyze_graph(prompts[idx])  # Function to analyze graph content

            # Add the image followed by its analysis
            html_content += f'<div>{img_tag}</div>'
            html_content += f'<h3>Graph {idx + 1} Analysis</h3><p>{img_analysis}</p>'
        
        elif item.startswith('<table'):  # It's a table
            table_html = item
            table_analysis = analyze_table(prompts[idx])  # Function to analyze table content

            # Add the table followed by its analysis
            html_content += f'<div>{table_html}</div>'
            html_content += f'<h3>Table {idx + 1} Analysis</h3><p>{table_analysis}</p>'
    
    # Wrap up the HTML with the content
    html_content = f"<html><head><style>table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}</style></head><body>{html_content}</body></html>"
    
    # Convert HTML to PDF
    pdf_path = 'static/chat_history.pdf'
    pdfkit.from_string(html_content, pdf_path)

    return send_file(pdf_path, as_attachment=True)




def help():
    example_prompts = """
    <h2>Here are some example prompts you can use to interact with your data analysis chatbot, especially for data analysis, machine learning, and visualization tasks:</h2>
    
    <h3>Data Exploration and Manipulation</h3>
    <ul>
        <li>"Show me the first 10 rows of the dataframe."</li>
        <li>"Display a summary of the dataframe."</li>
        <li>"How many unique values are there in the 'Category' column?"</li>
        <li>"Sort the dataframe by 'Revenue' in descending order and show the top 5 rows."</li>
        <li>"Filter the dataframe to only show rows where 'Sales' is greater than 1000."</li>
    </ul>
    
    <h3>Data Visualization</h3>
    <ul>
        <li>"Generate a bar plot of 'Product Category' vs. 'Total Sales'."</li>
        <li>"Create a scatter plot of 'Age' vs. 'Income' with a linear trend line."</li>
        <li>"Plot a histogram of 'Purchase Amount' with 20 bins."</li>
        <li>"Make a heatmap of the correlation matrix for numerical columns."</li>
        <li>"Generate a box plot for 'Sales' by 'Region'."</li>
    </ul>
    
    <h3>Machine Learning (using scikit-learn)</h3>
    <ul>
        <li>"Train a linear regression model to predict 'Price' based on 'Size' and 'Location'."</li>
        <li>"Create a decision tree to classify 'Customer Churn' based on 'Age' and 'Account Length'."</li>
        <li>"Split the data into training and test sets with a test size of 0.2, and train a logistic regression model to predict 'Purchased'."</li>
        <li>"Perform K-means clustering with 3 clusters on the 'Income' and 'Spending Score' columns."</li>
        <li>"Train a random forest model to classify 'Product Type' based on 'Features 1-5' and show the feature importances."</li>
    </ul>
    
    <h3>Statistical Analysis</h3>
    <ul>
        <li>"Calculate the correlation between 'Sales' and 'Advertising Spend'."</li>
        <li>"Run a t-test between 'Revenue' in Region A and Region B."</li>
        <li>"What is the average 'Customer Satisfaction' score for each product category?"</li>
        <li>"Calculate the standard deviation of 'Delivery Time' for each shipping method."</li>
        <li>"Show a summary of the statistical metrics for the 'Profit' column."</li>
    </ul>
    
    <h3>Advanced Analysis and Insights</h3>
    <ul>
        <li>"Generate insights about trends in 'Monthly Sales' over time."</li>
        <li>"Analyze the seasonal trend of 'Electricity Consumption' and create a time series plot."</li>
        <li>"Show me the top 3 most frequently purchased products."</li>
        <li>"Visualize the trend of 'Revenue' over the last 5 years."</li>
        <li>"Display insights on customer segmentation based on 'Age' and 'Spending Score'."</li>
    </ul>

    <p>These prompts should help you explore various data operations, visualize trends, and apply machine learning tasks using the available libraries (pandas, seaborn, matplotlib, scikit-learn). Let me know if you need further customization of prompts!</p>
    """
    return example_prompts


def info():
    info_text = """
    <h2>About dataNow! - Your Data Analysis Assistant</h2>
    
    <p><strong>Name:</strong> <em>dataNow!</em></p>
    
    <h3>Features</h3>
    <ul>
        <li><strong>Data Exploration:</strong> Quickly explore datasets, view summaries, and filter data to focus on the most relevant information.</li>
        <li><strong>Data Visualization:</strong> Create a wide variety of charts and graphs including bar plots, scatter plots, histograms, and more.</li>
        <li><strong>Machine Learning:</strong> Use pre-built machine learning models for tasks such as regression, classification, clustering, and feature importance analysis.</li>
        <li><strong>Statistical Analysis:</strong> Perform statistical operations like correlation, hypothesis testing, and summary statistics.</li>
        <li><strong>Advanced Insights:</strong> Generate insights based on trends, customer segmentation, and time series analysis.</li>
    </ul>

    <h3>Facts About dataNow!</h3>
    <ul>
        <li><strong>Platform:</strong> Built on Python with libraries such as pandas, scikit-learn, seaborn, and matplotlib.</li>
        <li><strong>Supported Operations:</strong> Data manipulation, visualization, machine learning, and statistical analysis all in one place.</li>
        <li><strong>Data Formats:</strong> Works with CSV, Excel, and other common data formats for seamless integration.</li>
        <li><strong>Designed For:</strong> Analysts, data scientists, and anyone needing to make sense of large datasets quickly.</li>
        <li><strong>Customizable:</strong> Easily extendable to fit specific use cases, including custom models and analysis tasks.</li>
    </ul>

    <h3>Everything You Need to Know</h3>
    <p><strong>dataNow!</strong> is an intelligent data analysis assistant that helps you explore and visualize your data effortlessly. It is perfect for businesses, data analysts, and anyone who works with large datasets. Whether you need to conduct basic data exploration, create meaningful visualizations, apply machine learning models, or dive into statistical analysis, dataNow! provides everything you need in one intuitive interface.</p>
    
    <p>With <strong>dataNow!</strong>, you can:</p>
    <ul>
        <li>Explore and filter your data with simple commands.</li>
        <li>Visualize trends, distributions, and relationships in your data with powerful plotting tools.</li>
        <li>Build and evaluate machine learning models for predictive analysis.</li>
        <li>Perform statistical analysis to gain deeper insights into your data.</li>
    </ul>

    <p>Whether you're a beginner or an expert, <strong>dataNow!</strong> makes data analysis faster and easier.</p>
    """
    return info_text



def analyze_graph(image_path):
    """Analyze the graph using Gemini API with improved prompts."""

    # Open the image
    img = Image.open(image_path)

    # Create an insightful prompt for Gemini API
    prompt = (
        "You are a skilled data analyst and machine learning expert. "
        "Analyze the contents of the following graph carefully and provide a comprehensive analysis, including the following aspects: "
        "- Key trends, patterns, or outliers visible in the graph. "
        "- Any correlations, insights, or anomalies that can be drawn from the visualization. "
        "- Specific observations about the data points, axes, and labels. "
        "- How the graph could be interpreted in a data-driven context. "
        "Please return the analysis in a concise report format, without markdown, and focus on presenting clear, actionable insights."
    )

    # Generate content using Gemini API
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content([prompt, img])

    return response.text


def analyze_table(table_path):
    """Analyze the table using Gemini API with improved prompts."""

    try:
        # Open the table file and read its contents
        with open(table_path, 'r') as table_file:
            table_data = table_file.read()

        # Create a detailed prompt for Gemini to analyze the table
        prompt = (
            "You are an experienced data analyst and statistician. "
            "Please provide a thorough analysis of the following table, covering the key data insights, trends, and patterns. "
            "Your analysis should include the following: "
            "- Descriptive statistics, such as mean, median, standard deviation, or distribution characteristics (if applicable). "
            "- Insights about any visible correlations or relationships between variables. "
            "- Observations on any notable outliers or anomalies in the data. "
            "- How the data can be interpreted in a business or decision-making context. "
            "Please focus on providing actionable insights and avoid markdown formatting. "
            "Present the analysis as a concise report, including any significant findings."
        )

        # Generate content using the Gemini API
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content([prompt, table_data])

        return response.text  # Return Gemini's analysis of the table

    except Exception as e:
        return f"Error in analyzing the table: {str(e)}"



if __name__ == "__main__":
    application.run(host="0.0.0.0", port=10000)

