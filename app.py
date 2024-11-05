from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import os
import pickle
import openai
import io
import matplotlib.pyplot as plt
import base64
import pdfkit

openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
UPLOAD_FOLDER = 'uploads'
MAX_SESSION_SIZE = 4093  
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
lista = []

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    session['chat_history'] = []  # Initialize chat history
    session['df'] = None  # Ensure DataFrame is reset for a new session
    global lista
    lista.clear()
    session['prompts'] = []  # Initialize prompts in session
    return render_template('chat_ui.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read the CSV
        df = pd.read_csv(filepath)

        # Clean DataFrame (optional)
        df.dropna(inplace=True)  # Removes any rows with NaN values

        # Ensure DataFrame size fits within session limit
        while True:
            # Serialize the DataFrame to check its size
            serialized_df = pickle.dumps(df)
            if len(serialized_df) <= MAX_SESSION_SIZE:
                break  # Size is acceptable, exit loop

            # If too large, drop the last row
            if len(df) > 0:
                df = df.iloc[:-1]  # Remove the last row
            else:
                return jsonify({'error': 'DataFrame is too large to store in session.'}), 400

        # Update the session with the latest DataFrame
        session['df'] = serialized_df  # Serialize the DataFrame to store in session
        session['chat_history'].append({'role': 'system', 'message': 'File uploaded and cleaned successfully!'})

        return jsonify({'message': 'File uploaded and cleaned successfully!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask_openai', methods=['POST'])
def ask_openai():
    user_input = request.form.get('user_input')
    
    try:
        # Check if the DataFrame exists in the session
        if session.get('df') is None:
            return jsonify({'error': 'No DataFrame found. Please upload a CSV file first.'}), 400

        # Load the DataFrame from session
        df = pickle.loads(session['df'])

        # Convert df.info() and df.describe() to string
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        df_info = info_buffer.getvalue()
        
        describe_buffer = io.StringIO()
        df.describe().to_string(buf=describe_buffer)
        df_describe = describe_buffer.getvalue()

        # Create messages for the OpenAI API
        messages = [
            {"role": "user", "content": f"ONLY OUTPUT THE PYTHON CODE NOTHING ELSE, assign it to a variable called result. If the prompt doesn't make sense to make python, then don't. Before each answer you make, double check for syntax Here is the DataFrame info:\n{df_info}\n\nHere is the DataFrame description:\n{df_describe}\n\n{user_input}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        message = response['choices'][0]['message']['content']
        local_vars = {'df': df}
        exec(message, {"__builtins__": None}, local_vars)

        result = local_vars.get('result', None)

        if isinstance(result, pd.DataFrame):
            result_html = result.to_html(classes='table table-striped', index=False)

            # Store results and prompts in session-specific variables
            lista.append(result_html)
            session['prompts'].append(user_input)

            print(f"Session before appending result: {session.get('lista', [])}")
            
            session['chat_history'].append({'role': 'system', 'message': result_html})
            
            return jsonify({'html': result_html})  # Return as JSON

        if isinstance(result, plt.Axes):
            result = result.figure  # Get the figure from Axes

        if isinstance(result, plt.Figure):
            img_buffer = io.BytesIO()
            result.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close(result)  # Close the figure to free up memory
            session['prompts'].append(user_input)

            # Create an HTML img tag and append to lista
            img_tag = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/>' 
            lista.append(img_tag)

            return jsonify({'image': f'data:image/png;base64,{img_base64}'})  # Return image as base64

        return jsonify({'message': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_chat_history')
def get_chat_history():
    return jsonify(session.get('chat_history', []))

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    # Create HTML content for the PDF using only `lista`

    html_content = '<h1>Data Report</h1>'
    
    analysis_content = '<h2>Analysis of Tables and Graphs</h2>'
    
    # Loop through each item in the session['lista'] to generate the report
    for idx, item in enumerate(lista):
        print(item)
        if item.startswith('<table'):  # It's a table
            table_html = item
            table_analysis = analyze_table(idx)  # Function to analyze table content
            analysis_content += f'<h3>Table {idx + 1} Analysis</h3><p>{table_analysis}</p>'
            html_content += table_html  # Append the table directly to HTML content
            
        elif item.startswith('<img src="data:image/png;base64'):  # It's an image (graph)
            img_tag = item
            img_analysis = analyze_graph(idx)  # Function to analyze graph content
            analysis_content += f'<h3>Graph {idx + 1} Analysis</h3><p>{img_analysis}</p>'
            html_content += img_tag  # Append the image directly to HTML content
    
    # Append the analysis content to the HTML content
    html_content = f"<html><head><style>table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}</style></head><body>{analysis_content}{html_content}</body></html>"
    
    # Convert HTML to PDF
    pdf_path = 'static/chat_history.pdf'
    pdfkit.from_string(html_content, pdf_path)

    return send_file(pdf_path, as_attachment=True)

def analyze_table(table_index):
    """
    Function to analyze a table and generate insights using OpenAI.
    This function is called for each table in the session['lista'].
    """
    df = pickle.loads(session['df'])  # Load the original DataFrame
    
    # Get the table that was generated (from session['lista'])
    table_html = lista[table_index]
    
    # Generate a textual analysis of the table (based on df.describe and df.info)
    try:
        # Prepare the analysis for the table
        describe_buffer = io.StringIO()
        df.describe().to_string(buf=describe_buffer)
        table_description = describe_buffer.getvalue()
        
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        table_info = info_buffer.getvalue()
        
        # Create a prompt for OpenAI API to analyze the table
        messages = [
            {"role": "system", "content": "You are a data analyst. Provide insights based on the table's summary below in just plain text, no markdown."},
            {"role": "user", "content": f"Here is the DataFrame summary:\n{table_info}\n\nAnd here are the statistical details:\n{table_description}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        table_analysis = response['choices'][0]['message']['content']
        print(table_analysis)
        return table_analysis
    except Exception as e:
        return f"Error in analyzing the table: {str(e)}"

def analyze_graph(graph_index):
    """
    Function to analyze a graph and generate insights using OpenAI.
    This function is called for each graph in the session['lista'].
    """
    try:
        # Get the graph data that was used to create the plot
        df = pickle.loads(session['df'])  # Load the original DataFrame
        graph_prompt = f"Analyze this graph based on the data from the DataFrame.\nData:\n{df.head()}\nGraph {graph_index + 1}:"

        # Call OpenAI API to analyze the graph based on its data
        messages = [
            {"role": "system", "content": "You are a data analyst. Analyze the graph based on the given data in just plain text, no markdown."},
            {"role": "user", "content": graph_prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        graph_analysis = response['choices'][0]['message']['content']
        print(graph_analysis)
        return graph_analysis
    except Exception as e:
        return f"Error in analyzing the graph: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
