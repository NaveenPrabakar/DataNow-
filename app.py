from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import os
import pickle
import openai
import io
import matplotlib.pyplot as plt
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pdfkit

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set your secret key
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

lista = []
prompts = []

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    session['chat_history'] = []  # Initialize chat history
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

        # Read and validate CSV
        df = pd.read_csv(filepath)
        if 'Location' not in df.columns or 'Lemon' not in df.columns:
            return jsonify({'error': 'CSV must contain "Location" and "Lemon" columns'}), 400

        df.dropna(inplace=True)
        session['df'] = pickle.dumps(df)
        session['chat_history'].append({'role': 'system', 'message': 'File uploaded and cleaned successfully!'})

        return jsonify({'message': 'File uploaded and cleaned successfully!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask_openai', methods=['POST'])
def ask_openai():
    user_input = request.form.get('user_input')
    
    try:
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
        print(message)
        local_vars = {'df': df}
        exec(message, {"__builtins__": None}, local_vars)

        result = local_vars.get('result', None)

        if isinstance(result, pd.DataFrame):
            result_html = result.to_html(classes='table table-striped', index=False)

            lista.append(result_html)
            prompts.append(user_input)
            
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
            prompts.append(user_input)

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

    print(lista)
    # Create HTML content for the PDF using only `lista`
    html_content = '<h1>Data Report</h1>'
    
    # Append tables from `lista`
    for table_html in lista:

        html_content += table_html  # Append the tables directly to HTML content

    # Ensure proper structure for HTML
    html_content = f"<html><head><style>table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}</style></head><body>{html_content}</body></html>"

    # Convert HTML to PDF
    pdf_path = 'static/chat_history.pdf'
    pdfkit.from_string(html_content, pdf_path)

    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
