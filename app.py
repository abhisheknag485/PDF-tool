from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import MarianMTModel, MarianTokenizer, pipeline

app = Flask(__name__)

pdf_text = ""  # Variable to store extracted text

# Load the MarianMT model and tokenizer for translation between English ('en') and Hindi ('hi')
model_name = 'Helsinki-NLP/opus-mt-en-hi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global pdf_text
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            pdf_text = extract_text_from_pdf(uploaded_file)
        return render_template('index.html')

@app.route('/extract_text', methods=['POST'])
def extract_text():
    return render_template('index.html', extracted_text=pdf_text)

@app.route('/translate_text', methods=['POST'])
def translate_text():
    translated_text = translate_to_hindi(pdf_text)
    return render_template('index.html', translated_text=translated_text)

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    wordcloud_image = create_wordcloud(pdf_text)
    return render_template('index.html', wordcloud_image=wordcloud_image)

@app.route('/answer_question', methods=['POST'])
def answer_question():
    if request.method == 'POST':
        user_question = request.form['question']
        answer = perform_question_answering(user_question, pdf_text)
        return render_template('index.html', answer=answer)

# Function to extract text from uploaded PDF file
from PyPDF2 import PdfReader

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf = PdfReader(uploaded_file)
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to translate text to Hindi
def translate_to_hindi(text):
    translated_chunks = []
    max_chunk_size = 500  # Adjust as needed based on the model's limitations
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    for chunk in chunks:
        translated = tokenizer(chunk, return_tensors="pt")
        translated_text = model.generate(**translated, max_length=150)
        decoded_translation = tokenizer.decode(translated_text[0], skip_special_tokens=True)
        translated_chunks.append(decoded_translation)
    translated_text = ' '.join(translated_chunks)
    return translated_text

# Function to generate word cloud from text
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud_image_path = 'static/wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)
    return wordcloud_image_path

# Function to perform question answering
def perform_question_answering(question, context):
    qa_pipeline = pipeline("question-answering")
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

if __name__ == '__main__':
    app.run(debug=True)
