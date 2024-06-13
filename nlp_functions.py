import PyPDF2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from transformers import MarianMTModel, MarianTokenizer
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file_path):
    text = ''
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Preprocess text (tokenization, lowercase, remove stopwords)
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Extract top keywords from the text
def extract_keywords(text, n=100):
    preprocessed_text = preprocess_text(text)
    word_freq = Counter(preprocessed_text)
    keywords = word_freq.most_common(n)
    return [keyword[0] for keyword in keywords]

# Translate text from English to Hindi
def translate_text(text):
    # Load the MarianMT model and tokenizer for translation between English ('en') and Hindi ('hi')
    model_name = 'Helsinki-NLP/opus-mt-en-hi'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    # Split the text into smaller chunks (you may need to adjust the chunk size)
    max_chunk_size = 500  # Adjust as needed based on the model's limitations
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    translated_chunks = []
    for chunk in chunks:
        # Translate text from English to Hindi
        translated = tokenizer(chunk, return_tensors="pt")
        translated_text = model.generate(**translated, max_length=150)
        decoded_translation = tokenizer.decode(translated_text[0], skip_special_tokens=True)
        translated_chunks.append(decoded_translation)

    # Join the translated chunks
    translated_text = ' '.join(translated_chunks)
    return translated_text

# Generate a word cloud from the PDF text
def generate_wordcloud(pdf_text):
    text = re.sub(r'[^a-zA-Z\s]', '', pdf_text)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Perform question answering for given questions and context
def question_answering(questions, context):
    # Use a question-answering model (pipeline) to answer the provided questions
    # Use the context (pdf_text) for answering
    # Return a list of answers for each question
    return answers_list

# Summarize text using TextRank
def text_summarization(text, num_sentences=50):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    text_summary = ""
    for sentence in summary:
        text_summary += str(sentence)
    return text_summary

# Perform topic modeling using Latent Dirichlet Allocation (LDA)
def topic_modeling(pdf_text, num_topics=5, no_top_words=10):
    preprocessed_text = ' '.join(preprocess_text(pdf_text))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(topic_words)
    return topics
