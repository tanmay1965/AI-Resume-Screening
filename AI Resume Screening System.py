import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "" 
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text  # Avoid appending None
    return text

# Function for ranking resumes based on job description
def rank_resume(job_description, resumes):
    # Combine Job Description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Compute cosine similarity
    job_description_vector = vectors[0].reshape(1, -1)  # Reshape for correct input
    resume_vectors = vectors[1:]  # Exclude job description

    cosine_similarities = cosine_similarity(job_description_vector, resume_vectors)[0]
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening and Ranking System")

# Job Description Input
st.header("Job Description")
job_description = st.text_area("Enter the Job Description")

# File Upload
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Choose Resume files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    
    # Rank Resumes
    scores = rank_resume(job_description, resumes)

    # Display Scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)
