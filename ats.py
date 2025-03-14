import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Function to extract data from Resume
def extract_text_from_pdf(file):
    # Open the PDF file in read-binary mode
    pdf=PdfReader(file)
    # Extract text from the first page of the PDF
    text=" " 
    for page in pdf.pages:
        text += page.extract_text()
    return text
#Function for Ranking the resume based on Job description
def rank_resume(job_description, resume):
   #Combine Job Description with resumes
   documents=[job_description]+resume
   vectorizer=TfidfVectorizer().fit_transform(documents)
   vectors=vectorizer.toarray()
   # Calculate cosine similarity between job description and each resume
   job_description_vector=vectors[0]
   resume_vectors=vectors[1:]
   cosine_similarities=cosine_similarity()
   return cosine_similarities

#Streamlit app
st.title("AI Resume Screening and Ranking System")

#job Description
st.header("Job Description")
job_description=st.text_area("Enter the Job Description")

#File Upload
st.header("Upload Resume")
uploaded_file = st.file_uploader("Choose a Resume file", type=["pdf"],accept_multiple_files=True)

if uploaded_file and job_description:
    st.header("Ranking Resume")
    resumes=[]
    for file in uploaded_file:
        text=extract_text_from_pdf(file)
        resumes.append(text)

#Rank Resume
scores=rank_resume(job_description,resumes)

#Display Scores
results=pd.DataFrame({"Resume":[file.name for file in uploaded_file],"Score": [scores]* len(uploaded_file)})
results=results.sort_values(by=scores,ascending=False)

st.write(results)