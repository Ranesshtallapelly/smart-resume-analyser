#1
import streamlit as st
import docx2txt
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords

# --------- NLTK Setup (run once) ---------
nltk.download("stopwords")
nltk.download("punkt")

# --------- Functions --------- # 3 extracts the text from the resume
def extract_text_from_pdf(file):
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX"""
    return docx2txt.process(file)

def extract_keywords(text):#extracts the keywords from the file
    """Extract meaningful keywords"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special chars
    words = text.split()
    stop_words = set(stopwords.words("english"))
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return set(keywords)

# --------- Sample Job Dataset ---------
job_dataset = [
    {"title": "Data Scientist", "skills": {"python", "machine", "learning", "sql", "nlp", "aws"}},
    {"title": "AI Engineer", "skills": {"python", "deep", "learning", "tensorflow", "pytorch"}},
    {"title": "Cloud Engineer", "skills": {"aws", "azure", "gcp", "docker", "kubernetes"}},
    {"title": "Full Stack Developer", "skills": {"html", "css", "javascript", "react", "node", "sql"}},
]

def recommend_jobs(resume_keywords):
    recommendations = []
    for job in job_dataset:
        matched = resume_keywords.intersection(job["skills"])
        score = (len(matched) / len(job["skills"])) * 100 if job["skills"] else 0
        recommendations.append((job["title"], score, matched))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:3]  # top 3 jobs

# --------- Streamlit App ---------
st.title("🚀 CareerPilot - Resume Analyzer + Job Recommender")

# Upload Resume #2 let user upload the file
uploaded_file = st.file_uploader("📂 Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

resume_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        try:
            resume_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    else:  # DOCX
        try:
            resume_text = extract_text_from_docx(uploaded_file)
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")

    if resume_text.strip():
        st.subheader("📄 Extracted Resume Text:")
        st.write(resume_text[:1000])  # show only first 1000 chars
    else:
        st.warning("⚠️ Could not extract text from this file. Try another one.")

# Job Description input
job_description = st.text_area("📌 Paste Job Description Here")

if uploaded_file and job_description:
    # Extract keywords
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)

    # Calculate match
    matched_skills = resume_keywords.intersection(jd_keywords)
    missing_skills = jd_keywords - resume_keywords
    match_score = (len(matched_skills) / len(jd_keywords)) * 100 if jd_keywords else 0

    # Display results
    st.subheader("✅ Resume Analysis Results:")
    st.write(f"⭐ Match Score: **{match_score:.2f}%**")
    st.write(f"🎯 Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")
    st.write(f"⚠️ Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}")

    # Job Recommendations
    st.subheader("💼 Recommended Jobs:")
    recommendations = recommend_jobs(resume_keywords)
    for job, score, matched in recommendations:
        st.write(f"🔹 **{job}** → Match: {score:.2f}% | Skills: {', '.join(matched) if matched else 'None'}")
