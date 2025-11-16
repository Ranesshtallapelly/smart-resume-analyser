"""
CareerPilot - Resume Analyzer & Job Recommender (enhanced)
Save as: app.py

Features:
- Upload PDF / DOCX or paste resume text
- Extract text (PyPDF2 / docx2txt), with optional OCR fallback for scanned PDFs
- Clean text with basic NLP (NLTK stopwords)
- ATS-style scoring (TF-IDF + cosine similarity)
- Optional SBERT semantic scoring (if sentence-transformers installed)
- Matched & missing skills, simple job recommendations
- Downloadable TXT and JSON results
- Streamlit UI
"""

import streamlit as st
import io
import json
import re
import os
from typing import Tuple, List, Dict, Set

# --- Optional heavy imports wrapped in try/except ---
# (we fallback gracefully and inform the user if missing)
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# OCR (optional)
try:
    import pdf2image
    from PIL import Image
    import pytesseract
except Exception:
    pdf2image = None
    pytesseract = None

# ML libs
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

# SBERT optional
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# NLTK stopwords (ensure downloaded)
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ----------------- Helpers -----------------
def safe_msg(msg: str):
    """Show an info message in Streamlit (centralized)."""
    st.info(msg)

def extract_text_from_pdf(file) -> str:
    """Attempt to extract text from PDF using PyPDF2.
    If PyPDF2 isn't available or returns no text, return empty string."""
    if PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        # Some PDFs may be encrypted / strange formats
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text from docx using docx2txt if available."""
    if docx2txt is None:
        return ""
    try:
        # streamlit's uploaded file is a BytesIO; docx2txt can accept a path, so we write temp
        data = file.read()
        # write to temp file
        tmp_path = "temp_resume.docx"
        with open(tmp_path, "wb") as f:
            f.write(data)
        text = docx2txt.process(tmp_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return text or ""
    except Exception:
        return ""

def extract_text_via_ocr(file) -> str:
    """OCR convert PDF pages to images then run pytesseract.
    Requires pdf2image and pytesseract installed and Tesseract available on system path."""
    if pdf2image is None or pytesseract is None:
        return ""
    try:
        # write bytes to a temp file because pdf2image needs a path
        tmp_pdf = "temp_ocr.pdf"
        with open(tmp_pdf, "wb") as f:
            f.write(file.read())
        pages = pdf2image.convert_from_path(tmp_pdf)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
        try:
            os.remove(tmp_pdf)
        except Exception:
            pass
        return text
    except Exception:
        return ""

def clean_text(text: str) -> str:
    """Lowercase, remove non-alphanum chars, remove stopwords, and filter short tokens."""
    if not text:
        return ""
    text = text.lower()
    # preserve spaces and alphanumerics
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    if STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    else:
        tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Compute TF-IDF cosine similarity in percentage. Graceful fallback if sklearn not available."""
    if TfidfVectorizer is None or cosine_similarity is None:
        return 0.0
    try:
        vec = TfidfVectorizer()
        mat = vec.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
        return float(sim * 100.0)
    except Exception:
        return 0.0

# SBERT scoring (semantic) if available
sbert_model = None
if SBERT_AVAILABLE:
    try:
        # smaller model for speed
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        sbert_model = None

def sbert_similarity(resume_text: str, jd_text: str) -> float:
    """Return SBERT-based similarity (0-100). Falls back to 0 if SBERT not available."""
    if sbert_model is None:
        return 0.0
    try:
        emb1 = sbert_model.encode(resume_text, convert_to_tensor=True)
        emb2 = sbert_model.encode(jd_text, convert_to_tensor=True)
        sim = sbert_util.pytorch_cos_sim(emb1, emb2).item()
        return float(sim * 100.0)
    except Exception:
        return 0.0

def extract_keywords_set(text: str) -> Set[str]:
    """Return a set of cleaned tokens (keywords) from text."""
    cleaned = clean_text(text)
    return set(cleaned.split()) if cleaned else set()

def recommend_jobs(resume_kw: Set[str], job_catalog: List[Dict], topk: int = 5, use_sbert=False, resume_text: str="", jd_texts=None) -> List[Dict]:
    """
    Simple recommender:
      - If use_sbert and SBERT available: compute semantic similarity between resume_text and each job description string.
      - Else: compute overlap percentage between resume_kw and required skills.
    job_catalog is list of dicts: {"title": "...", "skills": [...], "description": "..."}
    """
    recs = []
    for job in job_catalog:
        required = set([re.sub(r'[^a-z0-9\s]', ' ', s).strip().lower() for s in job.get("skills", [])])
        if use_sbert and sbert_model and "description" in job and job["description"].strip():
            score = sbert_similarity(resume_text, job["description"])
        else:
            matched = resume_kw.intersection(required)
            score = (len(matched) / len(required)) * 100.0 if required else 0.0
        recs.append({
            "title": job.get("title"),
            "score": round(score, 2),
            "required": sorted(list(required)),
            "matched": sorted(list(resume_kw.intersection(required))),
        })
    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs[:topk]

def make_report_text(resume_text: str, jd_text: str, tfidf_score: float, sbert_score: float, matched: Set[str], missing: Set[str], recs: List[Dict]) -> str:
    out = io.StringIO()
    out.write("CareerPilot - Resume Analysis Report\n")
    out.write("===============================\n\n")
    out.write(f"TF-IDF ATS Match Score: {tfidf_score:.2f}%\n")
    if SBERT_AVAILABLE and sbert_score is not None:
        out.write(f"SBERT Semantic Match Score: {sbert_score:.2f}%\n")
    out.write("\nMatched Skills:\n")
    out.write(", ".join(sorted(matched)) + "\n\n")
    out.write("Missing Skills:\n")
    out.write(", ".join(sorted(missing)) + "\n\n")
    out.write("Top Recommendations:\n")
    for r in recs:
        out.write(f"- {r['title']} (Match: {r['score']}%) — Matched: {', '.join(r['matched']) if r['matched'] else 'None'}\n")
    out.write("\nOriginal (cleaned) Job Description (first 1500 chars):\n")
    out.write(jd_text[:1500] + ("\n..." if len(jd_text) > 1500 else "\n"))
    out.write("\nResume Extract (first 1500 chars):\n")
    out.write(resume_text[:1500] + ("\n..." if len(resume_text) > 1500 else "\n"))
    return out.getvalue()

# ----------------- Sample job catalog (you can replace with a CSV/JSON) -----------------
SAMPLE_JOBS = [
    {"title": "Data Scientist", "skills": ["python", "machine learning", "sql", "statistics", "nlp"], "description": "Data Scientist role requiring Python, ML, SQL, statistics and NLP."},
    {"title": "AI Engineer", "skills": ["python", "deep learning", "tensorflow", "pytorch", "computer vision"], "description": "AI Engineer role: deep learning, TensorFlow, PyTorch, CV."},
    {"title": "Cloud Engineer", "skills": ["aws", "azure", "gcp", "docker", "kubernetes"], "description": "Cloud Engineer role needing cloud platforms, Docker and K8s."},
    {"title": "Full Stack Developer", "skills": ["html", "css", "javascript", "react", "nodejs"], "description": "Full Stack Developer with React, Node.js and JS skills."},
    {"title": "Data Analyst", "skills": ["excel", "sql", "python", "tableau", "data analysis"], "description": "Data Analyst with strong SQL, Excel and Tableau skills."},
    {"title": "MLOps Engineer", "skills": ["docker", "kubernetes", "ci/cd", "mlops", "aws"], "description": "MLOps tasks: containers, orchestration, CI/CD and deployments."}
]

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="CareerPilot", layout="wide")
st.title("🚀 CareerPilot — Resume Analyzer & Job Recommender")

st.markdown(
    """
    **How to use:** Upload a resume (PDF or DOCX) or paste resume text.  
    Paste or choose a sample Job Description. Click *Analyze* to get ATS score, matched & missing skills, and job recommendations.
    """
)

# Left: input; Right: results
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📂 Upload resume (PDF or DOCX) — OR paste below", type=["pdf", "docx"])
    pasted_resume = st.text_area("Or paste resume text (optional)", height=180)
    st.markdown("---")
    sample_choice = st.selectbox("Choose sample job description (or select Custom)", ["-- Select sample --"] + [j["title"] for j in SAMPLE_JOBS] + ["Custom"])
    if sample_choice == "Custom" or sample_choice == "-- Select sample --":
        jd_input = st.text_area("Paste Job Description here", height=180)
    else:
        selected_job = next((j for j in SAMPLE_JOBS if j["title"] == sample_choice), None)
        jd_input = st.text_area("Job Description", value=selected_job["description"] if selected_job else "", height=180)

    use_semantic = st.checkbox("Use semantic matching (SBERT) if available (slower, optional)", value=False)
    use_ocr_fallback = st.checkbox("Enable OCR fallback for scanned PDFs (optional & slower)", value=False)

    analyze_btn = st.button("🔎 Analyze Resume")

with right:
    st.info("Results will appear here after analysis.")

# Main processing
if analyze_btn:
    # load resume_text
    resume_raw = ""
    # prefer pasted text if provided
    if pasted_resume and pasted_resume.strip():
        resume_raw = pasted_resume.strip()
    elif uploaded is not None:
        # streamlit's UploadedFile supports .read()
        # first attempt standard extraction
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            resume_raw = extract_text_from_pdf(uploaded)
            if not resume_raw and use_ocr_fallback:
                # rewind uploaded bytes by reopening (uploaded is in-memory, read() consumed)
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
                # use OCR
                resume_raw = extract_text_via_ocr(uploaded)
                if not resume_raw:
                    st.warning("PDF text extraction failed and OCR fallback unavailable or failed.")
        elif name.endswith(".docx"):
            resume_raw = extract_text_from_docx(uploaded)
            if not resume_raw:
                st.warning("DOCX extraction failed. Try pasting resume text manually.")
        else:
            st.warning("Unsupported file extension.")
    else:
        st.warning("Please upload a resume or paste resume text.")
        resume_raw = ""

    if resume_raw:
        st.success("✅ Resume loaded")
        # show basic preview
        st.subheader("📄 Resume Preview")
        st.write(resume_raw[:2000] + ("..." if len(resume_raw) > 2000 else ""))

        # clean / analyze
        resume_clean = clean_text(resume_raw)
        jd_clean = clean_text(jd_input or "")

        if not jd_clean:
            st.warning("Please enter or select a Job Description to compare.")
        else:
            # keywords
            resume_kw = extract_keywords_set(resume_raw)
            jd_kw = extract_keywords_set(jd_input)

            matched = resume_kw.intersection(jd_kw)
            missing = jd_kw - resume_kw

            # compute scores
            tfidf_score = tfidf_similarity(resume_clean, jd_clean)
            sbert_score = None
            if use_semantic and SBERT_AVAILABLE:
                sbert_score = sbert_similarity(resume_raw, jd_input)
            elif use_semantic and not SBERT_AVAILABLE:
                st.warning("SBERT is not installed. Semantic matching unavailable; will use TF-IDF instead.")

            # job recommendations
            recs = recommend_jobs(resume_kw, SAMPLE_JOBS, topk=6, use_sbert=(use_semantic and SBERT_AVAILABLE), resume_text=resume_raw)

            # Display
            st.subheader("✅ Analysis Results")
            col1, col2 = st.columns([1, 1])
            col1.metric("ATS Match (TF-IDF)", f"{tfidf_score:.2f}%")
            if sbert_score is not None:
                col2.metric("Semantic Match (SBERT)", f"{sbert_score:.2f}%")
            else:
                col2.write("Semantic Match: Not used")

            st.write("### 🎯 Matched Skills")
            st.write(", ".join(sorted(matched)) if matched else "None recognized")

            st.write("### ⚠️ Missing Skills (from JD)")
            st.write(", ".join(sorted(missing)) if missing else "None — your resume covers JD keywords!")

            # Pie chart visualization (if matplotlib available)
            if plt is not None:
                try:
                    labels = ["Matched", "Missing"]
                    sizes = [len(matched), len(missing)]
                    fig, ax = plt.subplots()
                    ax.pie(sizes if sum(sizes) else [1,0], labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                except Exception:
                    pass

            st.write("### 💼 Top Job Recommendations")
            for r in recs:
                st.write(f"- **{r['title']}** — Match: {r['score']}% — Matched: {', '.join(r['matched']) if r['matched'] else 'None'}")

            # Downloadable outputs
            report_txt = make_report_text(resume_raw, jd_clean, tfidf_score, sbert_score, matched, missing, recs)
            st.download_button("📥 Download TXT Report", data=report_txt, file_name="CareerPilot_Report.txt", mime="text/plain")
            export_json = {
                "tfidf_score": round(tfidf_score, 2),
                "sbert_score": round(sbert_score, 2) if sbert_score is not None else None,
                "matched": sorted(list(matched)),
                "missing": sorted(list(missing)),
                "recommendations": recs
            }
            st.download_button("📥 Download JSON Results", data=json.dumps(export_json, indent=2), file_name="CareerPilot_Result.json", mime="application/json")

            # show notes about optional features and missing packages
            st.markdown("---")
            st.write("**Notes & Hints:**")
            if docx2txt is None:
                st.write("- docx2txt not installed: DOCX extraction won't work. Install via `python -m pip install docx2txt`.")
            if PyPDF2 is None:
                st.write("- PyPDF2 not installed: PDF text extraction won't work. Install via `python -m pip install PyPDF2`.")
            if (pdf2image is None or pytesseract is None) and use_ocr_fallback:
                st.write("- OCR fallback selected but pdf2image/pytesseract not available. Install and ensure Tesseract is on PATH.")
            if use_semantic and not SBERT_AVAILABLE:
                st.write("- Semantic matching requested but sentence-transformers not installed. Install via `python -m pip install sentence-transformers`.")
            st.write("- To host: use Streamlit Cloud, Render, or run `streamlit run app.py` locally.")

    else:
        st.warning("No resume text extracted. Try another file or paste the text manually.")
