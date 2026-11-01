import streamlit as st
import pandas as pd
import re

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("AI Resume Analyzer and Job Recommendation System")

st.write(
    "Upload your resume to analyze your skills and find suitable job roles."
)


# -----------------------------
# FUNCTION TO READ PDF
# -----------------------------

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


# -----------------------------
# FUNCTION TO READ DOCX
# -----------------------------

def extract_docx_text(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + " "

    return text


# -----------------------------
# CLEAN TEXT
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text


# -----------------------------
# SKILLS DATABASE
# -----------------------------

skills = [

    "python",
    "java",
    "c",
    "c++",
    "html",
    "css",
    "javascript",

    "sql",
    "mysql",
    "mongodb",

    "pandas",
    "numpy",

    "machine learning",
    "deep learning",

    "scikit-learn",
    "tensorflow",
    "pytorch",

    "nlp",
    "opencv",

    "power bi",
    "excel",

    "git",
    "github",

    "docker",
    "fastapi",
    "flask",

    "aws",
    "azure"
]


# -----------------------------
# JOB ROLES
# -----------------------------

job_roles = {

    "Data Analyst":
        "python sql excel pandas numpy power bi data analysis",

    "Machine Learning Engineer":
        "python machine learning pandas numpy scikit-learn tensorflow docker fastapi",

    "AI Engineer":
        "python machine learning deep learning tensorflow pytorch nlp",

    "NLP Engineer":
        "python nlp machine learning deep learning transformers",

    "Python Developer":
        "python flask fastapi sql git github"
}


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(

    "Upload your Resume",

    type=["pdf", "docx"]

)


if uploaded_file is not None:

    st.success("Resume uploaded successfully!")

    st.write("File Name:", uploaded_file.name)


    # -------------------------
    # EXTRACT TEXT
    # -------------------------

    if uploaded_file.name.endswith(".pdf"):

        resume_text = extract_pdf_text(uploaded_file)

    else:

        resume_text = extract_docx_text(uploaded_file)


    resume_text = clean_text(resume_text)


    # -------------------------
    # SHOW EXTRACTED TEXT
    # -------------------------

    with st.expander("View Extracted Resume Text"):

        st.write(resume_text)


    # -------------------------
    # SKILL EXTRACTION
    # -------------------------

    found_skills = []

    for skill in skills:

        if skill.lower() in resume_text:

            found_skills.append(skill)


    st.subheader("Skills Found")

    if found_skills:

        st.write(found_skills)

    else:

        st.warning("No skills found.")


    # -------------------------
    # TF-IDF MATCHING
    # -------------------------

    documents = [resume_text]

    documents.extend(job_roles.values())


    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)


    similarity_scores = cosine_similarity(

        vectors[0:1],

        vectors[1:]

    )[0]


    results = []


    for role, score in zip(

        job_roles.keys(),

        similarity_scores

    ):

        results.append({

            "Job Role": role,

            "Match Score": round(score * 100, 2)

        })


    results_df = pd.DataFrame(results)


    results_df = results_df.sort_values(

        by="Match Score",

        ascending=False

    )


    # -------------------------
    # DISPLAY MATCH SCORES
    # -------------------------

    st.subheader("Job Match Scores")

    st.dataframe(results_df)


    # -------------------------
    # TOP 3 RECOMMENDATIONS
    # -------------------------

    st.subheader("Top 3 Recommended Roles")


    top_roles = results_df.head(3)


    for index, row in top_roles.iterrows():

        st.write(

            row["Job Role"],

            "-",

            str(row["Match Score"]) + "%"

        )


    # -------------------------
    # SELECT TARGET ROLE
    # -------------------------

    st.subheader("Skill Gap Analysis")


    target_role = st.selectbox(

        "Select Target Job Role",

        list(job_roles.keys())

    )


    required_skills = job_roles[target_role].split()


    missing_skills = []


    for skill in required_skills:

        if skill not in resume_text:

            missing_skills.append(skill)


    st.write("Target Role:", target_role)


    st.write("Missing Skills:")


    if missing_skills:

        st.write(missing_skills)

    else:

        st.success(

            "Your resume contains most of the required skills!"

        )


    # -------------------------
    # LEARNING ROADMAP
    # -------------------------

    st.subheader("Learning Roadmap")


    if missing_skills:

        for i, skill in enumerate(

            missing_skills[:4],

            start=1

        ):

            st.write(

                "Week",

                i,

                ":",

                "Learn",

                skill

            )

    else:

        st.write(

            "Continue improving your projects and practical experience."

        )