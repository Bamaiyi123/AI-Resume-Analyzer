import streamlit as st
from pypdf import PdfReader
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Function to extract keywords
def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()

    # Known skills and technologies list
    skill_keywords = {
        'machine learning', 'artificial intelligence', 'artificial intelligence instructor', 'data analysis', 'data science', 'deep learning',
        'neural network', 'computer vision', 'natural language processing', 'nlp', 'python', 'java',
        'javascript', 'c++', 'c#', 'sql', 'mysql', 'postgresql', 'mongodb', 'react', 'angular', 'vue',
        'node.js', 'django', 'flask', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'tableau', 'power bi', 'excel', 'aws', 'azure', 'google cloud', 'docker', 'kubernetes',
        'git', 'github', 'linux', 'windows', 'macos', 'android', 'ios', 'swift', 'kotlin',
        'flutter', 'react native', 'html', 'css', 'bootstrap', 'sass', 'php', 'ruby', 'rails',
        'devops', 'agile', 'scrum', 'kanban', 'project management', 'software engineering',
        'web development', 'mobile development', 'full stack', 'front end', 'back end', 'api',
        'rest', 'graphql', 'microservices', 'cloud computing', 'big data', 'hadoop', 'spark',
        'kafka', 'elasticsearch', 'redis', 'cybersecurity', 'blockchain', 'iot', 'automation',
        'testing', 'qa', 'ci/cd', 'jenkins', 'github actions', 'bash', 'powershell'
    }

    # Extract noun phrases that contain skill-related words
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        if 2 <= len(chunk_text) <= 60:  # keep reasonable phrase length
            # Check if chunk contains any known skill words or is a known skill phrase
            if any(skill in chunk_text for skill in skill_keywords) or chunk_text in skill_keywords:
                keywords.add(chunk_text)
            # Also store known root skill phrase if part of a longer chunk
            for skill_phrase in skill_keywords:
                if skill_phrase in chunk_text:
                    keywords.add(skill_phrase)

    # Add known single skill words that appear in the text
    for token in doc:
        token_lemma = token.lemma_.lower()
        if token.is_alpha and len(token_lemma) > 2 and token.pos_ in ['NOUN', 'PROPN']:
            if token_lemma in skill_keywords:
                keywords.add(token_lemma)

    # Determine a stable maximum output if needed (keep all for matching quality)
    return sorted(keywords)


st.title("AI Resume Analyzer (Free Version)")

# Upload resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)
    resume_keywords = extract_keywords(resume_text)
    
    st.subheader("Resume Analysis")
    st.write("**Extracted Text Preview:**")
    st.text_area("Resume Text", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200, disabled=True)
    
    st.write("**Key Skills & Technologies (Top 20):**")
    st.write(", ".join(sorted(set(resume_keywords))))  # Ensure no duplicates on display

# Job description
job_desc = st.text_area("Paste Job Description")


# Analyze
if st.button("Analyze"):
    if uploaded_file and job_desc:
        job_keywords = extract_keywords(job_desc)

        resume_set = set(resume_keywords)
        job_set = set(job_keywords)

        # Better matching with phrase containment
        matched = {
            j for j in job_set
            for r in resume_set
            if j == r or j in r or r in j
        }

        missing = job_set - matched

        score = (len(matched) / len(job_set)) * 100 if job_set else 0

        st.subheader("Match Results")
        st.write(f"Match Score: {score:.2f}%")

        st.write("Job Keywords:")
        st.write(", ".join(sorted(job_set)))

        st.write("Matched Skills:")
        st.write(", ".join(sorted(matched)))

        st.write("Missing Skills:")
        st.write(", ".join(sorted(missing)))
    else:
        st.warning("Upload resume and paste job description")
