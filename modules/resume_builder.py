import io
import nltk
import os
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from modules import tips

# Download NLTK data safely only if not already downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)

# Skill database
skills_db = ['python', 'java', 'c', 'html', 'css', 'machine learning', 'data analysis',
             'communication', 'teamwork', 'oops', 'sql', 'javascript']

# Job roles and required skills
job_roles = {
    'Frontend Developer': ['html', 'css', 'javascript', 'react'],
    'Backend Developer': ['python', 'django', 'flask', 'sql'],
    'Data Analyst': ['python', 'excel', 'sql', 'data analysis'],
    'Machine Learning Engineer': ['python', 'ml', 'numpy', 'pandas'],
    'Full Stack Developer': ['html', 'css', 'javascript', 'python', 'django'],
    'Software Engineer': ['java', 'oops', 'dsa', 'problem solving'],
    'UI/UX Designer': ['figma', 'adobe', 'creativity'],
    'Communication Executive': ['communication', 'english', 'presentation']
}


def build_resume(uploaded_file):
    # Step 1: Extract resume text
    resume_bytes = uploaded_file.read()
    resume_text = extract_text(io.BytesIO(resume_bytes))

    # Step 2: Extract skills
    words = word_tokenize(resume_text.lower())
    filtered_words = [w for w in words if w.isalnum() and w not in stopwords.words('english')]
    extracted_skills = list(set(filtered_words) & set(skills_db))

    # Step 3: Predict best matching job role
    best_match = None
    max_match_count = 0
    for role, required_skills in job_roles.items():
        match_count = len(set(extracted_skills) & set(required_skills))
        if match_count > max_match_count:
            max_match_count = match_count
            best_match = role

    # Step 4: Resume rating
    expected_sections = ['objective', 'education', 'skills', 'experience',
                         'projects', 'certifications', 'contact']
    expected_keywords = ['python', 'java', 'ml', 'data', 'project',
                         'communication', 'teamwork', 'leadership']
    section_score = sum([1 for section in expected_sections if section in resume_text.lower()])
    keyword_score = sum([1 for word in expected_keywords if word in resume_text.lower()])
    total_score = section_score + keyword_score
    resume_rating = round((total_score / 15) * 10, 1)

    # Step 5: Generate tips
    improvement_tips = tips.generate_tips(resume_text)

    # Final result dictionary
    return {
        "Extracted Skills": extracted_skills,
        "Predicted Job Role": best_match if best_match else "No suitable role found.",
        "Resume Rating (out of 10)": resume_rating,
        "Improvement Tips": improvement_tips
    }
