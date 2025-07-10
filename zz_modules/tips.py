import re

# ✅ Expected sections and keywords — contact is NOT here!
expected_sections = ['education', 'skills', 'experience', 'projects', 'career objective', 'certifications']
expected_keywords = ['python', 'java', 'ml', 'data', 'project', 'communication', 'teamwork', 'leadership']

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())  # removes punctuation & symbols

def generate_tips(resume_text):
    tips = []
    cleaned_text = clean_text(resume_text)

    # 🛡️ Forcefully exclude checking for 'contact' section, even if accidentally mentioned
    force_exclude = ['contact', 'phone', 'email', 'address']

    # 🔍 Check only valid expected sections (excluding anything in force_exclude)
    for section in expected_sections:
        pattern = r'\b' + re.escape(section.lower()) + r'\b'
        if section.lower() not in force_exclude and not re.search(pattern, cleaned_text):
            tips.append(f"📌 Consider adding a **{section.capitalize()}** section.")

    # 🔍 Check how many expected keywords exist
    keyword_count = sum(1 for keyword in expected_keywords if re.search(r'\b' + keyword + r'\b', cleaned_text))
    if keyword_count < 5:
        tips.append("🧠 Add more technical and soft skills to improve keyword coverage.")

    # 💼 Specific advice
    if not re.search(r'\bproject\b', cleaned_text):
        tips.append("💼 Mention at least 1 or 2 **projects** with outcomes and technologies.")
    if not re.search(r'\bcertification\b', cleaned_text):
        tips.append("🎓 Add relevant certifications or online courses to boost credibility.")

    # 🌟 All good!
    if not tips:
        tips.append("🌟 Your resume looks well-structured! Just polish formatting if needed.")

    return tips
