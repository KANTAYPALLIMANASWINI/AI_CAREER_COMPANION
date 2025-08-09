import os
import nltk

# Set up nltk_data directory inside your app folder
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)

# Download punkt tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

# Add this directory to nltk data search path
nltk.data.path.append(nltk_data_dir)

import streamlit as st
from login import login
from chatbot import get_bot_response

from modules import resume_builder, missing_skills, tips
import base64

# 🌄 Background Setup Function (Perfect full cover & readable content)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #f0f2f6;
        }}

        .block-container {{
            background-color: rgba(255, 255, 255, 0.88); 
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.9);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🔐 Login check
if login():
    user_name = st.session_state.get("username")
    if user_name:
        st.success(f"You're logged in! Welcome back, {user_name}! 💖")
    else:
        st.success("You're logged in! Let's start your career journey 💖")

    # 💬 Chatbot section
    st.subheader("💬 Chat with Me")
    user_input = st.text_input("You:", "")

    if user_input:
        response = get_bot_response(user_input)
        st.text_area("Bot:", value=response, height=100)

    # 📄 Resume upload section
    st.subheader("📄 Upload Your Resume (PDF Only)")
    uploaded_file = st.file_uploader("Choose your resume file", type=["pdf"])

    if uploaded_file:
        with st.spinner("Analyzing your resume..."):
            result = resume_builder.build_resume(uploaded_file)
            st.success("Analysis complete! 💡")

            # 📊 Resume Summary
            st.subheader("📊 Resume Summary:")
            st.write(f"✅ **Extracted Skills:** {', '.join(result['Extracted Skills'])}")
            st.write(f"🎯 **Predicted Job Role:** {result['Predicted Job Role']}")
            st.write(f"⭐ **Resume Rating (Out of 10):** {result['Resume Rating (out of 10)']}")

            # ❌ Missing Skills
            missing = missing_skills.find_missing_skills(
                result["Extracted Skills"], result["Predicted Job Role"]
            )
            links = missing_skills.get_learning_links(missing)

            st.subheader("❌ Missing Skills for Your Target Role:")
            if missing:
                for skill in missing:
                    st.write(f"- {skill}")
            else:
                st.success("🎉 You have all the required skills!")

            # 🎓 Learning links
            st.subheader("🎓 Suggested Learning Resources:")
            if links:
                for skill, link in links.items():
                    st.markdown(f"🔗 **{skill.title()}**: [Click to Learn]({link})")
            else:
                st.info("No learning resources to suggest!")

            # 💡 Resume Tips
            st.subheader("💡 Tips to Improve Your Resume:")
            for tip in result["Improvement Tips"]:
                st.write("•", tip)

else:
    st.warning("Please login from the sidebar to continue 🔐")


