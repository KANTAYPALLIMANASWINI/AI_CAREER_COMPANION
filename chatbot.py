from modules import resume_builder, missing_skills

def get_bot_response(user_input):
    user_input = user_input.lower().strip()

    # 🌸 Greetings
    if any(word in user_input for word in ["hi", "hello", "hey", "hii", "hlo", "namaste", "yo"]):
        return (
            "Hii ! I'm your AI Career Buddy 🤖✨\n"
            "Ready to support you in your resume, job hunt, and skills journey! 🌈💼"
        )

    # 🙋‍♀️ Introduction
    elif "who are you" in user_input or "what can you do" in user_input:
        return (
            "I'm your **AI Career Companion** 💼\n"
            "I can analyze your resume, suggest job roles, find missing skills, and even give learning resources 🧠📘"
        )

    # 📄 Resume Help
    elif "resume" in user_input or "upload" in user_input:
        return (
            "Sure thing! 📄 Upload your resume below, and I’ll work my magic ."
        )

    # ⭐ Resume Rating
    elif "rate" in user_input or "rating" in user_input:
        return (
            "I'll give your resume a score and also tips to improve it further "
        )

    # 📊 Skills
    elif "skills" in user_input or "my skills" in user_input:
        return (
            "Let’s explore your skills and check if anything’s missing for your dream job! "
        )

    # ❌ Missing Skills
    elif "missing" in user_input or "what i don't have" in user_input:
        return (
            "Don’t worry! I’ll scan your resume and tell you what skills you're missing 🔍"
        )

    # 🎓 Learning / Courses
    elif any(word in user_input for word in ["learn", "course", "how to improve", "improve skills"]):
        return (
            "I’ve got learning links to help you level up fast! Ready to grow? "
        )

    # 💼 Job Role
    elif any(word in user_input for word in ["job", "career", "role", "position"]):
        return (
            "Based on your skills, I’ll predict a role that suits you best "
        )

    # 📝 Resume Tips
    elif "tips" in user_input or "improve" in user_input or "advice" in user_input:
        return (
            "I'll share tips to make your resume more powerful  — layout, sections, and content "
        )

    # 🙃 Motivation
    elif any(word in user_input for word in ["sad", "tired", "lost", "confused", "bore", "fail"]):
        return (
            "Ayy  don't worry… You’re doing amazing Take a breath, smile , and let's get back on track "
        )

    # 💝 Gratitude
    elif any(word in user_input for word in ["thank you", "thanks", "thankyou", "tnx"]):
        return (
            "Awww you're most welcome\nYou're the real rockstar  Keep shining always!"
        )

    # 💡 Help
    elif "help" in user_input or "support" in user_input:
        return (
            "I'm always here for you — ask me anything about resumes, job roles, or learning "
        )



    # 🎬 Start guide
    elif "start" in user_input or "how to use" in user_input:
        return (
            "Here’s how we roll 👇\n1. Upload your resume\n2. I’ll analyze it\n3. You get job role, skill gap & tips "
        )

    # 🌈 Fallback
    else:
        return (
            "Hmm I’m not sure what you meant But I’m always ready to help!\n"
            "Ask about your resume, jobs, skills, or just say hi "
        )
