import streamlit as st
import json
import os

USER_FILE = "users.json"

# Load existing users
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

# Save users
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def login():
    st.sidebar.title("🔐 Login or Register")
    action = st.sidebar.radio("Choose Action", ["Login", "Register"])

    users = load_users()

    if action == "Login":
        email = st.sidebar.text_input("📧 Email")
        password = st.sidebar.text_input("🔑 Password", type="password")

        if st.sidebar.button("Login"):
            if email in users and users[email]["password"] == password:
                st.session_state['logged_in'] = True
                st.session_state['username'] = users[email]["name"]
                
            else:
                st.session_state['logged_in'] = False
                st.session_state['username'] = ""
                st.error("Invalid email or password ❌")

    elif action == "Register":
        name = st.sidebar.text_input("👤 Full Name")
        email = st.sidebar.text_input("📧 Email")
        password = st.sidebar.text_input("🔑 Password", type="password")

        if st.sidebar.button("Register"):
            if email in users:
                st.warning("This email is already registered. Please login.")
            else:
                users[email] = {"name": name, "password": password}
                save_users(users)
                st.success("🎉 Registered successfully! Now you can login.")

    return st.session_state.get('logged_in', False)
