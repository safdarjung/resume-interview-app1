#!/bin/bash

# Set variables for your repository name and commit message.
REPO_NAME="resume-interview-app"
COMMIT_MSG="Initial commit: Add app.py, requirements.txt, and README.md"

# Create a new directory for the app and navigate into it.
mkdir "$REPO_NAME"
cd "$REPO_NAME" || exit

# Create app.py with the full Streamlit app code.
cat << 'EOF' > app.py
"""
This is a Streamlit app for a dynamic resume-based interview simulator.
Deploy this app on Streamlit Community Cloud for free.

Instructions:
1. Upload your resume (PDF) via the sidebar.
2. Answer the dynamic interview questions generated based on your resume.
3. Refine your answers based on AI evaluation and clarifications.
4. Once satisfied, proceed to the next question.
"""

import streamlit as st
import requests
import json
import PyPDF2

# ----------------------------
# Configuration and Constants
# ----------------------------
# YOUR_SITE_URL = "https://your-site.com"
YOUR_SITE_NAME = "YourSiteName"

# Model identifiers (using Gemini for dynamic question generation and clarification)
GEMINI_MODEL = "google/gemini-2.0-flash-thinking-exp:free"
QWEN_MODEL = "qwen/qwen-vl-plus:free"

# ----------------------------
# Helper Functions
# ----------------------------
def evaluate_with_model(model_name, answer, question):
    prompt = (
        f"Evaluate the candidate's answer to the interview question with a focus on practical skills, project experience, "
        f"and clarity of the core concepts. Consider that the candidate may describe hands-on experiences differently. \n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Please provide two scores out of 100:\n"
        f"1. Practical Skills and Project Experience: How well does the candidate demonstrate hands-on abilities?\n"
        f"2. Clarity and Depth of Concept Explanation: How clearly and thoroughly does the candidate explain the concepts?\n\n"
        f"Briefly mention any areas for improvement."
    )
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        evaluation = result.get("choices", [{}])[0].get("message", {}).get("content", "No evaluation provided.")
    except Exception as e:
        evaluation = f"Error during evaluation with {model_name}: {e}"
    
    return evaluation

def evaluate_answer(answer, question, selected_model):
    if selected_model == "Gemini":
        st.info("Calling Gemini model for evaluation...")
        evaluation = evaluate_with_model(GEMINI_MODEL, answer, question)
        return f"**Evaluation from Gemini:**\n{evaluation}"
    elif selected_model == "Qwen":
        st.info("Calling Qwen model for evaluation...")
        evaluation = evaluate_with_model(QWEN_MODEL, answer, question)
        return f"**Evaluation from Qwen:**\n{evaluation}"
    else:
        st.info("Calling Gemini model for evaluation...")
        evaluation_gemini = evaluate_with_model(GEMINI_MODEL, answer, question)
        st.info("Calling Qwen model for evaluation...")
        evaluation_qwen = evaluate_with_model(QWEN_MODEL, answer, question)
        combined_evaluation = (
            f"**Evaluation from Gemini:**\n{evaluation_gemini}\n\n"
            f"**Evaluation from Qwen:**\n{evaluation_qwen}"
        )
        return combined_evaluation

def generate_clarification(answer, question):
    prompt = (
        "You are an interviewer assisting a candidate in refining their answer. The candidate has just provided an answer to the following question. "
        "Please provide suggestions on how they can clarify, rephrase, or further elaborate their answer, highlighting areas that might need more detail. \n\n"
        f"Question: {question}\n\n"
        f"Candidate's Answer: {answer}\n\n"
        "Clarification and Suggestions:"
    )
    
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        clarification = result.get("choices", [{}])[0].get("message", {}).get("content", "No clarification provided.")
    except Exception as e:
        clarification = f"Error during clarification generation: {e}"
    return clarification

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def generate_dynamic_question(resume_text, conversation_history):
    history_text = ""
    if conversation_history:
        for i, entry in enumerate(conversation_history):
            history_text += f"Q{i+1}: {entry['question']}\nA{i+1}: {entry['answer']}\n"
    else:
        history_text = "No previous conversation."
    
    prompt = (
        "You are an interviewer. Given the candidate's resume and the conversation so far, "
        "please generate a dynamic, context-specific interview question that probes the candidate's hands-on skills "
        "and understanding of key concepts mentioned in the resume. Avoid generic questions and ensure the question "
        "is relevant to the candidate's background.\n\n"
        f"Candidate's Resume:\n{resume_text}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        "Interview Question:"
    )
    
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        question = result.get("choices", [{}])[0].get("message", {}).get("content", "No question generated.")
    except Exception as e:
        question = f"Error generating question: {e}"
    return question

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("Dynamic Resume-Based Interview Simulator")
    st.write(
        """
        This application simulates an interview that adapts dynamically to your resume.
        The system will generate interview questions based on your background and the conversation history.
        After submitting your answer, you can ask for further clarification or choose to move to the next question.
        """
    )
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Both"
    if "round" not in st.session_state:
        st.session_state.round = 0
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "current_evaluation" not in st.session_state:
        st.session_state.current_evaluation = ""
    if "current_clarification" not in st.session_state:
        st.session_state.current_clarification = ""

    st.sidebar.header("Model Selection")
    st.session_state.selected_model = st.sidebar.selectbox(
        "Choose the model(s) for evaluation:",
        ("Gemini", "Qwen", "Both")
    )
    
    st.sidebar.header("Upload Your Resume PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf", key="resume_uploader")
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.session_state.resume_text = resume_text
        st.sidebar.success("Resume uploaded and processed!")
    
    if st.session_state.resume_text == "":
        st.info("Please upload your resume PDF from the sidebar to begin the interview.")
        return

    MAX_ROUNDS = 5
    if st.session_state.round >= MAX_ROUNDS:
        st.header("Interview Completed")
        st.write("Thank you for participating in the interview. Below is your performance report:")
        report = ""
        for i, entry in enumerate(st.session_state.conversation_history):
            report += f"### Round {i+1}\n"
            report += f"**Question:** {entry['question']}\n\n"
            report += f"**Your Answer:** {entry['answer']}\n\n"
            report += f"**Evaluation:** {entry['evaluation']}\n\n"
            if entry.get("clarification"):
                report += f"**Clarification:** {entry['clarification']}\n\n"
            report += "---\n\n"
        st.markdown(report)
        return

    if st.session_state.current_question == "":
        st.session_state.current_question = generate_dynamic_question(
            st.session_state.resume_text, st.session_state.conversation_history
        )
    
    st.header(f"Interview Round {st.session_state.round + 1}")
    st.write(st.session_state.current_question)
    
    candidate_answer = st.text_area("Your Answer:", key=f"answer_{st.session_state.round}")
    
    if not st.session_state.awaiting_confirmation and st.button("Submit Answer", key=f"submit_{st.session_state.round}"):
        if candidate_answer.strip() == "":
            st.warning("Please enter your answer before submitting.")
        else:
            st.info("Evaluating your answer... Please wait.")
            evaluation = evaluate_answer(candidate_answer, st.session_state.current_question, st.session_state.selected_model)
            st.session_state.current_evaluation = evaluation
            st.session_state.current_answer = candidate_answer
            st.session_state.awaiting_confirmation = True
            st.success("Evaluation completed!")
            st.markdown(evaluation)
    
    if st.session_state.awaiting_confirmation:
        if st.button("Need More Clarification", key=f"clarify_{st.session_state.round}"):
            clarification = generate_clarification(st.session_state.current_answer, st.session_state.current_question)
            st.session_state.current_clarification = clarification
            st.info("Clarification / Suggestions:")
            st.markdown(clarification)
        if st.button("Proceed to Next Question", key=f"next_{st.session_state.round}"):
            st.session_state.conversation_history.append({
                "question": st.session_state.current_question,
                "answer": st.session_state.current_answer,
                "evaluation": st.session_state.current_evaluation,
                "clarification": st.session_state.current_clarification
            })
            st.session_state.round += 1
            st.session_state.awaiting_confirmation = False
            st.session_state.current_question = ""
            st.session_state.current_answer = ""
            st.session_state.current_evaluation = ""
            st.session_state.current_clarification = ""
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.button("Next", key="next_button", on_click=lambda: None)

if __name__ == "__main__":
    main()
EOF

# Create requirements.txt file.
cat << 'EOF' > requirements.txt
streamlit
requests
PyPDF2
EOF

# Create a simple README.md file.
cat << 'EOF' > README.md
# Resume Interview App

This is a dynamic resume-based interview simulator built with Streamlit. The app generates interview questions based on your resume and allows you to refine your answers with AI-based evaluations and clarifications.

## How to Use

1. **Upload Your Resume:** Use the sidebar to upload your resume in PDF format.
2. **Answer Questions:** The app generates dynamic interview questions tailored to your resume.
3. **Receive Feedback:** Submit your answer to get AI-based evaluation and clarification suggestions.
4. **Proceed:** Once satisfied, move on to the next question.

## Deployment

You can deploy this app for free on [Streamlit Community Cloud](https://share.streamlit.io/).

## Requirements

- Python 3.x
- Streamlit
- Requests
- PyPDF2

For more details, see [Streamlit Documentation](https://docs.streamlit.io/).
EOF

# Initialize a new Git repository.
git init

# Add files and commit.
git add .
git commit -m "$COMMIT_MSG"

# Create a new public GitHub repository and push the code.
# This uses the GitHub CLI. Ensure it's installed and you're logged in (gh auth login).
gh repo create "$REPO_NAME" --public --source=. --remote=origin --push

echo "Repository 'https://github.com/safdarjung/$REPO_NAME' created and code pushed to GitHub."
echo "Now visit https://share.streamlit.io/, connect your GitHub account, and deploy your app from the 'resume-interview-app' repository."
