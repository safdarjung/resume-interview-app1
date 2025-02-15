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
import openai
import json
import PyPDF2

# ----------------------------
# Configuration and Constants
# ----------------------------
# Retrieve the API key from Streamlit secrets.
try:
    API_KEY = st.secrets["openrouter"]["api_key"]
except KeyError:
    st.error("API key not found in secrets. Please add an [openrouter] section with 'api_key' to .streamlit/secrets.toml.")
    API_KEY = None

YOUR_SITE_URL = "https://your-site.com"
YOUR_SITE_NAME = "YourSiteName"

# Set OpenAI settings to use OpenRouter.
openai.api_key = API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# Model identifiers
GEMINI_MODEL = "google/gemini-2.0-flash-thinking-exp:free"
QWEN_MODEL = "qwen/qwen-vl-plus:free"

# ----------------------------
# Helper Functions
# ----------------------------
def evaluate_with_model(model_name, answer, question):
    """
    Uses the OpenAI ChatCompletion endpoint to evaluate the candidate's answer.
    """
    prompt = (
        f"Evaluate the candidate's answer to the interview question with a focus on practical skills, project experience, "
        f"and clarity of the core concepts. Consider that the candidate may describe hands-on experiences differently.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Please provide two scores out of 100:\n"
        f"1. Practical Skills and Project Experience: How well does the candidate demonstrate hands-on abilities?\n"
        f"2. Clarity and Depth of Concept Explanation: How clearly and thoroughly does the candidate explain the concepts?\n\n"
        f"Briefly mention any areas for improvement."
    )
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        evaluation = completion.choices[0].message.content
    except Exception as e:
        evaluation = f"Error during evaluation with {model_name}: {e}"
    return evaluation

def evaluate_answer(answer, question, selected_model):
    """
    Evaluates the candidate's answer using the selected model(s).
    """
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
    """
    Uses the OpenAI ChatCompletion endpoint to provide clarification suggestions for the candidate's answer.
    """
    prompt = (
        "You are an interviewer assisting a candidate in refining their answer. The candidate has just provided an answer to the following question. "
        "Please provide suggestions on how they can clarify, rephrase, or further elaborate their answer, highlighting areas that might need more detail.\n\n"
        f"Question: {question}\n\n"
        f"Candidate's Answer: {answer}\n\n"
        "Clarification and Suggestions:"
    )
    try:
        completion = openai.ChatCompletion.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        clarification = completion.choices[0].message.content
    except Exception as e:
        clarification = f"Error during clarification generation: {e}"
    return clarification

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from the uploaded PDF file using PyPDF2.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def generate_dynamic_question(resume_text, conversation_history):
    """
    Generates a dynamic interview question based on the candidate's resume and conversation history.
    """
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
    try:
        completion = openai.ChatCompletion.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        question = completion.choices[0].message.content
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
    
    # Initialize session state variables.
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

    # Sidebar: Model selection.
    st.sidebar.header("Model Selection")
    st.session_state.selected_model = st.sidebar.selectbox(
        "Choose the model(s) for evaluation:",
        ("Gemini", "Qwen", "Both")
    )
    
    # Sidebar: Upload resume.
    st.sidebar.header("Upload Your Resume PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf", key="resume_uploader")
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.session_state.resume_text = resume_text
        st.sidebar.success("Resume uploaded and processed!")
    
    if st.session_state.resume_text == "":
        st.info("Please upload your resume PDF from the sidebar to begin the interview.")
        return

    # End the interview after MAX_ROUNDS.
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

    # Generate a new question if not already set.
    if st.session_state.current_question == "":
        st.session_state.current_question = generate_dynamic_question(
            st.session_state.resume_text, st.session_state.conversation_history
        )
    
    st.header(f"Interview Round {st.session_state.round + 1}")
    st.write(st.session_state.current_question)
    
    candidate_answer = st.text_area("Your Answer:", key=f"answer_{st.session_state.round}")
    
    # When the answer is submitted.
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
    
    # If awaiting confirmation.
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
