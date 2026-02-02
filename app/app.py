import streamlit as st

st.set_page_config(page_title="MicroBuild", layout="wide")

st.title("MicroBuild Template")
st.caption("Ship fast: demo + screenshot + README + 5 bullets.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    problem = st.text_input("Problem (1 line)", placeholder="e.g., Turn a job description into ATS keywords + match score")
    user = st.text_input("User", placeholder="e.g., Job-seeking PM")
    context = st.text_area("Context / Data", height=220, placeholder="Paste JD / notes / text here...")

with col2:
    st.subheader("Output")
    st.write("**Generated Prompt (copy/paste into ChatGPT):**")
    prompt = f"""
You are an AI Product Manager assistant.

Problem: {problem}
User: {user}

Task:
1) Produce the best possible output for this problem.
2) Provide 5 bullets: Problem → User → Approach → Risks → Next step.
3) Keep it concise.

Context:
{context}
""".strip()
    st.code(prompt, language="markdown")
    st.download_button("Download prompt.txt", data=prompt, file_name="prompt.txt")
