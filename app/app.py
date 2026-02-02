import streamlit as st

st.set_page_config(page_title="MicroBuild", layout="wide")

st.title("MicroBuild Template")
st.caption("Ship fast: demo + screenshot + README + 5 bullets.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    problem = st.text_input("Problem (1 line)", value="Analyze a job description for ATS keywords, match score, and tailored resume bullets")
    user = st.text_input("User", value="Product Manager job seeker")
    context = st.text_area("Context / Data", height=220, placeholder="Paste JD / notes / text here...")
    highlights = st.text_area(
    "My Resume Highlights (paste 5–10 bullets)",
    height=160,
    placeholder="Paste your best bullets / metrics here..."
)

with col2:
    st.subheader("Output")
    st.write("**Generated Prompt (copy/paste into ChatGPT):**")
    prompt = f"""
You are an AI career assistant for a Product Manager.

Return output in this exact structure (concise):

1) Role Summary (max 3 lines): what this role really needs.
2) ATS Keyword Pack:
   - Hard skills/tools (10–15)
   - Domain/functional keywords (10–15)
   - Soft skills (6–10)
3) Match Score (0–100) + 5 reasons.
4) Gaps:
   - Missing keywords (max 10)
   - Missing experiences (max 5)
5) Tailored Resume Bullets:
   - Rewrite my highlights into 4 bullets in impact-first format.
   - Keep each bullet <= 2 lines.
6) Interview Ammo:
   - 5 likely questions for this JD
   - 1 strong story outline (Situation → Actions → Metrics → Result)

Candidate resume highlights:
{highlights}

Job description:
{context}
""".strip()
    st.code(prompt, language="markdown")
    st.download_button("Download prompt.txt", data=prompt, file_name="prompt.txt")
