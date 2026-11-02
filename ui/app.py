import streamlit as st

st.set_page_config(page_title='RAG Recap', layout='wide')

pages = [
    st.Page('pages/logs.py', title='Logs de conversion'),
    st.Page('pages/experiments.py', title='Expériences'),
    st.Page('pages/llm_metrics.py', title='Performances LLM'),
]

st.navigation(pages).run()
