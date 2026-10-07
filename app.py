import streamlit as st

from core.interaction import handle_query

st.set_page_config(page_title="Basic RAG Project", page_icon="📄")

st.title("Basic RAG Project")
st.caption("Upload optional — new documents get added to the knowledge base.")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "html", "htm", "txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

question = st.text_input("Ask a question about your documents")

if st.button("Submit", type="primary"):
    result = handle_query(uploaded_files or [], question)
    st.markdown("### Response")
    for warning in result.warnings:
        st.warning(warning)
    st.write(result.text)
    if result.steps:
        with st.expander(f"Agent steps ({len(result.steps)})"):
            for i, step in enumerate(result.steps, start=1):
                args = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
                st.markdown(f"{i}. `{step.tool}({args})` → {step.summary}")
