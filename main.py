import streamlit as st
from rag import ingest, ask

st.title("Real Estate Assistant")

st.sidebar.header("URL Inputs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()

# --- INGESTION ---
if st.sidebar.button("Process URLs"):
    urls = [u for u in [url1, url2, url3] if u != ""]
    if not urls:
        placeholder.text("❌ You must provide at least one valid URL.")
    else:
        placeholder.text("⏳ Ingesting URLs... Please wait.")
        ingest(urls)
        placeholder.text("✅ Ingestion complete! You can now ask questions.")

# --- QUERY SECTION ---
query = st.text_input("Ask your question about real estate articles:")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        try:
            answer = ask(query)
            st.subheader("Answer:")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")
