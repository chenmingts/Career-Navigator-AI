import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import pinecone
import time



# Load API Key
def load_api_key():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing PINECONE_API_KEY in .env")
    return api_key


# Load Embedding Model
@st.cache_resource
def load_embed_model():
    return InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2")


# Initialize Pinecone
@st.cache_resource
def init_pinecone(api_key):
    try:
        pinecone.init(
            api_key=api_key,
            environment="us-east-1-aws"
        )
        
        timeout_seconds = 30  # <--- increased from 10 to 20 seconds
        start_time = time.time()
        while True:
            try:
                indexes = pinecone.list_indexes()
                if "career-navigator-index" in indexes:
                    break
                else:
                    if time.time() - start_time > timeout_seconds:
                        st.error("Pinecone index 'career-navigator-index' does not exist after timeout. Please check Pinecone Console.")
                        st.stop()
                    time.sleep(1)
            except Exception as e:
                if time.time() - start_time > timeout_seconds:
                    st.error(f"Pinecone connection timeout: {str(e)}")
                    st.stop()
                time.sleep(1)

    except Exception as e:
        st.error(f"Pinecone connection failed: {str(e)}")
        st.stop()







# Perform semantic search
def search_vectors(index, model, query, category, top_k=5):
    vector = model.feature_extraction(query)

    if category == "resume":
        filter_expr = {"source": {"$eq": "resume"}}
    elif category == "job":
        filter_expr = {"source": {"$eq": "job"}}
    else:
        filter_expr = None

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_expr
    )

    return results.get("matches", [])


# Format for display
def format_result(match):
    meta = match.get("metadata", {})
    return {
        "label": meta.get("label", "Unknown"),
        "source": meta.get("source", "").upper(),
        "text": meta.get("text", ""),
        "score": round(match.get("score", 0.0), 4)
    }


# Main App
def main():
    st.set_page_config(page_title="Career Navigator AI", layout="centered")
    st.title("Career Navigator AI")
    st.subheader("Search your fit: Resume ↔ Job Description")

    query = st.text_input("Enter your search query")
    category = st.radio("Select content category:", ("all", "resume", "job"), horizontal=True)
    top_k = st.slider("Number of results", 1, 10, 5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a valid query.")
            return

        with st.spinner("Searching..."):
            try:
                api_key = load_api_key()
                model = load_embed_model()

                # Initialize Pinecone client
                init_pinecone(api_key=api_key)

                # Connect to the index
                index = pinecone.Index("career-navigator-index", host="https://career-navigator-index-9i95ecm.svc.aped-4627-b74a.pinecone.io")


                matches = search_vectors(index, model, query, category, top_k)
                results = [format_result(m) for m in matches]

                if not results:
                    st.info("No results found.")
                    return

                st.success(f"Found {len(results)} result(s).")

                for r in results:
                    label_line = f"{r['label']} ({r['source'].capitalize()}, Score: {r['score']})"
                    with st.expander(label_line):
                        st.markdown(
                            f"<div style='color: white; font-size: 16px; line-height: 1.6;'>{r['text'].replace(chr(10), '<br>')}</div>",
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"Error occurred: {str(e)}")



if __name__ == "__main__":
    main()
