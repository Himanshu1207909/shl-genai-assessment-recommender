import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_excel("Gen_AI shl Dataset.xlsx")
    queries = df["Query"].astype(str).tolist()

    assessment_corpus = set()
    for urls in df["Assessment_url"]:
        for u in str(urls).split(","):
            assessment_corpus.add(u.strip())

    return queries, sorted(list(assessment_corpus))


@st.cache_resource
def load_model_and_embeddings(queries, assessment_corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embeddings = model.encode(queries, convert_to_numpy=True)
    assessment_embeddings = model.encode(assessment_corpus, convert_to_numpy=True)
    return model, query_embeddings, assessment_embeddings


st.set_page_config(page_title="SHL GenAI Assessment Recommender")
st.title("🔍 SHL GenAI Assessment Recommendation System")

st.write("Enter a hiring query or job description")

query_input = st.text_area(
    "Hiring Query",
    placeholder="Hiring Java developers with backend experience"
)

top_k = st.slider("Number of recommendations", 3, 10, 5)

queries, assessment_corpus = load_data()
model, query_embeddings, assessment_embeddings = load_model_and_embeddings(
    queries, assessment_corpus
)

if st.button("Recommend"):
    if query_input.strip() == "":
        st.warning("Please enter a query")
    else:
        q_emb = model.encode([query_input], convert_to_numpy=True)
        scores = cosine_similarity(q_emb, assessment_embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]

        st.subheader("Recommended SHL Assessments")
        for i in top_idx:
            st.write(assessment_corpus[i])
