import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from preprocessing import vector_search


@st.cache_data
def read_data(data="data/data_curated.csv"):
    """Read the data from local."""
    return pd.read_csv(data)


@st.cache_resource()
def load_bert_model(name="paraphrase-MiniLM-L3-v2"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache_resource()
def load_faiss_index(path_to_faiss="models/faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("Vector-based searches with Sentence Transformers and Faiss")

    # User search
    user_input = st.text_area("Search box")

    # Filters
    st.sidebar.markdown("**Filters**")
    num_results = st.sidebar.slider("Number of search results", 5, 10, 15)

    # Fetch results
    if user_input:
        # Get paper IDs
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(data.index):
                l = data.loc[id_, "title"]
                f = data.loc[id_, "abstract"]
            else:
                continue

            st.write(
                f"**Paper Details**\n"
                f"**Title:** {l}\n"
                f"**Abstract:** {f}\n"
            )


if __name__ == "__main__":
    main()