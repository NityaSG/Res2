import streamlit as st
import json
from openai import OpenAI
# Initialize Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
# pc = Pinecone(api_key="5897e1f1-3d4d-497a-a214-8fc25cf7eb74")
st.write(os.getenv("PINECONE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("res")
client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)
# Function to get vector representation of the job description (Placeholder)
def get_vector_from_description(description):
    """
    Convert the job description into a vector.
    This is a placeholder function. You need to replace it with the actual logic
    that converts the job description into a vector using your preferred model.
    """
    response = client.embeddings.create(
        input=description,
        model="text-embedding-ada-002"
)
    # Placeholder vector, replace with actual vector generation logic
    return response.data[0].embedding

# Page Config
#st.set_page_config(page_title="Candidate Search", page_icon="🔍")

st.title("Candidate Search")

# Form for Job Description
with st.form("job_description_form"):
    st.subheader("Enter Job Description")
    job_description = st.text_area("Job Description", help="Enter the job description for which you are seeking candidates.")
    top_k = st.number_input("Number of Candidates to Return", min_value=1, value=5, step=1)
    submit_button = st.form_submit_button("Search Candidates")

if submit_button:
    # Convert job description to vector
    job_vector = get_vector_from_description(job_description)
    
    # Query Pinecone Index
    query_results = index.query(
        vector=job_vector,
        top_k=top_k,
        include_metadata=True
    )
    print(query_results)
    # Display Results
    if query_results["matches"]:
        st.subheader("Top Matching Candidates")
        for match in query_results["matches"]:
            # Adjust the fields based on your metadata structure
            st.markdown(f"**Name**: {match['metadata'].get('name', 'N/A')}, **Score**: {match['score']}")
            st.json(match['metadata'])  # Optional: display full metadata as JSON
    else:
        st.write("No matching candidates found.")
