import streamlit as st
import PyPDF2
from datetime import date
from io import BytesIO
import os
from openai import OpenAI
import json 
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("res")
client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)
candidate_custom_functions = [
    {
        'name': 'extract_resume',
        'description': 'Extract the important features from the resume',
        'parameters': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                    'description': 'Name of the person'
                },
                'Experiences': {
                    'type': 'string',
                    'description': 'A detailed description of the candidates work experience'
                },
                'Degree': {
                    'type': 'string',
                    'description': 'A detailed description of the candidates college/ university studies'
                },
                'grades': {
                    'type': 'integer',
                    'description': 'GPA of the candidate'
                },
                'Skills': {
                    'type': 'string',
                    'description': 'List of skills of the candidate' 
                },
                'Languages': {
                    'type': 'string',
                    'description': 'List of languages of the candidate'

                },
                'Projects':{
                    'type': 'string',
                    'description': 'detailed description of projects of the candidate'
                },
                'Achievements':{
                    'type':'string',
                    'description':'A detailed description of the candidates achievements'
                } 
            }
        }
    }
]
# Page Config
st.set_page_config(page_title="Analyst Job Application Form", page_icon="📈")

st.title("Analyst Job Application Form")

def extract_text_from_pdf(file):
    """Extract text from the uploaded PDF."""
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(file.getvalue()))
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text() + "\n"
    return text
a=""
with st.form("job_application_form"):
    st.subheader("Personal Information")
    # Personal Information
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")

    st.subheader("Professional Background")
    # Professional Background
    latest_job_title = st.text_input("Latest Job Title")
    company = st.text_input("Company Name")
    experience_start_date = st.date_input("Start Date of Latest Position", max_value=date.today())
    experience_end_date = st.date_input("End Date of Latest Position", min_value=experience_start_date, max_value=date.today())
    years_of_experience = st.number_input("Total Years of Experience", min_value=0, format="%d")

    st.subheader("Education")
    # Education
    highest_degree = st.selectbox("Highest Degree Obtained", 
                                  ["High School", "Bachelor's", "Master's", "PhD", "Other"])
    field_of_study = st.text_input("Field of Study")
    university = st.text_input("University Name")
    education_start_date = st.date_input("Start Date of Education", max_value=date.today())
    education_end_date = st.date_input("End Date of Education", min_value=education_start_date, max_value=date.today())

    st.subheader("Skills and Certifications")
    # Skills
    skills = st.text_area("List your relevant skills")
    certifications = st.text_area("List any relevant certifications")

    st.subheader("Resume")
    # Resume Upload
    resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    # Extract text from the uploaded PDF resume
    if resume is not None:
        extracted_text = extract_text_from_pdf(resume)
        a=extracted_text
        st.text_area("Extracted Text from Resume", value=extracted_text, height=300)

    # Submit button
    submit_button = st.form_submit_button("Submit Application")

if submit_button:
    st.success("Thank you for your application. We will review it and contact you soon.")
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [{'role': 'user', 'content': a}],
        functions = candidate_custom_functions,
        function_call = 'auto'
    )
    json_response = json.loads(response.choices[0].message.function_call.arguments)
    st.json(json_response)
    response = client.embeddings.create(
        input=a,
        model="text-embedding-ada-002"
)
    index.upsert(vectors=[{"id":first_name,"values":response.data[0].embedding,"metadata":{"first_name":first_name,"last_name":last_name,"email":email,"phone":phone,"latest_job_title":latest_job_title,"years_of_experience":years_of_experience}}])
    
