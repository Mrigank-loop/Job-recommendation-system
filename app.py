import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("💼 Job Recommendation System")

skills = st.text_input(
    "Enter your skills"
)

if skills:

    df = pd.read_csv("jobs.csv")

    data = df["Skills"].tolist()

    data.append(skills)

    cv = CountVectorizer()

    matrix = cv.fit_transform(data)

    similarity = cosine_similarity(matrix)

    scores = similarity[-1][:-1]

    df["Match Score"] = scores * 100

    recommendations = df.sort_values(
        by="Match Score",
        ascending=False
    )

    st.subheader("Recommended Jobs")

    st.dataframe(
        recommendations[
            ["Job Title", "Match Score"]
        ]
    )