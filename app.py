import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("💼 AI Job Recommendation System")

user_skills = st.text_input(
    "Enter your skills",
    placeholder="Python SQL Machine Learning"
)

if user_skills:

    df = pd.read_csv("jobs.csv")

    all_skills = df["Skills"].tolist()
    all_skills.append(user_skills)

    cv = CountVectorizer()

    matrix = cv.fit_transform(all_skills)

    similarity = cosine_similarity(matrix)

    scores = similarity[-1][:-1]

    df["Match Score"] = scores * 100
    df["Match Score"] = df["Match Score"].round(2)

    recommendations = df.sort_values(
        by="Match Score",
        ascending=False
    )

    st.subheader("🎯 Recommended Jobs")

    for _, row in recommendations.head(3).iterrows():

        st.markdown(f"## {row['Job Title']}")

        st.write(
            f"**Match Score:** {row['Match Score']}%"
        )

        st.write(
            f"**Salary Estimate:** {row['Salary']}"
        )

        st.write(
            f"**Description:** {row['Description']}"
        )

        required = set(
            row["Skills"].lower().split()
        )

        user = set(
            user_skills.lower().split()
        )

        missing = required - user

        st.write(
            f"**Missing Skills:** {', '.join(missing) if missing else 'None'}"
        )

        st.markdown("---")