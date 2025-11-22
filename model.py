import pandas as pd 
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from oop_project import studentdatavisulaizer
from mainapifile import APICleaner


# Load transformer once (efficient)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_for_student(student, jobs_df, top_n=5):
    """
    Recommend jobs for a student using transformer embeddings.
    """
    if jobs_df.empty:
        print(f" No jobs available to recommend for {student['Name']}")
        return []

    # Prepare student text
    student_text = student["student_text"]

    # Prepare job text
    jobs_df["job_text"] = (jobs_df["title"].fillna("") + " " +
                           jobs_df["description"].fillna("")).str.lower()
 
    # Encode
    student_emb = embedder.encode(student_text, convert_to_tensor=True)
    job_embs = embedder.encode(jobs_df["job_text"].tolist(), convert_to_tensor=True)

    # Similarity
    sim_scores = util.cos_sim(student_emb, job_embs)[0].cpu().numpy()

    # Pick top N jobs
    top_indices = sim_scores.argsort()[-top_n:][::-1]

    recommendations = []
    for job_idx in top_indices:
        job = jobs_df.iloc[job_idx]
        recommendations.append({
            "StudentName": student["Name"],
            "JobTitle": job["title"],
            "Company": job.get("company_name", ""),
            "Category": job.get("category", ""),
            "City": job.get("city", ""),
            "JobURL": job["redirect_url"],
            "SimilarityScore": float(sim_scores[job_idx])
        })
    return recommendations

def main():
    # 1. Load and clean student data

    student_visualizer = studentdatavisulaizer("student_real_data.csv")
    student_df = student_visualizer.df.copy()

    print("\n✅ Student Data Loaded & Cleaned")
    print(student_df[["Name", "student_text"]].head(10))

    # 2. API setup (broad fetch)

    api_id = "1df8c0b9"
    api_key = "512fd34e24d8179441c513585fbbff5f"
    job_cleaner = APICleaner(api_id=api_id, api_key=api_key, results_per_page=50)

    # Fetch broad pool of jobs
    jobs_df = job_cleaner.load_and_clean_data(category="it-jobs", pages=3)

    print(f"\n Jobs Loaded: {len(jobs_df)}")

    # 3. Generate recommendations per student
 
    all_recommendations = []
    for _, student in student_df.iterrows():
        recs = recommend_for_student(student, jobs_df, top_n=3)
        all_recommendations.extend(recs)

    rec_df = pd.DataFrame(all_recommendations)
    rec_df.to_csv("student_job_recommendations_transformer.csv", index=False)
    print("\n All student recommendations saved to student_job_recommendations_transformer.csv")


if __name__ == "__main__":
    main()