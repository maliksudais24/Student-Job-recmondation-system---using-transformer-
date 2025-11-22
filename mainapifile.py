import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class APICleaner:
    def __init__(self, api_id, api_key, results_per_page: int = 100):
        self.api_id = api_id
        self.api_key = api_key
        self.results_per_page = results_per_page

    def load_and_clean_data(self, country="us", category="it-jobs", pages=3):
        """
        Fetch a broad pool of jobs (no query).
        Default: IT jobs in the US, 3 pages.
        """
        all_jobs = []

        for page in range(1, pages + 1):
            base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
            url = f"{base_url}?app_id={self.api_id}&app_key={self.api_key}&results_per_page={self.results_per_page}&category={category}"
            
            response = requests.get(url)
            data = response.json()

            if "results" not in data:
                print(f"API Error on page {page}")
                continue

            all_jobs.extend(data["results"])

        if not all_jobs:
            return pd.DataFrame()

        df = pd.json_normalize(all_jobs)
        df.drop(
            columns=[
                "contract_time", "latitude", "longitude", "id", "adref",
                "__CLASS__", "salary_is_predicted"
            ],
            inplace=True,
            errors="ignore",
        )

        # Clean text columns
        for col in ["description", "title"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.lower()
                    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )

        # ✅ Salary cleaning (only if cols exist)
        if "salary_max" in df.columns and "salary_min" in df.columns:
            df = df[df["salary_max"].notna() & df["salary_min"].notna()]
            df = df[(df["salary_max"] > 0) & (df["salary_min"] > 0)]
            df["salary_max"] = df["salary_max"].astype(int)
            df["salary_min"] = df["salary_min"].astype(int)
            df["average_salary"] = ((df["salary_max"] + df["salary_min"]) / 2).astype(int)
            df.drop(columns=["salary_max", "salary_min"], inplace=True, errors="ignore")

        # ✅ Dates
        if "created" in df.columns:
            df["created"] = pd.to_datetime(df["created"], errors="coerce").dt.date

        # ✅ Location split
        if "location.area" in df.columns:
            df["country"] = df["location.area"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
            df["state"] = df["location.area"].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
            df["city"] = df["location.area"].apply(lambda x: x[3] if isinstance(x, list) and len(x) > 3 else None)
        df.drop(columns=["location.__CLASS__", "location.display_name", "location.area"], inplace=True, errors="ignore")

        # ✅ redirected URL
        if "redirect_url" in df.columns:
            df = df[df['redirect_url'].notnull()]
            df = df[df['redirect_url'].str.startswith('http')]

        # ✅ Company
        if "company.display_name" in df.columns:
            df["company_name"] = df["company.display_name"].fillna("").astype(str).str.strip().str.title()
        df.drop(columns=["company.__CLASS__", "company.display_name"], inplace=True, errors="ignore")

        # ✅ Category
        if "category.label" in df.columns:
            df["category"] = df["category.label"].fillna("").astype(str).str.strip().str.title()
        df.drop(columns=["category.__CLASS__", "category.label", "category.tag"], inplace=True, errors="ignore")

        self.df = df
        return self.df
     
# Example runner
if __name__ == "__main__":
    api_id = "1df8c0b9"
    api_key = "512fd34e24d8179441c513585fbbff5f"

    api_cleaner = APICleaner(api_id=api_id, api_key=api_key, results_per_page=20)
    jobs_df = api_cleaner.load_and_clean_data()

    print("✅ Job Data Preview:")
    print(jobs_df.head())
