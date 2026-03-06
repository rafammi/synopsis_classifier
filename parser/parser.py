import os
import requests
import time
import pandas as pd
import tqdm as tqdm


# Fetch Genres -> they're generally in form of IDS

def run_parser(API_KEY: str, start_year: int, end_year: int, max_limit: int = 50000):
    print("Getting Genre mappings...")
    genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={API_KEY}"
    genre_response = requests.get(genre_url).json()
    genre_dict = {g["id"]: g["name"] for g in genre_response["genres"]}
    print("genre_dict")

    all_movies = []

    print("Getting movies...")

    for year in range(start_year, end_year):
        page = 1
        total_pages = 1

        while page <= total_pages:
            url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}"
            params = {
                "primary_release_year": year,
                "page": page
            }

            response = requests.get(url, params=params)

            if response.status_code != 200:
                print("Error:", response.status_code)
                break

            data = response.json()

            total_pages = min(data["total_pages"], 500)

            for movie in data["results"]:
                if not movie["overview"]:
                    continue
                genres = [genre_dict[g] for g in movie["genre_ids"] if g in genre_dict]
                if not genres:
                    continue
                all_movies.append({
                    "title": movie["title"],
                    "overview": movie["overview"],
                    "genres": "|".join(genres),
                    "year": year
                })
                print(f"Movie appended: {movie['title']}")
                print(f"Movie genre: {'|'.join(genres)}")
                print(f"Current len: {len(all_movies)}")
                print("---------------------------------")
                if len(all_movies) >= max_limit:
                    break

            page += 1
            time.sleep(0.1)

            if len(all_movies) >= max_limit:
                break

        if len(all_movies) >= max_limit:
            break

    print(f"Movies parsed: {len(all_movies)}")
    print("Saving to CSV...")
    df = pd.DataFrame(all_movies)
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "movies_dataset_tmdb_2009_2020.csv")
    df.to_csv(output_path, index=False)
    print("Saved.")