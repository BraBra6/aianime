# recommender.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

anime_df = pd.read_csv("anime-dataset-cleaned.csv")
anime_df["Score"] = pd.to_numeric(anime_df["Score"], errors="coerce")
anime_df["Genres"] = anime_df["Genres"].fillna("")
anime_df["Studios"] = anime_df["Studios"].fillna("")
anime_df["Synopsis"] = anime_df["Synopsis"].fillna("")
anime_df["Genres"] = anime_df["Genres"].str.lower().str.replace(",", " ")

def combine_features(row):
    return row['Genres'] + " " + row['Studios'] + " " + row['Synopsis']

anime_df["combined_features"] = anime_df.apply(combine_features, axis=1)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(anime_df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix)

anime_df["Name_lower"] = anime_df["Name"].str.lower()
indices = pd.Series(anime_df.index, index=anime_df["Name_lower"]).drop_duplicates()

def get_recommendations(title, n=5):
    title = title.lower().strip()

    if title in indices:
        idx = indices[title]
    else:
        matches = anime_df[anime_df["Name_lower"].str.contains(title)]
        if matches.empty:
            return [f"❌ Titre '{title}' non trouvé dans le dataset."]
        elif len(matches) > 1:
            noms = matches["Name"].tolist()
            return [f"🔍 Plusieurs correspondances trouvées pour '{title}':"] + noms[:5]
        else:
            idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx].tolist()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in sim_scores:
        anime_index = i[0]
        try:
            score = float(anime_df.iloc[anime_index]["Score"])
        except:
            continue
        name = anime_df.iloc[anime_index]["Name"]
        if anime_index != idx and score >= 7:
            recommended.append(name)
        if len(recommended) == n:
            break

    return recommended if recommended else ["Aucune recommandation pertinente trouvée."]
print("Chargement du modèle terminé")