# anime_recommender_netflix.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load the dataset
anime_df = pd.read_csv("animedata.csv", encoding='ISO-8859-1')  # ou 'latin1'
anime_df.dropna(subset=["Name", "Genres", "Studios", "Synopsis", "Score", "Image URL"], inplace=True)

# Prepare data
anime_df['combined_features'] = anime_df['Genres'].fillna("") + " " + anime_df['Studios'].fillna("") + " " + anime_df['Synopsis'].fillna("")
anime_df['Name_lower'] = anime_df['Name'].str.lower()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(anime_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Map titles to indices
indices = pd.Series(anime_df.index, index=anime_df['Name_lower']).drop_duplicates()

# List of known animes with broken image URLs
broken_images = [
    "bleach: thousand-year blood war", "tokyo revengers", "akudama drive", "hellÕs paradise",
    "hell's paradise", "mashle: magic and muscles", "ousama ranking", "dr. stone",
    "classroom of the elite", "death parade", "devilman crybaby", "ao ashi"
]

# Recommender function
def get_recommendations(title, n=6):
    title = title.lower().strip()
    if title not in indices:
        matches = anime_df[anime_df['Name_lower'].str.contains(title)]
        if matches.empty:
            return ["Titre introuvable"]
        idx = matches.index[0]
    else:
        idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    results = []
    for i, _ in sim_scores:
        anime = anime_df.iloc[i]
        fallback_img = 'https://via.placeholder.com/120x170?text=No+Image'
        img_url = fallback_img if anime['Name'].lower() in broken_images else anime['Image URL']
        block = f"""
        <div style='display: flex; gap: 1rem; padding: 1rem; border-bottom: 1px solid #333;'>
            <img src="{img_url}" alt="{anime['Name']}" style="width: 120px; height: auto; border-radius: 8px;" onerror="this.src='{fallback_img}'"/>
            <div>
                <h3 style="margin-bottom: 0.3rem;">{anime['Name']}</h3>
                <p style="font-style: italic; margin: 0;">{anime['Genres']}</p>
                <p style="margin-top: 0.5rem; max-width: 400px;">{anime['Synopsis'][:150]}...</p>
            </div>
        </div>
        """
        results.append(block)

    return "".join(results)

# Gradio interface
with gr.Blocks(css="body { background-color: #0f0f0f; color: white; font-family: Arial; } h1, p { color: white; }") as demo:
    gr.Markdown("""
        # 🎬 Système de Recommandation d'Animes (Style Netflix)
        Entrez un anime que vous aimez et obtenez des recommandations visuelles similaires.
    """)
    anime_input = gr.Textbox(label="Anime", placeholder="ex: Naruto, One Piece, Jujutsu Kaisen")
    output = gr.HTML()
    btn = gr.Button("Submit")

    btn.click(fn=get_recommendations, inputs=anime_input, outputs=output)

    gr.Markdown("""
    <br><br>
    💡 Ce système utilise TF-IDF et la similarité cosinus sur les descriptions, genres et studios.
    """)

demo.launch(share=True)
