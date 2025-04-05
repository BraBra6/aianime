# app.py
import gradio as gr
from recommender import get_recommendations

def recommend(title):
    results = get_recommendations(title)
    return "\n".join(results)

iface = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(placeholder="Entrez un titre d'anime, ex: Naruto"),
    outputs="text",
    title="Système de Recommandation d'Anime",
    description="Entrez un anime que vous aimez, et l'IA vous suggérera des titres similaires."
)

iface.launch(share=True, inbrowser=True)