import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Command(BaseCommand):
    help = "Compare two movies and optionally a prompt using Hugging Face embeddings"

    def handle(self, *args, **kwargs):
        # Load Hugging Face access token
        load_dotenv('../openAI.env')
        client = InferenceClient(token=os.environ.get('hf_token'))

        # Cambia estos títulos por cualquier par de películas que quieras comparar
        movie1 = Movie.objects.get(title="Carmencita")
        movie2 = Movie.objects.get(title="The Sea")

        def get_embedding(text):
            embedding = client.feature_extraction(text, model=EMBEDDING_MODEL)
            return np.array(embedding, dtype=np.float32)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Generate embeddings of both movies
        emb1 = get_embedding(movie1.description)
        emb2 = get_embedding(movie2.description)

        # Compute similarity between movies
        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(f"Similitud entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}")

        # Optional: Compare against a prompt
        prompt = "película sobre marineros y el mar"
        prompt_emb = get_embedding(prompt)

        sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"Prompt: '{prompt}'")
        self.stdout.write(f"Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
        self.stdout.write(f"Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")
