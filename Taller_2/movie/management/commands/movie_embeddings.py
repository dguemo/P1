import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Command(BaseCommand):
    help = "Generate and store embeddings for all movies in the database"

    def handle(self, *args, **kwargs):
        # Load Hugging Face access token
        load_dotenv('../openAI.env')
        client = InferenceClient(token=os.environ.get('hf_token'))

        # Fetch all movies from the database
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in the database")

        def get_embedding(text):
            embedding = client.feature_extraction(text, model=EMBEDDING_MODEL)
            return np.array(embedding, dtype=np.float32)

        # Iterate through movies and generate embeddings
        for movie in movies:
            try:
                emb = get_embedding(movie.description)
                # Store embedding as binary in the database
                movie.emb = emb.tobytes()
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"Embedding stored for: {movie.title}"))
            except Exception as e:
                self.stderr.write(f"Failed to generate embedding for {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS("Finished generating embeddings for all movies"))
