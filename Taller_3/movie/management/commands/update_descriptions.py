import os
from huggingface_hub import InferenceClient
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Update movie descriptions using a Hugging Face text-generation model"

    def handle(self, *args, **kwargs):
        # Load environment variables from the .env file
        load_dotenv('../openAI.env')

        # Initialize the Hugging Face inference client with the access token
        client = InferenceClient(
            token=os.environ.get('hf_token'),
        )

        # Helper function to send prompt and get completion from Hugging Face
        def get_completion(prompt, model="meta-llama/Llama-3.1-8B-Instruct"):
            messages = [{"role": "user", "content": prompt}]
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=400,
                temperature=0,  # No creativity, deterministic response
            )
            return response.choices[0].message.content.strip()

        # Instruction to guide the AI response (clear, concise, with genre info)
        instruction = (
            "Vas a actuar como un aficionado del cine que sabe describir de forma clara, "
            "concisa y precisa cualquier película en menos de 200 palabras. La descripción "
            "debe incluir el género de la película y cualquier información adicional que sirva "
            "para crear un sistema de recomendación."
        )

        # Fetch all movies from the database
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        # Process each movie
        for movie in movies:
            self.stdout.write(f"Processing: {movie.title}")
            try:
                # Construct the prompt combining the instruction and the current description
                prompt = (
                    f"{instruction} "
                    f"Vas a actualizar la descripción '{movie.description}' de la película '{movie.title}'."
                )

                updated_description = get_completion(prompt)

                # Save the new description to the database
                movie.description = updated_description[:1500]
                movie.save()

                self.stdout.write(self.style.SUCCESS(f"Updated: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {str(e)}")

            # NO DEBES QUITAR EL BREAK: solo se actualiza la primera película para evitar
            # un consumo elevado de la API.
            break
