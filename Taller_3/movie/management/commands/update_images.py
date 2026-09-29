import os
from huggingface_hub import InferenceClient
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate images with a Hugging Face text-to-image model and update movie image field"

    def handle(self, *args, **kwargs):
        # Load environment variables from the .env file
        load_dotenv('../openAI.env')

        # Initialize the Hugging Face inference client with the access token
        client = InferenceClient(
            token=os.environ.get('hf_token'),
        )
        # Folder to save images
        images_folder = 'media/movie/images/'
        os.makedirs(images_folder, exist_ok=True)

        # Fetch all movies
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        for movie in movies:
            try:
                # Call the helper function
                image_relative_path = self.generate_and_download_image(client, movie.title, images_folder)

                # Update database
                movie.image = image_relative_path
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"Saved and updated image for: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {e}")

            # Process just the first movie for demonstration.
            # NO DEBES QUITAR EL BREAK: las imágenes de todas las películas ya fueron
            # generadas y se cargan con el comando update_images_from_folder.
            break

        self.stdout.write(self.style.SUCCESS("Process finished (only first movie updated)."))

    def generate_and_download_image(self, client, movie_title, save_folder):
        """
        Generates an image using a Hugging Face text-to-image model.
        Returns the relative image path or raises an exception.
        """
        prompt = f"Movie poster of {movie_title}"

        # Generate image with Hugging Face (FLUX.1-schnell)
        image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")

        # Prepare the filename and full save path
        image_filename = f"m_{movie_title}.png"
        image_path_full = os.path.join(save_folder, image_filename)

        # Save the image
        image.save(image_path_full)

        # Return relative path to be saved in the DB
        return os.path.join('movie/images', image_filename)
