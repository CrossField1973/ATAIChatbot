import json
import random
from rdflib import Namespace, Literal
from fuzzywuzzy import process
import pickle
import os

class Multimedia:
    def __init__(self, graph):
        # Define the namespace prefixes for the knowledge graph first
        self.ns1 = Namespace("http://www.wikidata.org/prop/direct/")
        self.rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        
        # Load images.json
        with open('./datasources/images.json', 'r') as file:
            self.images_data = json.load(file)
        
        self.graph = graph
        self._actor_name_to_id = {}  # Cache für Name -> ID Mapping
        self._load_actor_names()
        self._build_name_cache()  # Neue Methode
        self._cast_image_cache = {}
        self._build_image_cache()

    def _load_actor_names(self):
        """
        Lädt Schauspieler-Namen aus einer Pickle-Datei oder erstellt diese, falls sie nicht existiert.
        """
        pickle_path = './datasources/actor_names.pkl'
        
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                self._actor_names = pickle.load(f)
        else:
            # Erstelle die Pickle-Datei, wenn sie nicht existiert
            self._actor_names = [
                str(label) for label in self.graph.objects(predicate=self.rdfs.label)
            ]
            with open(pickle_path, 'wb') as f:
                pickle.dump(self._actor_names, f)

    def _build_name_cache(self):
        """
        Erstellt einen Cache für schnellere Name-zu-ID-Lookups
        """
        cache_path = './datasources/actor_name_cache.pkl'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self._actor_name_to_id = pickle.load(f)
        else:
            print("Building actor name cache...")
            for subject in self.graph.subjects(predicate=self.rdfs.label):
                for name in self.graph.objects(subject=subject, predicate=self.rdfs.label):
                    for imdb_id in self.graph.objects(subject=subject, predicate=self.ns1.P345):
                        if str(imdb_id).startswith('nm'):
                            self._actor_name_to_id[str(name)] = str(imdb_id)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(self._actor_name_to_id, f)

    def _build_image_cache(self):
        """
        Erstellt einen Cache für Schauspieler-zu-Bilder-Mapping
        """
        for entry in self.images_data:
            for cast_id in entry.get("cast", []):
                if cast_id not in self._cast_image_cache:
                    self._cast_image_cache[cast_id] = []
                self._cast_image_cache[cast_id].append(f"image:{entry['img'].replace('.jpg', '')}")

    # Function to find an image for a given cast member
    def find_cast_image(self, cast_ids, together=False):
        """
        Finds images for given cast members by IMDb IDs.
        :param cast_ids: A list of cast members' IMDb IDs.
        :param together: Boolean indicating if all IDs must appear together.
        :return: A list of image addresses prefixed with 'image:'.
        """
        if together:
            # Für together=True müssen wir weiterhin die ursprüngliche Logik verwenden
            return super().find_cast_image(cast_ids, together=True)
        
        results = []
        for cast_id in cast_ids:
            if cast_id in self._cast_image_cache:
                results.extend(self._cast_image_cache[cast_id])
        return results

    # Function to check if the query is a multimedia query
    def is_multimedia_query(self, query):
        """
        Checks if the query is about cast members' appearances.
        :param query: The user query string.
        :return: True if the query is about cast members' appearances, False otherwise.
        """
        return any(keyword in query.lower() for keyword in ["picture", "photo", "look like", "show me", "let me know"])

    def get_imdb_id(self, cast_name):
        """
        Optimierte Version mit Cache-Nutzung
        """
        return self._actor_name_to_id.get(cast_name)

    def extract_actor_names(self, query):
        """
        Optimierte Version der Namensextraktion
        """
        potential_matches = []
        words = query.split()
        
        # Erstelle alle möglichen Namenskombinationen (max. 3 Wörter)
        name_combinations = []
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                name_combinations.append(" ".join(words[i:j]))
        
        # Verwende process.extract statt extractOne für Batch-Processing
        matches = process.extract(query, self._actor_names, limit=5)
        
        for match, score in matches:
            if score > 80:
                imdb_id = self._actor_name_to_id.get(match)
                if imdb_id:
                    potential_matches.append((match, imdb_id))
        
        return potential_matches

    # Function to handle user queries
    def handle_multimedia_query(self, query):
        """
        Handles user queries about cast members' appearances.
        Prioritizes user_avatars and images with single actors.
        """
        try:
            actor_info = self.extract_actor_names(query)
            print(f"Debug - Found actor info: {actor_info}")
            
            if not actor_info:
                return "Sorry, I couldn't find any matching actors in your query."

            actor_info = [(name, id) for name, id in actor_info if id.startswith('nm')]
            
            if not actor_info:
                return "Sorry, I couldn't find any valid actor IDs."

            cast_ids = [id for _, id in actor_info]
            actor_names = {id: name for name, id in actor_info}
            print(f"Debug - Cast IDs: {cast_ids}")

            # For a single actor
            if len(cast_ids) == 1:
                cast_id = cast_ids[0]
                solo_user_avatar_images = []
                solo_other_images = []
                other_images = []
                
                # Search all images
                for entry in self.images_data:
                    if entry.get("cast") == [cast_id]:  # Only this actor
                        if entry.get("type") == "user_avatar":
                            solo_user_avatar_images.append(f"image:{entry['img'].replace('.jpg', '')}")
                        else:
                            solo_other_images.append(f"image:{entry['img'].replace('.jpg', '')}")
                    elif cast_id in entry.get("cast", []):  # Actor with others
                        if entry.get("type") == "user_avatar":
                            other_images.append(f"image:{entry['img'].replace('.jpg', '')}")
                
                # Prefer user_avatars with only this actor
                if solo_user_avatar_images:
                    return random.choice(solo_user_avatar_images)
                # Then other images with only this actor
                elif solo_other_images:
                    return random.choice(solo_other_images)
                # Then user_avatars with multiple actors
                elif other_images:
                    return random.choice(other_images)
                # Fallback to all available images
                elif cast_id in self._cast_image_cache:
                    return random.choice(self._cast_image_cache[cast_id])
                return f"Sorry, I couldn't find any images for {actor_names[cast_id]}."
            
            # For multiple actors
            individual_images = []
            for cast_id in cast_ids[:3]:
                try:
                    # First search for user_avatars
                    user_avatar_images = [
                        f"image:{entry['img'].replace('.jpg', '')}"
                        for entry in self.images_data
                        if cast_id in entry.get("cast", []) and entry.get("type") == "user_avatar"
                    ]
                    
                    if user_avatar_images:
                        individual_images.append(random.choice(user_avatar_images))
                    else:
                        # Fallback to other images
                        actor_images = self._cast_image_cache.get(cast_id, [])
                        if actor_images:
                            individual_images.append(random.choice(actor_images))
                        else:
                            return f"Sorry, I couldn't find any images for {actor_names[cast_id]}."
                except Exception as img_error:
                    print(f"Debug - Error finding images for {cast_id}: {str(img_error)}")
                    continue

            if individual_images:
                return " ".join(individual_images)
            else:
                return "Sorry, I couldn't find any matching images."
            
        except Exception as e:
            print(f"Debug - Error in handle_multimedia_query: {str(e)}")
            return f"Sorry, an error occurred while processing your request: {str(e)}"
