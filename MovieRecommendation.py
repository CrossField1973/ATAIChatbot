import rdflib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from imdb import IMDb
import time
import pickle
import os


class MovieRecommendation:
    movies_df = None
    feature_matrix = None
    nlp = None
    ia = None
    graph = None

    def __init__(self):
        self.graph = rdflib.Graph()
        self.graph.parse('./14_graph.nt', format='turtle')
        while len(self.graph) == 0:
            print("Waiting for the graph to load...")
            time.sleep(1)
        print(f"Parsing complete. Graph contains {len(self.graph)} triples.")
        self.ia = IMDb()

        self.pickle_file = "./movies_data.pkl"  # File to store pickled data

        # Try loading pickled data first
        self.movies_df = self.load_pickle()

        if self.movies_df is None:
            print("Pickled data not found or outdated. Loading from RDF graph...")

            # Extract data and save as pickle
            self.movies_df = self.extract_movie_data_from_graph()
            self.save_pickle(self.movies_df)

        print(f"Movies DataFrame loaded with {len(self.movies_df)} rows.")

        # Handle missing columns
        if 'release_date' in self.movies_df:
            self.movies_df['release_date'] = self.movies_df['release_date'].apply(self.extract_year)
        else:
            self.movies_df['release_date'] = None  # Default value for missing release dates

        if 'plot_keywords' not in self.movies_df:
            self.movies_df['plot_keywords'] = ''  # Default empty string if missing
        self.movies_df['plot_keywords'] = self.movies_df['plot_keywords'].fillna('').apply(
            lambda x: x.lower().replace(",", " ")
        )

        # Combine features into a single string for vectorization
        self.movies_df['combined_features'] = self.movies_df.apply(
            lambda row: f"{row['genres']} {row['directors']} {row['actors']} {row['release_date']} {row['plot_keywords']}",
            axis=1
        )

        # Vectorize combined features
        vectorizer = TfidfVectorizer()
        self.feature_matrix = vectorizer.fit_transform(self.movies_df['combined_features'])

        # Load spaCy's English model
        self.nlp = spacy.load("en_core_web_lg")

    def extract_year(self, release_date):
        """
        Extract the year from the release_date field.
        Args:
        - release_date (str): A string representing the release date (e.g., "2001-01-01").
        
        Returns:
        - int or None: The year as an integer, or None if the date is invalid.
        """
        try:
            return int(str(release_date)[:4])
        except (ValueError, TypeError):
            return None

    def load_pickle(self):
        """
        Load the pickled movie DataFrame if it exists.
        Returns:
        - pd.DataFrame or None: The pickled DataFrame if available and valid, otherwise None.
        """
        if os.path.exists(self.pickle_file):
            try:
                with open(self.pickle_file, 'rb') as f:
                    movies_df = pickle.load(f)
                print("Pickled data loaded successfully.")
                return movies_df
            except Exception as e:
                print(f"Error loading pickled data: {e}")
                return None
        else:
            print("Pickle file not found.")
            return None

    def save_pickle(self, df):
        """
        Save the movie DataFrame to a pickle file.
        Args:
        - df (pd.DataFrame): The DataFrame to pickle.
        """
        try:
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(df, f)
            print("Pickled data saved successfully.")
        except Exception as e:
            print(f"Error saving pickled data: {e}")

    def extract_movie_data_from_graph(self):
        """
        Extract movie data (title, genre, director, year, actors) from the RDF graph.
        
        Returns:
        - pd.DataFrame: A DataFrame containing movie data.
        """
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX ns1: <http://www.wikidata.org/prop/direct/>
        PREFIX ns2: <http://schema.org/>

        SELECT ?movie ?title ?description ?imdb_id ?release_date
            (GROUP_CONCAT(DISTINCT ?genre_value; separator=", ") AS ?genres) 
            (GROUP_CONCAT(DISTINCT ?actor_value; separator=", ") AS ?actors) 
            (GROUP_CONCAT(DISTINCT ?director_value; separator=", ") AS ?directors)
            (GROUP_CONCAT(DISTINCT ?keyword_value; separator=", ") AS ?plot_keywords)
        WHERE {
            ?movie ns1:P31 <http://www.wikidata.org/entity/Q11424> ;
                rdfs:label ?title ;
                ns2:description ?description .
            
            # Genres
            OPTIONAL { ?movie ns1:P136 ?genre. }
            BIND(COALESCE(?genre, "") AS ?genre_value)
            
            # Actors
            OPTIONAL { ?movie ns1:P161 ?actor. }
            BIND(COALESCE(?actor, "") AS ?actor_value)

            # IMDB ID
            OPTIONAL { ?movie ns1:P345 ?imdb_id. }
            
            # Release Date
            OPTIONAL { ?movie ns1:P577 ?release_date. }
            
            # Directors
            OPTIONAL { ?movie ns1:P57 ?director. }
            BIND(COALESCE(?director, "") AS ?director_value)
            
            # Keywords
            OPTIONAL { ?movie ns1:P921 ?keyword. }
            BIND(COALESCE(?keyword, "") AS ?keyword_value)

            FILTER (LANG(?title) = "en")
            FILTER (LANG(?description) = "en")
        }
        GROUP BY ?movie ?title ?description ?imdb_id ?release_date

        """

        # Convert results into a DataFrame
        try:
            results = self.graph.query(query)
            print(f"Query returned {len(results)} results.")

            # Convert results to DataFrame
            movies_data = []
            for row in results:
                movies_data.append({
                    'movie': str(row.movie),
                    'title': str(row.title),
                    'description': str(row.description),
                    'imdb_id': str(row.imdb_id) if row.imdb_id else None,
                    'release_date': row.release_date if row.release_date else None,
                    'genres': str(row.genres) if row.genres else '',
                    'actors': str(row.actors) if row.actors else '',
                    'directors': str(row.directors) if row.directors else '',
                    'plot_keywords': str(row.plot_keywords) if row.plot_keywords else ''
                })

            if not movies_data:
                print("No data found in the query results.")
                return pd.DataFrame()

            return pd.DataFrame(movies_data)
        except Exception as e:
            print(f"Error executing SPARQL query: {e}")
            return pd.DataFrame()

        
    def is_recommendation_query(self, message: str) -> bool:
        # Keywords indicating a recommendation query
        recommendation_keywords = [
            "recommend", "suggest", "movie", "film", "watch", "recommendation", "suggestion"
        ]
        return any(word in message.lower() for word in recommendation_keywords)

    def extract_movie_titles(self, message):
        """
        Use Named Entity Recognition (NER) to extract movie titles from the user's query.
        
        Args:
        - query (str): The natural language input from the user.
        
        Returns:
        - list: A list of movie titles found in the query.
        """

        # Extract potential movie titles with spaCy
        doc = self.nlp(message)
        movie_titles = []
        unrecognized_entities = []
        for ent in doc.ents:
            # Validate candidates as movies using IMDbPY
            results = self.ia.search_movie(ent.text)
            for movie in results[:1]:  # Limit to the top result for speed
                if ent.text.lower() in movie["title"].lower():
                    movie_titles.append(movie["title"])
                else:
                    unrecognized_entities.append(ent.text)
        return movie_titles, unrecognized_entities
    

    def suggestMovie(self, movie_titles, numRecommendations=3, similarity_threshold=0.3):
        # Find indices of provided movies
        movie_indices = []
        for title in movie_titles:
            matching_indices = self.movies_df[self.movies_df['title'].str.contains(title, case=False)].index
            if matching_indices.size > 0:
                movie_indices.append(matching_indices[0])
            else:
                print(f"Movie '{title}' not found in dataset. Skipping.")
        
        if not movie_indices:
            # Fallback: Provide recommendations based on general popular genres
            print("No matching movies found. Providing fallback recommendations...")
            fallback_recommendations = self.movies_df.sample(numRecommendations)[['title', 'genres']].to_dict(orient='records')
            return [(rec['title'], "general fallback") for rec in fallback_recommendations]
        
        # Compute similarity scores
        similarity_scores = cosine_similarity(self.feature_matrix[movie_indices], self.feature_matrix)
        aggregated_scores = similarity_scores.mean(axis=0)  # Normalize by number of input movies
        
        # Filter and rank recommendations
        similar_movies = [
            (self.movies_df.iloc[i]['title'], aggregated_scores[i])
            for i in range(len(aggregated_scores))
            if i not in movie_indices and aggregated_scores[i] > similarity_threshold
        ]
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        
        # Add reasons and enforce diversity
        recommendations = []
        genre_weight = 0.6  # Increased
        director_weight = 0.3
        actor_weight = 0.1  # Reduced
        year_weight = 0.05  # Significantly reduced
        keyword_weight = 0.7  # Strongest indicator

        input_genres = set(', '.join(self.movies_df.loc[movie_indices, 'genres'].dropna()).split(", "))
        input_keywords = set(', '.join(self.movies_df.loc[movie_indices, 'plot_keywords'].dropna()).split(", "))
        
        for movie, score in similar_movies:
            reasons = []
            weighted_score = 0
            
            # Compare genres
            recommended_genres = set(self.movies_df[self.movies_df['title'] == movie]['genres'].iloc[0].split(", "))
            genre_overlap = input_genres.intersection(recommended_genres)
            if genre_overlap:
                reasons.append("similar genres")
                weighted_score += len(genre_overlap) * genre_weight
            else:
                continue  # Skip recommendations without genre overlap

            # Compare directors and actors
            for feature, weight in zip(['directors', 'actors'], [director_weight, actor_weight]):
                input_features = set(', '.join(self.movies_df.loc[movie_indices, feature].dropna()).split(", "))
                recommended_features = set(self.movies_df[self.movies_df['title'] == movie][feature].iloc[0].split(", "))
                overlap = input_features.intersection(recommended_features)
                if overlap:
                    reasons.append(f"similar {feature}")
                    weighted_score += len(overlap) * weight
            
            # Compare plot keywords
            recommended_keywords = set(self.movies_df[self.movies_df['title'] == movie]['plot_keywords'].iloc[0].split(", "))
            keyword_overlap = input_keywords.intersection(recommended_keywords)
            if keyword_overlap:
                reasons.append("similar keywords")
                weighted_score += len(keyword_overlap) * keyword_weight
            else:
                continue  # Skip recommendations without keyword overlap

            # Year proximity
            input_years = self.movies_df.loc[movie_indices, 'release_date'].dropna().astype(int).values
            recommended_year = int(self.movies_df[self.movies_df['title'] == movie]['release_date'].values[0])
            if any(abs(recommended_year - year) <= 3 for year in input_years):
                reasons.append("close year")
                weighted_score += year_weight
            
            # Add if reasons exist and sort by weighted score
            if reasons:
                recommendations.append((movie, ', '.join(reasons), weighted_score))
            
            if len(recommendations) >= numRecommendations:
                break
        
        # Sort final recommendations by weighted score
        recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)
        
        return recommendations