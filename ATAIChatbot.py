import configparser
import os
import sys
import pickle

module_dir = os.path.abspath("C:\\Users\\z004re3w\\Documents\\ATAIChatbot\\speakeasy-python-client-library\\")
sys.path.append(module_dir)

from typing import List
import time
import locale

from SparQLTask import SparQLTask
from EmbeddingTask import EmbeddingTask
from speakeasypy import Speakeasy, Chatroom
from MovieRecommendation import MovieRecommendation
from Crowdsourcing import Crowdsourcing
from Multimedia import Multimedia
import rdflib
from hot_reload import HotReloader
from concurrent.futures import ThreadPoolExecutor
from llama_handler import LlamaHandler

_ = locale.setlocale(locale.LC_ALL, '')
#init_notebook_mode(connected=True)

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2


class Agent:
    def __init__(self, username, password):
        print("Starting agent initialization...")
        
        # Initialize the graph
        self.graph = self._initialize_graph()
        
        # Initialize Llama handler with desired model size
        print("Loading Llama model...")
        try:
            self.llm = LlamaHandler(model_size="3.2-3b")  # Versuche zuerst Llama 3.2
        except Exception as e:
            print(f"Failed to load Llama 3.2, trying Llama 3.1: {e}")
            try:
                self.llm = LlamaHandler(model_size="3.1-8b")  # Versuche dann Llama 3.1
            except Exception as e:
                print(f"Failed to load Llama 3.1, falling back to Llama 2: {e}")
                self.llm = LlamaHandler(model_size="2-7b")  # Fallback zu Llama 2
        
        # Initialize hot reloaders for each component with immediate initialization
        print("Initializing class instances...")
        self.sparql_reloader = HotReloader(
            "SparQLTask",
            class_name="SparQLTask",
            class_args={"graph": self.graph}
        ).get_instance()  # Sofortige Initialisierung
        
        self.embedding_reloader = HotReloader(
            "EmbeddingTask",
            class_name="EmbeddingTask",
            class_args={"graph": self.graph, "path": "./"}
        ).get_instance()  # Sofortige Initialisierung
        
        self.movie_reloader = HotReloader(
            "MovieRecommendation",
            class_name="MovieRecommendation",
            class_args={"graph": self.graph}
        ).get_instance()  # Sofortige Initialisierung
        
        self.multimedia_reloader = HotReloader(
            "Multimedia",
            class_name="Multimedia",
            class_args={"graph": self.graph}
        ).get_instance()  # Sofortige Initialisierung
        
        self.crowdsourcing_reloader = HotReloader(
            "Crowdsourcing",
            class_name="Crowdsourcing",
            class_args={}
        ).get_instance()  # Sofortige Initialisierung
        
        print("Class instances initialized!")
        
        # Initialize Speakeasy
        print(f"Connecting to Speakeasy at {DEFAULT_HOST_URL}...")
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()
        print("Agent initialization complete!")

    def _initialize_graph(self):
        """Initialize RDF graph with pickle support"""
        print("Initializing RDF graph...")
        graph = rdflib.Graph()
        pickle_path = './datasources/graph.pickle'
        
        try:
            if os.path.exists(pickle_path):
                print("Loading graph from pickle file...")
                with open(pickle_path, 'rb') as f:
                    graph = pickle.load(f)
                print(f"Graph loaded from pickle. Contains {len(graph)} triples.")
            else:
                print("Parsing graph from turtle file...")
                graph.parse('./datasources/14_graph.nt', format='turtle')
                while len(graph) == 0:
                    print("Waiting for the graph to load...")
                    time.sleep(1)
                print(f"Parsing complete. Graph contains {len(graph)} triples.")
                
                # Save graph to pickle file
                print("Saving graph to pickle file...")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(graph, f)
                print("Graph saved to pickle file.")
                
        except Exception as e:
            print(f"Error handling graph: {e}")
            raise
            
        return graph

    def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    try:
                        # Get current instances
                        sparql_task = self.sparql_reloader
                        embedding_task = self.embedding_reloader
                        movie_recommendation = self.movie_reloader
                        multimedia = self.multimedia_reloader

                        # Check if the message is empty
                        if (message.message.strip() == ""):
                            room.post_messages("Please enter a question.")

                        # Check for SPARQL query
                        elif sparql_task.is_sparql_query(message.message):
                            result = sparql_task.process_sparql_query(message.message)
                            self.logAnswer("SparQL", result)
                            room.post_messages(result)
                            
                        # Check for multimedia query
                        elif multimedia.is_multimedia_query(message.message):
                            result = multimedia.handle_multimedia_query(message.message)
                            self.logAnswer("Multimedia", result)
                            room.post_messages(result)
                            
                        # Check for recommendation query
                        elif movie_recommendation.is_recommendation_query(message.message):
                            # Check movie titles
                            movie_titles, unrecognized_titles = movie_recommendation.extract_movie_titles(message.message)
                            if not movie_titles:
                                room.post_messages("I couldn't detect any movie titles. Could you rephrase your question?")
                            else:
                                recommended_movies = movie_recommendation.suggestMovie(movie_titles, numRecommendations=5)
                                recommended_answer = ""
                                if recommended_movies:
                                    for movie, reasons, _ in recommended_movies:
                                        recommended_answer += f"Recommended: {movie} (Reason: {reasons})\n"
                                else:
                                    recommended_answer = "The following movies were found on IMDb but are not in our database:\n"
                                    recommended_answer += "\n".join(movie_titles) + "\n"

                                for title in unrecognized_titles:
                                    recommended_answer += f"Couldn't recognize the following movie: {title}\n"

                                self.logAnswer("Recommendation", recommended_answer)
                                room.post_messages(recommended_answer)
                            
                        # Try embedding query as primary fallback
                        else:
                            try:
                                result = embedding_task.get_embedding_answer(message.message)
                                # Prüfe ob die Antwort sinnvoll ist
                                if result and not result.startswith("I couldn't"):
                                    self.logAnswer("Embedding", result)
                                    room.post_messages(result)
                                else:
                                    # Wenn Embedding keine gute Antwort liefert, nutze Llama als zweiten Fallback
                                    llm_result = self.llm(message.message)
                                    self.logAnswer("LLM Fallback", llm_result)
                                    room.post_messages(llm_result)
                            except Exception as e:
                                print(f"Error in embedding processing: {e}")
                                # Bei Fehlern im Embedding direkt zu Llama als Fallback
                                llm_result = self.llm(message.message)
                                self.logAnswer("LLM Fallback", llm_result)
                                room.post_messages(llm_result)
                            
                    except Exception as e:
                        print(f"Error processing query: {e}")
                        # Fallback to Llama even in case of errors
                        try:
                            result = self.llm(
                                f"I encountered an error processing your request. Here's my best attempt to help: {message.message}"
                            )
                            room.post_messages(result)
                        except:
                            room.post_messages("Unfortunately, I cannot provide an answer right now.")
                    
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.post_messages(f"Thank you for your reaction!")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

    def logAnswer(self, answer_type, answer):
        print(
        f"\t- Answer type: {answer_type} "
        f"- Answer: {answer} "
        f"- {self.get_time()}")
    

if __name__ == '__main__':
    try:
        config = configparser.ConfigParser()
        config.read("./credentials.txt")
        username = config.get('credentials', 'username')
        password = config.get('credentials', 'password')
        
        demo_bot = Agent(username, password)
        demo_bot.listen()
        
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {str(e)}")
        # Fügen Sie eine Pause hinzu, damit Sie die Fehlermeldung lesen können
        input("Drücken Sie Enter zum Beenden...")
    

