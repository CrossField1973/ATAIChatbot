import configparser
import os
import sys
import pickle
import argparse

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
from llama_handler import LlamaHandler

_ = locale.setlocale(locale.LC_ALL, '')
#init_notebook_mode(connected=True)

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2


class Agent:
    def __init__(self, username, password, debug=False, disable: []=[]):
        self.debug = debug
        
        print("Starting agent initialization...")
        
        # Initialize the graph
        self.graph = self._initialize_graph()
        
        # Initialize Llama handler with desired model size
        print("Loading Llama model...")
        try:
            self.llm = LlamaHandler(model_size="2-7b")
        except Exception as e:
            print(f"Failed to load Llama 2-7b: {e}")
        
        # Initialize hot reloaders for each component with immediate initialization
        print("Initializing class instances...")
        self.sparQLTask = SparQLTask(self.graph, self.llm)
        # self.sparQLTask.analyze_predicates()
        self.embeddingTask = EmbeddingTask(self.graph, "./")
        self.movieRecommendation = MovieRecommendation(self.graph)
        self.multimedia = Multimedia(self.graph)
        self.crowdsourcing = Crowdsourcing(self.graph, "./datasources/crowd_data.tsv")
        
        print("Class instances initialized!")
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
        # Initialize Speakeasy
        print(f"Connecting to Speakeasy at {DEFAULT_HOST_URL}...")
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()
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
                        sparql_task = self.sparQLTask
                        embedding_task = self.embeddingTask
                        movie_recommendation = self.movieRecommendation
                        multimedia = self.multimedia

                        # Check if the message is empty
                        if (message.message.strip() == ""):
                            room.post_messages("Please enter a question.")
                            
                        # 1. Check for multimedia query
                        elif multimedia.is_multimedia_query(message.message):
                            result = multimedia.handle_multimedia_query(message.message)
                            self.logAnswer("Multimedia", result)
                            room.post_messages(result)
                            
                        # 2. Check for recommendation query
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
                            
                        # 3. Check for factual question
                        else:
                            result = self.answer_factual(message.message)
                            room.post_messages(result)
                            
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

    def listen_debug(self):
        print("Welcome to the debug mode of the ATAIChatbot. Please specify the type of answer you want to get.")
        while True:
            print(f"Possible: \n- sparql \n- crowdsourcing \n- embedding \n- fallback \n- nlp2sparql \n- multimedia \n- recommendation \n- exit \n")
            answer_type = input("Please specify the type of answer: ")
            question = input("Please specify the question: ")
            answer = ""
            formatted_answer = ""
            match answer_type.lower():
                case "sparql":
                    answer = self.sparQLTask.generate_query(question)
                    formatted_answer = self.llm.generate_natural_answer(question, answer)
                case "crowdsourcing":
                    answer, kappa, distribution = self.crowdsourcing.get_crowd_answer(question)
                    formatted_answer = self.llm.generate_natural_answer(question, answer)
                    print("Answer: ", answer)
                    print("Kappa: ", kappa)
                    print("Distribution: ", distribution)
                    if answer and answer != "Keine Crowd-Antwort verfügbar.":
                        formatted_answer = self.crowdsourcing.append_crowdsourcing_information(
                            formatted_answer, kappa, distribution
                        )
                    print(formatted_answer)
                case "embedding":
                    answer = self.embeddingTask.get_embedding_answer(question)
                    formatted_answer = self.llm.generate_natural_answer(question, answer)
                case "fallback":
                    answer = self.llm.get_response(question)
                    formatted_answer = self.llm.generate_natural_answer(question, answer)
                case "nlp2sparql":
                    answer = self.sparQLTask.generate_query(question)
                    formatted_answer = self.llm.generate_natural_answer(question, answer)
                case "multimedia":
                    answer = self.multimedia.handle_multimedia_query(question)
                    formatted_answer = self.llm.generate_natural_answer(question, answer)
                case "recommendation":
                    answer = self.movieRecommendation.suggestMovie(question)
                    formatted_answer = self.llm.generate_natural_answer(question, answer)
                case "exit":
                    break
            
            print(formatted_answer)


    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

    def logAnswer(self, answer_type, answer):
        print(
        f"\t- Answer type: {answer_type} "
        f"- Answer: {answer} "
        f"- {self.get_time()}")
    
    def answer_factual(self, question: str) -> str:
        """
        Versucht eine faktische Frage zu beantworten, indem verschiedene Methoden nacheinander probiert werden:
        1. SPARQL-Query über SparQLTask
        2. Crowdsourcing-Daten
        3. Embedding-basierte Antwort als Fallback
        
        Args:
            question: Die zu beantwortende Frage in natürlicher Sprache
            
        Returns:
            str: Die beste verfügbare Antwort
        """
        try:
            # Hole aktuelle Instanzen und prüfe ob sie existieren
            sparql_task = self.sparQLTask
            embedding_task = self.embeddingTask
            crowdsourcing = self.crowdsourcing
            
            if not all([sparql_task, embedding_task, crowdsourcing]):
                print("Warning: Some components failed to initialize")
                return "Ein interner Fehler ist aufgetreten. Bitte versuchen Sie es später erneut."
            
            # 1. Versuche SPARQL-Antwort
            try:
                # Generiere SPARQL-Query
                query, source = sparql_task.generate_query(question)
                print(f"Generated SPARQL query: {query}")
                # Versuche Antwort zu bekommen
                sparql_answer = sparql_task.answer_question(question)
                print(f"SPARQL answer: {sparql_answer}")

                # Prüfe ob eine valide Antwort zurückgegeben wurde
                if sparql_answer and not any(error_msg in sparql_answer for error_msg in [
                    "Keine Ergebnisse gefunden",
                    "Fehler bei der Verarbeitung der Frage",
                    "Konnte keine valide SPARQL-Query generieren"
                ]):
                    self.logAnswer("SPARQL", sparql_answer)
                    return sparql_answer
                    
            except Exception as e:
                print(f"SPARQL-Anfrage fehlgeschlagen: {e}")
                query = None  # Setze query auf None wenn SPARQL fehlschlägt
            
            # 2. Versuche Crowdsourcing-Antwort
            try:
                if query:  # Nur wenn eine valide SPARQL-Query existiert
                    crowd_answer = crowdsourcing.get_crowd_answer(query)
                    if crowd_answer and crowd_answer != "Keine Crowd-Antwort verfügbar.":
                        self.logAnswer("Crowdsourcing", crowd_answer)
                        return crowd_answer
                    
            except Exception as e:
                print(f"Crowdsourcing-Anfrage fehlgeschlagen: {e}")
            
            # 3. Fallback: Embedding-basierte Antwort
            try:
                embedding_answer = embedding_task.get_embedding_answer(question)
                if embedding_answer and not embedding_answer.startswith("I couldn't"):
                    self.logAnswer("Embedding", embedding_answer)
                    return embedding_answer
                    
            except Exception as e:
                print(f"Embedding-Anfrage fehlgeschlagen: {e}")
            
            # Wenn keine Methode erfolgreich war
            return "Leider konnte ich keine Antwort auf diese Frage finden."
            
        except Exception as e:
            print(f"Fehler in answer_factual: {e}")
            return "Es ist ein Fehler bei der Verarbeitung der Frage aufgetreten."

def parse_arguments():
    # Optionaler Debug-Modus
    parser = argparse.ArgumentParser(description="ATAI Chatbot")
    parser.add_argument("--debug", type=bool, help="Start in debug mode", default=False)
    parser.add_argument("--disable", type=bool, help="Start in debug mode", default=[])
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = parse_arguments()
        debug = args.debug
        
        config = configparser.ConfigParser()
        config.read("./credentials.txt")
        username = config.get('credentials', 'username')
        password = config.get('credentials', 'password')
        
        demo_bot = Agent(username, password)
        if debug:
            demo_bot.listen_debug()
        else:
            demo_bot.listen()
        
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {str(e)}")
        # Fügen Sie eine Pause hinzu, damit Sie die Fehlermeldung lesen können
        input("Drücken Sie Enter zum Beenden...")
    

