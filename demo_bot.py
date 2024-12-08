import configparser
import os
import sys

module_dir = os.path.abspath("C:\\Users\\z004re3w\\Documents\\ATAI-New\\ATAIChatbot\\speakeasy-python-client-library\\")
sys.path.append(module_dir)

from typing import List
import time
import locale

from SparQLTask import SparQLTask
from EmbeddingTask import EmbeddingTask
from speakeasypy import Speakeasy, Chatroom
from MovieRecommendation import MovieRecommendation
import Crowdsourcing


_ = locale.setlocale(locale.LC_ALL, '')
#init_notebook_mode(connected=True)

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2


class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
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

                    # Implement your agent here #
                    try:

                        # Check if the message is empty
                        if (message.message.strip() == ""):
                            room.post_messages("Please enter a question.")

                        # Is SPARQL query
                        elif sparQLTask.is_valid_sparql(message.message):
                            sparql_results = sparQLTask.process_sparql_query(message.message)
                            print(
                            f"\t- Answer type: SPARQL "
                            f"- Answer {sparql_results} "
                            f"- {self.get_time()}")
                            room.post_messages(sparql_results)

                        # Is recommendation query
                        elif movieRecommendation.is_recommendation_query(message.message):
                            # Check movie titles
                            movie_titles, unrecognized_titles = movieRecommendation.extract_movie_titles(message.message)
                
                            if not movie_titles:
                                room.post_messages("I couldn't detect any movie titles. Could you rephrase your question?")
                            
                            else:
                                # Recommend movies
                                recommended_movies = movieRecommendation.suggestMovie(movie_titles, numRecommendations=5, similarity_threshold=0.01)
                                
                                # Prepare answer
                                recommended_answer = ""
                                if recommended_movies:
                                    for movie, reasons, _ in recommended_movies:
                                        recommended_answer += f"Recommended: {movie} (Reason: {reasons})\n"
                                else:
                                    recommended_answer = "The following movies were found on IMDb but are not in our database:\n"
                                    recommended_answer += "\n".join(movie_titles) + "\n"

                                for title in unrecognized_titles:
                                    recommended_answer += f"Couldn't recognize the following movie: {title}\n"

                                print(
                                f"\t- Answer type: Recommendation "
                                f"- Answer {recommended_answer} "
                                f"- {self.get_time()}")
                                room.post_messages(recommended_answer)

                        # Embedding query
                        else:
                            message_to_post = embeddingTask.get_embedding_answer(sentence=message.message)
                            print(
                            f"\t- Answer type: Embedding "
                            f"- Answer {message_to_post} "
                            f"- {self.get_time()}")
                            room.post_messages(message_to_post)
                    except Exception as e:
                        print(f"Error processing query: {e}")
                        room.post_messages("Unfortunately, I cannot provide an answer right now.")

                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
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
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())
    

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("./credentials.txt")
    username = config.get('credentials', 'username')
    password = config.get('credentials', 'password')
    sparQLTask = SparQLTask()
    embeddingTask = EmbeddingTask(sparQLTask.graph, './')
    movieRecommendation = MovieRecommendation()

    # Load data
    df = Crowdsourcing.load_crowd_data('./crowd_data.tsv')

    # Filter malicious workers
    filtered_df = Crowdsourcing.filter_malicious_workers(df)

    # Compute Fleiss' kappa
    kappa = Crowdsourcing.compute_fleiss_kappa(filtered_df)
    print(f"Fleiss' kappa for this batch: {kappa}")

    # Aggregate answers
    aggregated_results = Crowdsourcing.aggregate_answers(filtered_df)
    print("Aggregated Results:")
    print(aggregated_results)

    demo_bot = Agent(username, password)
    demo_bot.listen()
    

