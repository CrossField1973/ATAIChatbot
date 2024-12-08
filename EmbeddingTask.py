import csv
import os
import pickle

import numpy as np
import rdflib
import nltk
from nltk.tokenize import word_tokenize
download_path = "./ltk_data"  # Change this to your preferred directory
# Download NLTK data to the specified path
nltk.data.path.append(download_path)
nltk.download('punkt', download_dir=download_path)
nltk.download('punkt_tab', download_dir=download_path)
nltk.download('averaged_perceptron_tagger', download_dir=download_path)
nltk.download('stopwords', download_dir=download_path)
nltk.download('wordnet', download_dir=download_path)
nltk.download('omw-1.4', download_dir=download_path)
nltk.download('maxent_ne_chunker', download_dir=download_path)
nltk.download('words', download_dir=download_path)
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import NamespaceWrapper
from fuzzywuzzy import process
from scipy.special import softmax


class EmbeddingTask:
    def __init__(self, graph, path):
        self.path = path
        self.key_embeddings = None
        self.relationship_bert = None
        self.name2rel = None
        self.rel2name = None
        self.lbl2ent = None
        self.ent2lbl = None
        self.id2rel = None
        self.rel2id = None
        self.id2ent = None
        self.ent2id = None
        self.graph = graph
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.entity_emb = np.load(f'{path}entity_embeds.npy')
        self.relation_emb = np.load(f'{path}relation_embeds.npy')
        self.load_entity_dictionaries()
        self.ent2lbl_and_lbl2ent()
        self.rel2name_and_name2rel()
        self.load_or_compute_embeddings()

    def find_closest_key(self, query, keys, threshold=80):
        match, score = process.extractOne(query, keys)
        return match if score >= threshold else None

    def load_entity_dictionaries(self):
        with open(f'{self.path}entity_ids.del',
                  'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open(f'{self.path}relation_ids.del',
                  'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

    def ent2lbl_and_lbl2ent(self):
        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(NamespaceWrapper.RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

    def rel2name_and_name2rel(self):
        rel2name_temp = {}
        for _, pred, _ in self.graph:
            label = self.graph.value(pred, NamespaceWrapper.RDFS.label)
            if label:
                rel2name_temp[pred] = str(label)
            else:
                rel2name_temp[pred] = pred.split('/')[-1]

        self.rel2name = {rdflib.term.URIRef(rel): name for rel, name in rel2name_temp.items()}
        self.name2rel = {name: rel for rel, name in self.rel2name.items()}

    def load_or_compute_embeddings(self):
        if os.path.exists(f'{self.path}relationship_bert.pkl'):
            with open(f'{self.path}relationship_bert.pkl', "rb") as f:
                self.relationship_bert = pickle.load(f)
            print("Loaded relationship_bert from file.")
        else:
            relationship_bert = {key: self.get_embedding(key) for key in self.name2rel.keys()}
            with open(f'{self.path}relationship_bert.pkl', "wb") as f:
                pickle.dump(relationship_bert, f)
            self.relationship_bert = pickle.load(f)
            print("Computed and saved relationship_bert.")

        if os.path.exists(f'{self.path}key_embeddings.pkl'):
            with open(f'{self.path}key_embeddings.pkl', "rb") as f:
                self.key_embeddings = pickle.load(f)
            print("Loaded key_embeddings from file.")
        else:
            key_embeddings = {key: self.get_embedding(key) for key in self.lbl2ent.keys()}
            with open(f'{self.path}key_embeddings.pkl', "wb") as f:
                pickle.dump(key_embeddings, f)
            self.key_embeddings = pickle.load(f)
            print("Computed and saved key_embeddings.")


    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        outputs = self.model(**inputs)

        # Ensure consistent pooling method with precomputed embeddings
        # Adjust based on how `entity_embeds.npy` was generated
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()


    def find_best_entity_matches(self, sentence, threshold=0.5):
        words = word_tokenize(sentence)

        # Extract potential movie titles by matching dictionary keys with sentence tokens
        found_titles = [
            self.find_closest_key(word, self.key_embeddings.keys())
            for word in word_tokenize(sentence)
        ]
        found_titles = list(filter(None, found_titles))
        longest_title = max(found_titles, key=len) if found_titles else None

        # If we found potential titles, compute similarity
        matches = []
        if longest_title:
            entity_embedding = self.get_embedding(longest_title).reshape(1, -1)
            similarities = {
                key: cosine_similarity(entity_embedding, emb.reshape(1, -1)).item()
                for key, emb in self.key_embeddings.items()
            }
            best_match_key = max(similarities, key=similarities.get)
            if similarities[best_match_key]:
                matches.append((longest_title, best_match_key, similarities[best_match_key], self.lbl2ent[best_match_key]))

        return matches

    def find_best_relationship_matches(self, sentence, threshold=0.5):
        # Tokenize the text
        words = word_tokenize(sentence)

        # Extract potential movie titles by matching dictionary keys with sentence tokens
        found_titles = [
            self.find_closest_key(word, self.key_embeddings.keys())
            for word in word_tokenize(sentence)
        ]
        found_titles = list(filter(None, found_titles))

        print(f"Short:{found_titles}")
        # If we found potential titles, compute similarity
        matches = []
        if found_titles:
            entity_embedding = self.get_embedding(found_titles).reshape(1, -1)
            similarities = {
                key: cosine_similarity(entity_embedding, emb.reshape(1, -1)).item()
                for key, emb in self.key_embeddings.items()
            }
            best_match_key = max(similarities, key=similarities.get)
            if similarities[best_match_key]:
                matches.append((found_titles, best_match_key, similarities[best_match_key], self.name2rel[best_match_key]))

            return matches
        if not found_titles:
            return "I couldn't identify a related entity. Could you rephrase or provide more details?"
        if not matches:
            return "I couldn't find a relationship match for this query."

    

    def get_embedding_answer(self, sentence):
        try:
            entity_matches = self.find_best_entity_matches(sentence)
            if not entity_matches:
                return "I couldn't identify a related entity. Could you rephrase or provide more details?"

            entity_text, entity_matched, entity_similarity, entity_uri = entity_matches[0]

            relationship_matches = self.find_best_relationship_matches(sentence)
            if not relationship_matches:
                return "I couldn't find a relationship match for this query."

            relationship_text, relationship_matched, relationship_similarity, relationship_uri = relationship_matches[0]

            head = self.entity_emb[self.ent2id[NamespaceWrapper.uri_to_prefixed(entity_uri)]]
            pred = self.relation_emb[self.rel2id[NamespaceWrapper.uri_to_prefixed(relationship_uri)]]
            lhs = head + pred
            dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
            probabilities = softmax(-dist)
            most_likely = np.argsort(probabilities)[-3:]

            predictions = [(self.id2ent[idx], self.ent2lbl.get(self.id2ent[idx], "Unknown")) for idx in most_likely]
            return f"Possible answers: {', '.join([f'{lbl}' for _, lbl in predictions])}"


        except Exception as e:
            print(f"Error processing query: {e}")
            return "Unfortunately, I cannot provide an answer right now."
