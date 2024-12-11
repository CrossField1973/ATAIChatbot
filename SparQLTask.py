import rdflib
import time
import re
from rdflib.plugins.sparql.parser import parseQuery
from rdflib import Graph, URIRef, Literal
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import spacy
from typing import Optional, Tuple


class SparQLTask:
    def __init__(self, graph, llm):
        """
        Initialisiert die SPARQLTask mit einem RDF-Graphen und Modellen für Query-Generierung.
        
        Args:
            graph: RDF-Graph
            llama_model_path: Pfad zum LLaMA-Modell
        """
        self.graph = graph
        self.llm = llm
        self.query_generator = QueryGenerator(self.llm)
        
        
    def answer_question(self, question):
        """
        Beantwortet eine Frage durch Generierung und Ausführung einer SPARQL-Query.
        
        Args:
            question: Die Frage in natürlicher Sprache
            
        Returns:
            str: Die formatierte Antwort
        """
        try:
            # Generiere SPARQL-Query
            if self.is_valid_sparql(question):
                query = question
                source = "manuell"
            else:
                query, source = self.query_generator.generate_query(question)
            
            # Führe Query aus
            results = self.graph.query(query)
            
            # Formatiere Ergebnisse
            formatted_results = self._format_results(results)
            
            # Generiere natürlichsprachliche Antwort
            natural_answer = self.generate_natural_answer(question, formatted_results)
            
            # Füge Informationen über die Query-Quelle hinzu
            source_info = f"\n(Query generiert mit {source})"
            
            return natural_answer + source_info
            
        except Exception as e:
            print(f"Fehler bei der Verarbeitung der Frage: {str(e)}")
            return "Fehler bei der Verarbeitung der Frage."
    
    def _format_results(self, results):
        """
        Formatiert die SPARQL-Ergebnisse in einen lesbaren String.
        
        Args:
            results: SPARQL-Ergebnisse
            
        Returns:
            str: Formatierte Ergebnisse
        """
        if not results:
            return "Keine Ergebnisse gefunden."
            
        formatted = []
        for row in results:
            # Behandle verschiedene Ergebnistypen
            if len(row) == 1:
                value = row[0]
                if isinstance(value, URIRef):
                    formatted.append(str(value).split('/')[-1])
                elif isinstance(value, Literal):
                    formatted.append(str(value))
                else:
                    formatted.append(str(value))
            else:
                # Für mehrere Variablen in der Query
                row_values = []
                for value in row:
                    if isinstance(value, URIRef):
                        row_values.append(str(value).split('/')[-1])
                    elif isinstance(value, Literal):
                        row_values.append(str(value))
                    else:
                        row_values.append(str(value))
                formatted.append(", ".join(row_values))
        
        if len(formatted) == 1:
            return formatted[0]
        return "\n".join(f"- {item}" for item in formatted)

    def generate_query(self, question):
        return self.query_generator.generate_query(question)

    def is_valid_sparql(self, query):
        return self.query_generator.is_valid_sparql(query)

    def analyze_predicates(self):
        """
        Analysiert alle verwendeten Prädikate im Graphen und gibt diese mit Beispielen zurück.
        """
        query = """
        SELECT DISTINCT ?predicate (COUNT(?s) as ?count) (SAMPLE(?s) AS ?example_subject) (SAMPLE(?o) as ?example_object)
        WHERE {
            ?s ?predicate ?o .
        }
        GROUP BY ?predicate
        ORDER BY DESC(?count)
        """
        
        try:
            results = self.graph.query(query)
            print("Gefundene Prädikate im Graphen:")
            print("--------------------------------")
            for row in results:
                predicate = str(row.predicate)
                count = str(row.count)
                example_s = str(row.example_subject)
                example_o = str(row.example_object)
                print(f"\nPrädikat: {predicate}")
                print(f"Anzahl: {count}")
                print(f"Beispiel: {example_s} -> {example_o}")
                
                # Optional: Zeige ein konkretes Beispiel mit Labels
                example_query = f"""
                SELECT ?subject_label ?object_label
                WHERE {{
                    <{example_s}> <{predicate}> <{example_o}> .
                    OPTIONAL {{ <{example_s}> rdfs:label ?subject_label . FILTER(LANG(?subject_label) = "en") }}
                    OPTIONAL {{ <{example_o}> rdfs:label ?object_label . FILTER(LANG(?object_label) = "en") }}
                }}
                LIMIT 1
                """
                example_results = self.graph.query(example_query)
                for ex_row in example_results:
                    if ex_row.subject_label and ex_row.object_label:
                        print(f"Beispiel mit Labels: {ex_row.subject_label} -> {ex_row.object_label}")
            
        except Exception as e:
            print(f"Fehler bei der Analyse der Prädikate: {str(e)}")

class QueryGenerator:
    def __init__(self, llm):
        # Template-System initialisieren
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # LLaMA initialisieren
        self.llm = llm  # Speichere den übergebenen LlamaHandler

        # Rest des Konstruktors bleibt gleich...
        self.query_templates = {
            # Film -> Regisseur
            "movie_director": """
                SELECT DISTINCT ?director_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P57 ?director .
                    ?director rdfs:label ?director_name .
                    FILTER(LANG(?director_name) = 'en')
                }}
                ORDER BY ?director_name
            """,
            
            # Regisseur -> Filme
            "director_movies": """
                SELECT DISTINCT ?movie_name WHERE {{
                    ?director rdfs:label "{}"@en .
                    ?movie ns1:P57 ?director .
                    ?movie rdfs:label ?movie_name .
                    FILTER(LANG(?movie_name) = 'en')
                }}
                ORDER BY ?movie_name
            """,
            
            # Film -> Schauspieler
            "movie_actors": """
                SELECT DISTINCT ?actor_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P161 ?actor .
                    ?actor rdfs:label ?actor_name .
                    FILTER(LANG(?actor_name) = 'en')
                }}
                ORDER BY ?actor_name
            """,
            
            # Schauspieler -> Filme
            "actor_movies": """
                SELECT DISTINCT ?movie_name WHERE {{
                    ?actor rdfs:label "{}"@en .
                    ?movie ns1:P161 ?actor .
                    ?movie rdfs:label ?movie_name .
                    FILTER(LANG(?movie_name) = 'en')
                }}
                ORDER BY ?movie_name
            """,
            
            # Film -> Genre
            "movie_genre": """
                SELECT DISTINCT ?genre WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P136 ?genre_uri .
                    ?genre_uri rdfs:label ?genre .
                    FILTER(LANG(?genre) = 'en')
                }}
            """,
            
            # Film -> Erscheinungsdatum
            "movie_release_date": """
                SELECT DISTINCT ?date WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P577 ?date .
                }}
            """,
            
            # Film -> Laufzeit
            "movie_runtime": """
                SELECT DISTINCT ?runtime WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P2047 ?runtime .
                }}
            """,
            
            # Genre -> Filme
            "movies_by_genre": """
                SELECT DISTINCT ?movie_name WHERE {{
                    ?movie ns1:P136 ?genre .
                    ?genre rdfs:label "{}"@en .
                    ?movie rdfs:label ?movie_name .
                    FILTER(LANG(?movie_name) = 'en')
                }}
                ORDER BY ?movie_name
                LIMIT 10
            """,
            
            # Schauspieler -> Geburtsdatum
            "actor_birthdate": """
                SELECT DISTINCT ?date WHERE {{
                    ?actor rdfs:label "{}"@en .
                    ?actor ns1:P569 ?date .
                }}
            """,
            
            # Film -> Budget
            "movie_budget": """
                SELECT DISTINCT ?budget WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P2130 ?budget .
                }}
            """,

            # Film -> Produktionsfirma
            "movie_production_company": """
                SELECT DISTINCT ?company WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P272 ?company_uri .
                    ?company_uri rdfs:label ?company .
                    FILTER(LANG(?company) = 'en')
                }}
            """,

            # Film -> Sprache
            "movie_language": """
                SELECT DISTINCT ?language WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P364 ?language_uri .
                    ?language_uri rdfs:label ?language .
                    FILTER(LANG(?language) = 'en')
                }}
            """,

            # Film -> Drehbuchautor
            "movie_writer": """
                SELECT DISTINCT ?writer_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P58 ?writer .
                    ?writer rdfs:label ?writer_name .
                    FILTER(LANG(?writer_name) = 'en')
                }}
            """,

            # Film -> Auszeichnungen
            "movie_awards": """
                SELECT DISTINCT ?award WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P166 ?award_uri .
                    ?award_uri rdfs:label ?award .
                    FILTER(LANG(?award) = 'en')
                }}
            """,

            # Schauspieler -> Nationalität
            "actor_nationality": """
                SELECT DISTINCT ?nationality WHERE {{
                    ?actor rdfs:label "{}"@en .
                    ?actor ns1:P27 ?nat_uri .
                    ?nat_uri rdfs:label ?nationality .
                    FILTER(LANG(?nationality) = 'en')
                }}
            """,

            # Film -> Sequel
            "movie_sequel": """
                SELECT DISTINCT ?sequel_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?sequel ns1:P155 ?movie .
                    ?sequel rdfs:label ?sequel_name .
                    FILTER(LANG(?sequel_name) = 'en')
                }}
            """,

            # Film -> Prequel
            "movie_prequel": """
                SELECT DISTINCT ?prequel_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P155 ?prequel .
                    ?prequel rdfs:label ?prequel_name .
                    FILTER(LANG(?prequel_name) = 'en')
                }}
            """,

            # Filme eines bestimmten Jahres
            "movies_by_year": """
                SELECT DISTINCT ?movie_name WHERE {{
                    ?movie ns1:P577 ?date .
                    ?movie rdfs:label ?movie_name .
                    FILTER(YEAR(?date) = {})
                    FILTER(LANG(?movie_name) = 'en')
                }}
                ORDER BY ?movie_name
                LIMIT 10
            """,

            # Neue Templates für Filme
            "movie_cinematographer": """
                SELECT DISTINCT ?cinematographer_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P344 ?cinematographer .
                    ?cinematographer rdfs:label ?cinematographer_name .
                    FILTER(LANG(?cinematographer_name) = 'en')
                }}
            """,
            
            "movie_filming_location": """
                SELECT DISTINCT ?location_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P915 ?location .
                    ?location rdfs:label ?location_name .
                    FILTER(LANG(?location_name) = 'en')
                }}
            """,
            
            "movie_based_on": """
                SELECT DISTINCT ?source_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P144 ?source .
                    ?source rdfs:label ?source_name .
                    FILTER(LANG(?source_name) = 'en')
                }}
            """,
            
            "movie_distributor": """
                SELECT DISTINCT ?distributor_name WHERE {{
                    ?movie rdfs:label "{}"@en .
                    ?movie ns1:P750 ?distributor .
                    ?distributor rdfs:label ?distributor_name .
                    FILTER(LANG(?distributor_name) = 'en')
                }}
            """,
            
            # Neue Templates für Personen
            "person_citizenship": """
                SELECT DISTINCT ?country_name WHERE {{
                    ?person rdfs:label "{}"@en .
                    ?person ns1:P27 ?country .
                    ?country rdfs:label ?country_name .
                    FILTER(LANG(?country_name) = 'en')
                }}
            """,
            
            "person_occupation": """
                SELECT DISTINCT ?occupation_name WHERE {{
                    ?person rdfs:label "{}"@en .
                    ?person ns1:P106 ?occupation .
                    ?occupation rdfs:label ?occupation_name .
                    FILTER(LANG(?occupation_name) = 'en')
                }}
            """
        }
        
        self.patterns = {
            "movie_director": [
                r"who (?:directed|was the director of) (.+?)\??$",
                r"who is the director of (.+?)\??$",
                r"(.+?)'s director\??$",
                r"tell me who directed (.+?)\??$",
                r"name the director of (.+?)\??$"
            ],
            
            "director_movies": [
                r"what (?:movies|films) did (.+?) direct\??$",
                r"which (?:movies|films) were directed by (.+?)\??$",
                r"list (?:movies|films) by (.+?)\??$",
                r"show me all movies directed by (.+?)\??$",
                r"what has (.+?) directed\??$"
            ],
            
            "movie_actors": [
                r"who (?:acted|starred|played) in (.+?)\??$",
                r"who are the actors in (.+?)\??$",
                r"cast of (.+?)\??$",
                r"who was in the cast of (.+?)\??$",
                r"list the actors in (.+?)\??$",
                r"tell me who acted in (.+?)\??$"
            ],
            
            "actor_movies": [
                r"what (?:movies|films) did (.+?) (?:act|star|play) in\??$",
                r"which (?:movies|films) features (.+?)\??$",
                r"list (?:movies|films) with (.+?)\??$",
                r"what has (.+?) appeared in\??$",
                r"show me movies with (.+?)\??$"
            ],
            
            "movie_genre": [
                r"what (?:is|are) the genres? of (.+?)\??$",
                r"which genre is (.+?)\??$",
                r"(.+?) genre\??$",
                r"what type of movie is (.+?)\??$",
                r"what kind of film is (.+?)\??$"
            ],
            
            "movie_release_date": [
                r"when was (.+?) released\??$",
                r"release date of (.+?)\??$",
                r"when did (.+?) come out\??$",
                r"what year was (.+?) released\??$",
                r"when did (.+?) premiere\??$"
            ],
            
            "movie_runtime": [
                r"how long is (.+?)\??$",
                r"what is the (?:runtime|duration|length) of (.+?)\??$",
                r"(.+?) runtime\??$",
                r"duration of (.+?)\??$",
                r"how many minutes is (.+?)\??$"
            ],
            
            "movies_by_genre": [
                r"what (?:movies|films) are (.+?)\??$",
                r"list (?:movies|films) in the (.+?) genre\??$",
                r"show me (.+?) (?:movies|films)\??$",
                r"what are some (.+?) movies\??$",
                r"give me examples of (.+?) films\??$"
            ],
            
            "actor_birthdate": [
                r"when was (.+?) born\??$",
                r"what is (.+?)'s birth date\??$",
                r"(.+?) birth date\??$",
                r"what's (.+?)'s date of birth\??$",
                r"when is (.+?)'s birthday\??$"
            ],
            
            "movie_budget": [
                r"what was the budget of (.+?)\??$",
                r"how much did (.+?) cost\??$",
                r"(.+?) budget\??$",
                r"how much money was spent on (.+?)\??$",
                r"what was the cost of making (.+?)\??$"
            ],

            # NEU: Produktionsfirma
            "movie_production_company": [
                r"who produced (.+?)\??$",
                r"which company produced (.+?)\??$",
                r"what company made (.+?)\??$",
                r"who was the production company for (.+?)\??$",
                r"which studio made (.+?)\??$"
            ],

            # NEU: Sprache
            "movie_language": [
                r"what language is (.+?) in\??$",
                r"what's the original language of (.+?)\??$",
                r"which language was (.+?) filmed in\??$",
                r"what language was (.+?) made in\??$"
            ],

            # NEU: Drehbuchautor
            "movie_writer": [
                r"who wrote (.+?)\??$",
                r"who was the writer of (.+?)\??$",
                r"who wrote the screenplay for (.+?)\??$",
                r"who's the screenwriter of (.+?)\??$"
            ],

            # NEU: Auszeichnungen
            "movie_awards": [
                r"what awards did (.+?) win\??$",
                r"which awards were won by (.+?)\??$",
                r"what awards was (.+?) nominated for\??$",
                r"list the awards for (.+?)\??$"
            ],

            # NEU: Nationalität
            "actor_nationality": [
                r"what nationality is (.+?)\??$",
                r"where is (.+?) from\??$",
                r"what is (.+?)'s nationality\??$",
                r"which country is (.+?) from\??$"
            ],

            # NEU: Sequel
            "movie_sequel": [
                r"what's the sequel to (.+?)\??$",
                r"which movie follows (.+?)\??$",
                r"what came after (.+?)\??$",
                r"is there a sequel to (.+?)\??$"
            ],

            # NEU: Prequel
            "movie_prequel": [
                r"what's the prequel to (.+?)\??$",
                r"which movie came before (.+?)\??$",
                r"what preceded (.+?)\??$",
                r"is there a prequel to (.+?)\??$"
            ],

            # NEU: Filme eines Jahres
            "movies_by_year": [
                r"what movies came out in (\d{4})\??$",
                r"list (?:movies|films) from (\d{4})\??$",
                r"show me (?:movies|films) released in (\d{4})\??$",
                r"what was released in (\d{4})\??$"
            ],

            # Neue Patterns für Filme
            "movie_cinematographer": [
                r"who was the (?:cinematographer|director of photography) (?:for|of) (.+?)\??$",
                r"who filmed (.+?)\??$",
                r"who shot (.+?)\??$",
                r"(.+?) cinematographer\??$"
            ],
            
            "movie_filming_location": [
                r"where was (.+?) filmed\??$",
                r"where did they film (.+?)\??$",
                r"filming location(?:s)? of (.+?)\??$",
                r"where did they shoot (.+?)\??$"
            ],
            
            "movie_based_on": [
                r"what is (.+?) based on\??$",
                r"what was the source material for (.+?)\??$",
                r"what inspired (.+?)\??$",
                r"(.+?) source material\??$"
            ],
            
            "movie_distributor": [
                r"who distributed (.+?)\??$",
                r"which company distributed (.+?)\??$",
                r"distributor of (.+?)\??$",
                r"who released (.+?)\??$"
            ],
            
            # Neue Patterns für Personen
            "person_citizenship": [
                r"what citizenship does (.+?) have\??$",
                r"what nationality is (.+?)\??$",
                r"which country is (.+?) from\??$",
                r"what is (.+?)'s citizenship\??$"
            ],
            
            "person_occupation": [
                r"what (?:is|was) (.+?)'s (?:job|occupation|profession)\??$",
                r"what does (.+?) do for a living\??$",
                r"what kind of work does (.+?) do\??$",
                r"what is (.+?)'s profession\??$"
            ]
        }

    def _extract_entity(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrahiert die relevante Entität und den Query-Typ aus der Frage.
        
        Args:
            question: Die Frage in natürlicher Sprache
            
        Returns:
            Tuple von (entity, query_type) oder (None, None) wenn keine Übereinstimmung
        """
        # Bereinige die Frage
        question = question.strip().lower()
        
        # Entferne Anführungszeichen falls vorhanden
        question = question.replace('"', '').replace('"', '').replace('"', '')
        
        # Prüfe alle Patterns
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip()
                    
                    # Bereinige die Entität
                    entity = re.sub(r'^the\s+', '', entity, flags=re.IGNORECASE)
                    entity = entity.strip()
                    
                    # Nutze spaCy für Namenerkennung bei Personen
                    if query_type in ['actor_movies', 'actor_birthdate', 'director_movies']:
                        doc = self.nlp(entity)
                        for ent in doc.ents:
                            if ent.label_ in ['PERSON', 'ORG']:
                                entity = ent.text
                                break
                    
                    return entity, query_type
                    
        return None, None

    def generate_query(self, question: str) -> Tuple[str, str]:
        """
        Versucht eine SPARQL-Query zu generieren, erst mit Templates,
        dann mit LLaMA als Fallback.
        """
        # try:
        #     # 1. Versuche Template-Ansatz
        #     entity, query_type = self._extract_entity(question)
        #     if entity and query_type and query_type in self.query_templates:
        #         # Füge Präfixe hinzu
        #         prefixes = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX dbo: <http://dbpedia.org/ontology/>
# """
        #         # Formatiere die Query mit der extrahierten Entität
        #         query_template = self.query_templates[query_type]
        #         query = prefixes + query_template.format(entity)
        #         
        #         # Entferne überflüssige Leerzeichen und Zeilenumbrüche
        #         query = " ".join(query.split())
        #         
        #         if self.is_valid_sparql(query):
        #             print(f"Generated valid SPARQL query: {query}")
        #             return query, 'template'
        #             
        # except Exception as e:
        #     print(f"Template generation failed: {e}")

        # 2. LLaMA Fallback
        try:
            query = self._generate_with_llama(question)
            if query and self.is_valid_sparql(query):
                return query, 'llama'
        except Exception as e:
            print(f"LLaMA generation failed: {e}")

        raise ValueError(f"Could not generate valid SPARQL query for: {question}")

    def _generate_with_llama(self, question: str) -> Optional[str]:
        """
        Generiert SPARQL-Query mit LLaMA.
        """
        prompt = f"""
        Generate a SPARQL query for the following movie-related question.
        Use these prefixes:
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>

        Question: {question}

        Return only the SPARQL query, nothing else.
        
        SPARQL Query:
        """

        try:
            # Verwende LlamaHandler direkt
            generated_text = self.llm(prompt)
            return generated_text.strip()

        except Exception as e:
            print(f"Error in LLaMA generation: {e}")
            return None

    def is_valid_sparql(self, query: str) -> bool:
        """
        Überprüft, ob eine SPARQL-Query gültig ist.
        """
        try:
            parseQuery(query)
            return True
        except Exception as e:
            print(f"Invalid SPARQL query: {e}")
            return False
