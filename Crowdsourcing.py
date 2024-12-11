from rdflib import Graph, URIRef, Literal, Namespace, XSD
import pandas as pd
from collections import Counter, defaultdict
from statsmodels.stats.inter_rater import fleiss_kappa
from fuzzywuzzy import fuzz

class Crowdsourcing:
    def __init__(self, graph, tsv_file):
        """Initialisiert die Crowdsourcing-Klasse mit den Daten aus der TSV-Datei und einem RDF-Graphen."""
        try:
            self.data = pd.read_csv(tsv_file, sep='\t')
            print(f"Geladene Datensätze: {len(self.data)}")
            
            self.graph = graph
            self.filtered_data = self.filter_malicious_workers()
            print(f"Datensätze nach Filter: {len(self.filtered_data)}")
            
            print("\nStarte Aggregation der Antworten...")
            self.aggregated_results = self.aggregate_answers()
            print(f"Aggregierte Ergebnisse: {len(self.aggregated_results)}")
            
            print("\nBerechne Fleiss Kappa...")
            self.kappas = self.compute_fleiss_kappa()
            
            print("\nErstelle HIT Mapping...")
            self.hit_mapping = self._create_hit_mapping()
            
            self.added_relations = []
            
            print("\nAktualisiere Graph mit Crowd-Daten...")
            self.update_graph_with_crowd_data()
            
            print("\nDrucke hinzugefügte Relationen...")
            self.print_added_relations()
            
        except Exception as e:
            print(f"Fehler in __init__: {str(e)}")
            raise
    
    def filter_malicious_workers(self, min_approval_rate=0.8, max_work_time=3600):
        """
        Filtert Antworten von böswilligen Crowd-Workern.
        
        Args:
            min_approval_rate: Minimale Zustimmungsrate (default: 0.8 oder 80%)
            max_work_time: Maximale Arbeitszeit in Sekunden (default: 3600)
        """
        # Zuerst die Daten anzeigen
        print("\nVerfügbare Assignment Status:")
        print(self.data['AssignmentStatus'].value_counts())
        
        filtered = self.data[
            (self.data['LifetimeApprovalRate'].str.rstrip('%').astype(float) / 100 >= min_approval_rate) &
            (self.data['WorkTimeInSeconds'].astype(float) <= max_work_time) &
            (self.data['WorkTimeInSeconds'].astype(float) > 0) &
            (self.data['AssignmentStatus'] == 'Submitted')  # Fokus auf "Submitted" Status
        ]
        
        print("\nFilter-Statistiken:")
        print(f"Approval Rate Filter: {len(self.data[self.data['LifetimeApprovalRate'].str.rstrip('%').astype(float) / 100 >= min_approval_rate])}")
        print(f"Work Time Filter: {len(self.data[self.data['WorkTimeInSeconds'].astype(float) <= max_work_time])}")
        print(f"Status Filter: {len(self.data[self.data['AssignmentStatus'] == 'Submitted'])}")
        
        return filtered
    
    def aggregate_answers(self):
        """Aggregiert Antworten mit Mehrheitsabstimmung."""
        try:
            print("Debug: Starte Aggregation...")
            aggregated_results = {}
            
            # Zeige die ersten paar Zeilen der gefilterten Daten
            print("\nErste Zeilen der gefilterten Daten:")
            print(self.filtered_data[['HITId', 'AnswerLabel', 'Input1ID', 'Input2ID', 'Input3ID']].head())
            
            for hit_id, group in self.filtered_data.groupby('HITId'):
                try:
                    print(f"\nVerarbeite HIT ID: {hit_id}")
                    print(f"Anzahl Antworten: {len(group)}")
                    
                    # Überprüfe, ob genügend Antworten vorhanden sind
                    if len(group) == 0:
                        print(f"Überspringe HIT {hit_id} - keine Antworten")
                        continue
                        
                    answer_counts = Counter(group['AnswerLabel'])
                    print(f"Antwortverteilung: {dict(answer_counts)}")
                    
                    # Überprüfe, ob es eine eindeutige Mehrheit gibt
                    max_count = max(answer_counts.values())
                    majority_answers = [ans for ans, count in answer_counts.items() if count == max_count]
                    
                    if len(majority_answers) > 1:
                        print(f"Keine eindeutige Mehrheit für HIT {hit_id}")
                        majority_answer = majority_answers[0]  # Nimm die erste Antwort als Default
                    else:
                        majority_answer = majority_answers[0]
                        
                    print(f"Mehrheitsantwort: {majority_answer}")
                    
                    # Stelle sicher, dass die Werte existieren
                    if group['Input1ID'].empty or group['Input2ID'].empty or group['Input3ID'].empty:
                        print(f"Überspringe HIT {hit_id} - fehlende Input-Werte")
                        continue
                        
                    triple_info = {
                        'subject': group['Input1ID'].iloc[0],
                        'predicate': group['Input2ID'].iloc[0],
                        'object': group['Input3ID'].iloc[0]
                    }
                    
                    fix_info = None
                    if majority_answer == 'INCORRECT':
                        fix_positions = Counter(group[group['FixPosition'].notna()]['FixPosition'])
                        if fix_positions:
                            most_common_pos = fix_positions.most_common(1)[0][0]
                            fix_values = group[group['FixPosition'] == most_common_pos]['FixValue']
                            fix_value = fix_values.mode()[0] if not fix_values.empty else None
                            if fix_value:
                                fix_info = {'position': most_common_pos, 'value': fix_value}
                    
                    aggregated_results[hit_id] = {
                        'majority_answer': majority_answer,
                        'answer_distribution': dict(answer_counts),
                        'triple': triple_info,
                        'fix_info': fix_info
                    }
                    
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von HIT {hit_id}: {str(e)}")
                    continue
            
            print(f"\nGesamtzahl aggregierter Ergebnisse: {len(aggregated_results)}")
            if len(aggregated_results) == 0:
                print("Warnung: Keine Ergebnisse wurden aggregiert")
                return {}
                
            return aggregated_results
            
        except Exception as e:
            print(f"Fehler in aggregate_answers: {str(e)}")
            return {}
    
    def compute_fleiss_kappa(self):
        """Berechnet Fleiss' Kappa für jeden Batch."""
        kappas = {}
        for hit_type_id, group in self.filtered_data.groupby('HITTypeId'):
            hit_ratings = defaultdict(lambda: {'CORRECT': 0, 'INCORRECT': 0})
            
            for _, row in group.iterrows():
                hit_ratings[row['HITId']][row['AnswerLabel']] += 1
            
            matrix = [[counts['CORRECT'], counts['INCORRECT']] 
                     for counts in hit_ratings.values()]
            
            if matrix:
                kappas[hit_type_id] = fleiss_kappa(matrix)
            
        return kappas
    
    def _create_hit_mapping(self):
        """Erstellt ein Mapping von Triple-Informationen zu HITIds."""
        hit_mapping = {}
        for hit_id, result in self.aggregated_results.items():
            triple = result['triple']
            key_combinations = [
                f"{triple['subject']} {triple['predicate']} {triple['object']}",
                f"{triple['subject']} {triple['object']}",
                f"{triple['predicate']} {triple['object']}",
                triple['subject'],
                triple['object']
            ]
            for key in key_combinations:
                if key not in hit_mapping:
                    hit_mapping[key] = []
                hit_mapping[key].append(hit_id)
        
        return hit_mapping
    
    def update_graph_with_crowd_data(self):
        """Aktualisiert den RDF-Graphen basierend auf den Crowdsourcing-Daten."""
        try:
            if not self.aggregated_results:
                print("Keine aggregierten Ergebnisse verfügbar")
                return
            
            print("\nFüge Triples zum Graphen hinzu...")
            for hit_id, result in self.aggregated_results.items():
                try:
                    triple = result['triple']
                    if result['majority_answer'] == 'CORRECT':
                        # Stelle sicher, dass die URIs korrekt formatiert sind
                        subject = triple['subject'] if triple['subject'].startswith('http') else f"http://www.wikidata.org/entity/{triple['subject'].replace('wd:', '')}"
                        predicate = triple['predicate'] if triple['predicate'].startswith('http') else f"http://www.wikidata.org/prop/direct/{triple['predicate'].replace('wdt:', '')}"
                        
                        # Überprüfe, ob das Objekt ein Datum ist
                        object_value = triple['object']
                        if isinstance(object_value, str) and any(
                            object_value.count(sep) >= 2 for sep in ['-', '/', '.']
                        ):
                            # Behandle als Datum
                            object_uri = Literal(object_value, datatype=XSD.date)
                            print(f"Datum erkannt: {object_value}")
                        else:
                            # Behandle als normale URI
                            object_uri = URIRef(object_value if object_value.startswith('http') else f"http://www.wikidata.org/entity/{object_value}")
                        
                        print(f"Füge Triple hinzu für HIT {hit_id}:")
                        print(f"Subject: {subject}")
                        print(f"Predicate: {predicate}")
                        print(f"Object: {object_uri}")
                        
                        self.graph.add((URIRef(subject), URIRef(predicate), object_uri))
                        self.added_relations.append({
                            'subject': subject.split('/')[-1],
                            'predicate': predicate.split('/')[-1],
                            'object': str(object_uri),
                            'type': 'original'
                        })
                        
                except Exception as e:
                    print(f"Fehler beim Hinzufügen von Triple für HIT {hit_id}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Fehler in update_graph_with_crowd_data: {str(e)}")
    
    def print_added_relations(self):
        """Gibt alle zum Graphen hinzugefügten Relationen aus."""
        print("\nHinzugefügte Relationen:")
        print("------------------------")
        for relation in self.added_relations:
            relation_str = f"{relation['subject']} --[{relation['predicate']}]--> {relation['object']}"
            if relation['type'] != 'original':
                relation_str += f" (Korrigiert: {relation['type']})"
            print(relation_str)
        print(f"\nGesamtzahl der hinzugefügten Relationen: {len(self.added_relations)}")
    
    def get_crowd_answer(self, sparql_query):
        """
        Führt eine SPARQL-Query auf dem erweiterten Knowledge Graph aus und gibt die formatierte Antwort zurück.
        
        Returns:
            tuple: (answer, kappa, answer_distribution)
        """
        try:
            # Führe die SPARQL-Query auf dem erweiterten Graphen aus
            results = self.graph.query(sparql_query)
            results_list = list(results)
            
            if not results_list:
                return ("Keine Crowd-Antwort verfügbar.", None, None)
            
            # Formatiere die Ergebnisse
            formatted_results = []
            for row in results_list:
                if len(row) == 1:
                    value = row[0]
                    if isinstance(value, URIRef):
                        formatted_results.append(str(value).split('/')[-1])
                    elif isinstance(value, Literal):
                        formatted_results.append(str(value))
                    else:
                        formatted_results.append(str(value))
                else:
                    row_values = []
                    for value in row:
                        if isinstance(value, URIRef):
                            row_values.append(str(value).split('/')[-1])
                        elif isinstance(value, Literal):
                            row_values.append(str(value))
                        else:
                            row_values.append(str(value))
                    formatted_results.append(", ".join(row_values))
            
            answer = "\n".join(f"- {item}" for item in formatted_results) if len(formatted_results) > 1 else formatted_results[0]
            
            # Finde das relevante HIT für diese Antwort
            hit_id = self._find_best_matching_hit(sparql_query)
            
            if hit_id and hit_id in self.aggregated_results:
                result = self.aggregated_results[hit_id]
                hit_type_id = self.filtered_data[self.filtered_data['HITId'] == hit_id]['HITTypeId'].iloc[0]
                kappa = self.kappas.get(hit_type_id)
                answer_distribution = result['answer_distribution']
                return (answer, kappa, answer_distribution)
            
            return (answer, None, None)
            
        except Exception as e:
            print(f"Fehler bei der Ausführung der SPARQL-Query: {e}")
            return ("Leider konnte ich keine Antwort auf diese Frage finden.", None, None)
    
    def append_crowdsourcing_information(self, answer: str, kappa: float, distribution: dict) -> str:
        """
        Formatiert die Antwort mit Crowdsourcing-Informationen.
        
        Args:
            answer: Die Basis-Antwort
            kappa: Fleiss' Kappa Wert für den Batch
            distribution: Verteilung der Antworten
        
        Returns:
            str: Formatierte Antwort mit Crowdsourcing-Informationen
        """
        if not answer or answer == "Keine Crowd-Antwort verfügbar.":
            return answer
        
        formatted_answer = answer
        
        # Füge Kappa-Information hinzu, wenn verfügbar
        if kappa is not None:
            formatted_answer += f" - according to the crowd, who had an inter-rater agreement of {kappa:.2f} in this batch"
        
        # Füge Verteilungsinformation hinzu, wenn verfügbar
        if distribution:
            support_votes = distribution.get('CORRECT', 0)
            reject_votes = distribution.get('INCORRECT', 0)
            formatted_answer += f". The answer distribution for this specific task was {support_votes} support votes and {reject_votes} reject votes"
        
        formatted_answer += "."
        
        return formatted_answer
    
    def _find_best_matching_hit(self, question, threshold=80):
        """Findet die am besten passende HIT für eine gegebene Frage."""
        best_match = None
        best_score = 0
        
        question = question.lower().strip()
        
        for key, hit_ids in self.hit_mapping.items():
            score = fuzz.ratio(question, key.lower())
            if score > threshold and score > best_score:
                best_score = score
                best_match = hit_ids[0]
        
        return best_match


if __name__ == "__main__":
    # Load the Turtle file
    turtle_file = "./datasources/14_graph.nt"  # Replace with your Turtle file name
    g = Graph()
    g.parse(turtle_file, format="turtle")

    # Define namespace for RDFS (for rdfs:label)
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

    # SPARQL query to extract unique predicates
    query = """
    SELECT DISTINCT ?predicate ?label
    WHERE {
        ?subject ?predicate ?object .
        OPTIONAL { ?predicate <http://www.w3.org/2000/01/rdf-schema#label> ?label }
    }
    """

    # Execute the query
    results = g.query(query)

    # Extract and print the predicates as a list
    print("List of predicates and their labels:")
    for row in results:
        predicate = str(row.predicate)
        label = str(row.label) if row.label else "No label"
        print(f"Predicate: {predicate}, Label: {label}")