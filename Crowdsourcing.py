import pandas as pd
from collections import Counter
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF

class Crowdsourcing:
    def __init__(self):
        print("Initializing Crowdsourcing module...")
        try:
            # Load data during initialization
            print("Loading crowd data...")
            self.df = self.load_crowd_data('./datasources/crowd_data.tsv')
            
            # Filter malicious workers
            print("Filtering malicious workers...")
            self.filtered_df = self.filter_malicious_workers(self.df)
            
            # Get initial aggregated results
            print("Aggregating answers...")
            self.aggregated_results = self.aggregate_answers(self.filtered_df)
            
            print("Computing basic agreement statistics...")
            self.agreement_stats = self.compute_basic_agreement(self.filtered_df)
            
            print(f"Crowdsourcing initialized with agreement rate: {self.agreement_stats:.2f}")
            print("Initial aggregated results computed")
            
        except Exception as e:
            print(f"Error during Crowdsourcing initialization: {str(e)}")
            raise

    def load_crowd_data(self, file_path):
        return pd.read_csv(file_path, sep='\t')

    def filter_malicious_workers(self, df, min_approval_rate=0.8, max_time_multiplier=5, min_time_multiplier=0.1):
        """
        Filters out malicious workers based on time and approval rate criteria.
        """
        print("Applying worker filtering criteria...")
        
        # Konvertiere LifetimeApprovalRate zu float
        print("Converting approval rate to float...")
        df['LifetimeApprovalRate'] = pd.to_numeric(df['LifetimeApprovalRate'].str.rstrip('%'), errors='coerce') / 100
        
        # Konvertiere WorkTimeInSeconds zu float falls nötig
        print("Converting work time to float...")
        df['WorkTimeInSeconds'] = pd.to_numeric(df['WorkTimeInSeconds'], errors='coerce')
        
        # Berechne Durchschnittszeit
        avg_time = df['WorkTimeInSeconds'].mean()
        print(f"Average work time: {avg_time:.2f} seconds")
        
        # Filtere ungültige Werte
        filtered_df = df.dropna(subset=['LifetimeApprovalRate', 'WorkTimeInSeconds'])
        
        # Wende Filter an
        filtered_df = filtered_df[
            (filtered_df['LifetimeApprovalRate'] >= min_approval_rate) & 
            (filtered_df['WorkTimeInSeconds'] <= avg_time * max_time_multiplier) & 
            (filtered_df['WorkTimeInSeconds'] >= avg_time * min_time_multiplier)
        ]
        
        # Additional filtering based on agreement with majority
        print("Computing worker reliability...")
        worker_reliability = {}
        for hit_id in filtered_df['HITId'].unique():
            hit_answers = filtered_df[filtered_df['HITId'] == hit_id]
            if len(hit_answers) > 0:  # Sicherheitscheck
                majority_answer = hit_answers['AnswerLabel'].mode().iloc[0]
                
                for _, row in hit_answers.iterrows():
                    worker_id = row['WorkerId']
                    if worker_id not in worker_reliability:
                        worker_reliability[worker_id] = {'correct': 0, 'total': 0}
                    
                    worker_reliability[worker_id]['total'] += 1
                    if row['AnswerLabel'] == majority_answer:
                        worker_reliability[worker_id]['correct'] += 1

        # Keep workers with at least 60% agreement with majority
        reliable_workers = [
            worker_id for worker_id, stats in worker_reliability.items()
            if stats['total'] > 0 and (stats['correct'] / stats['total'] >= 0.6)
        ]
        
        final_filtered_df = filtered_df[filtered_df['WorkerId'].isin(reliable_workers)]
        print(f"Filtered from {len(df)} to {len(final_filtered_df)} entries")
        return final_filtered_df

    def aggregate_answers(self, df):
        """
        Aggregates answers using simple majority voting.
        """
        aggregated = {}
        for hit_id in df['HITId'].unique():
            answers = df[df['HITId'] == hit_id]['AnswerLabel']
            majority_vote = Counter(answers).most_common(1)[0][0]
            aggregated[hit_id] = majority_vote
        return aggregated

    def compute_basic_agreement(self, df):
        """
        Computes basic agreement rate between workers.
        """
        majority_answers = self.aggregate_answers(df)
        total_matches = 0
        total_answers = 0
        
        for hit_id in df['HITId'].unique():
            hit_answers = df[df['HITId'] == hit_id]['AnswerLabel']
            majority = majority_answers[hit_id]
            matches = sum(answer == majority for answer in hit_answers)
            total_matches += matches
            total_answers += len(hit_answers)
        
        return total_matches / total_answers if total_answers > 0 else 0.0

    def update_knowledge_graph(self, graph, crowd_data):
        """
        Updates the knowledge graph based on crowdsourced data.
        """
        for _, row in crowd_data.iterrows():
            if row['FinalAnswer'] == 'CORRECT':
                triple = (URIRef(row['Input1ID']), URIRef(row['Input2ID']), Literal(row['Input3ID']))
                graph.add(triple)
            elif row['FinalAnswer'] == 'INCORRECT' and not pd.isna(row['FixValue']):
                if row['FixPosition'] == 'subject':
                    triple = (URIRef(row['FixValue']), URIRef(row['Input2ID']), Literal(row['Input3ID']))
                elif row['FixPosition'] == 'predicate':
                    triple = (URIRef(row['Input1ID']), URIRef(row['FixValue']), Literal(row['Input3ID']))
                elif row['FixPosition'] == 'object':
                    triple = (URIRef(row['Input1ID']), URIRef(row['Input2ID']), Literal(row['FixValue']))
                graph.add(triple)
        return graph
