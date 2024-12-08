import pandas as pd
from crowdkit.aggregation import MajorityVote
from crowdkit import FleissKappa
from crowdkit import WorkerFilter
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF

# Load crowdsourced data
def load_crowd_data(file_path):
    return pd.read_csv(file_path, sep='\t')

# Filter malicious workers using crowd-kit
def filter_malicious_workers_with_workerfilter(df, min_approval_rate=0.8, max_time_multiplier=5, min_time_multiplier=0.1):
    """
    Filters out malicious workers using WorkerFilter and time-based criteria.

    Args:
        df (pd.DataFrame): Crowdsourced data DataFrame.
        min_approval_rate (float): Minimum approval rate for workers.
        max_time_multiplier (float): Multiplier for upper task completion time threshold.
        min_time_multiplier (float): Multiplier for lower task completion time threshold.

    Returns:
        pd.DataFrame: Filtered DataFrame with non-malicious workers.
    """
    # Step 1: Pre-filter based on time-based criteria
    avg_time = df['WorkTimeInSeconds'].mean()
    time_filtered_df = df[
        (df['LifetimeApprovalRate'] >= min_approval_rate) & 
        (df['WorkTimeInSeconds'] <= avg_time * max_time_multiplier) & 
        (df['WorkTimeInSeconds'] >= avg_time * min_time_multiplier)
    ]

    # Step 2: Use WorkerFilter for majority-based filtering
    worker_filter = WorkerFilter(
        strategy='majority',  # Filter based on majority agreement
        threshold=0.6         # Threshold for agreement with majority
    )
    filtered_workers = worker_filter.fit_predict(time_filtered_df[['WorkerId', 'HITId', 'AnswerLabel']])
    final_filtered_df = time_filtered_df[time_filtered_df['WorkerId'].isin(filtered_workers)]

    return final_filtered_df


# Aggregate answers using MajorityVote from crowd-kit
def aggregate_answers(df):
    """
    Aggregates answers using the majority voting method.

    Args:
        df (pd.DataFrame): Filtered DataFrame with non-malicious workers.

    Returns:
        pd.DataFrame: Aggregated answers with majority votes and distributions.
    """
    answers = df[['HITId', 'WorkerId', 'AnswerLabel']]
    majority_vote = MajorityVote()
    aggregated = majority_vote.fit_predict(answers)
    return aggregated

# Compute Fleiss' kappa using crowd-kit
def compute_fleiss_kappa(df):
    """
    Computes Fleiss' kappa for inter-rater agreement.

    Args:
        df (pd.DataFrame): Filtered DataFrame with non-malicious workers.

    Returns:
        float: Fleiss' kappa score.
    """
    answers = df[['HITId', 'WorkerId', 'AnswerLabel']]
    return FleissKappa().fit(answers)

# Update RDF graph with crowdsourced data
def update_knowledge_graph(graph, crowd_data):
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
