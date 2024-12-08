import rdflib
import time
import re
from rdflib.plugins.sparql.parser import parseQuery


class SparQLTask:
    graph = None

    def __init__(self):
        self.graph = rdflib.Graph()
        self.graph.parse('./14_graph.nt', format='turtle')
        while len(self.graph) == 0:
            print("Waiting for the graph to load...")
            time.sleep(1)
        print(f"Parsing complete. Graph contains {len(self.graph)} triples.")

    def is_sparql_query(self, message: str) -> bool:
        # Define a regex pattern for SPARQL queries
        sparql_pattern = r"""
            ^\s*(                    # Optional leading whitespace
                PREFIX\s+[^\n]+\n    # PREFIX definitions (optional, repeated)
                )*                   # Allow multiple PREFIX lines
                (                    # Begin capturing SPARQL core queries
                    SELECT\s+.+      # SELECT followed by any text
                    |ASK\s+WHERE\s*  # ASK followed by WHERE
                    |CONSTRUCT\s+WHERE # CONSTRUCT followed by WHERE
                    |DESCRIBE\s+.+   # DESCRIBE followed by any text
                )                    # End capturing SPARQL core queries
                .*?;?\s*$            # Optional trailing text and semicolon
            """
            # Match the message against the pattern
        return bool(re.match(sparql_pattern, message, re.IGNORECASE | re.VERBOSE))

    def is_valid_sparql(self, query):
        try:
            parseQuery(query)
            return True
        except Exception as e:
            print(f"Invalid SPARQL query: {e}")
            return False

    def process_sparql_query(self, query: str) -> str:
        try:
            if self.is_modifying_query(query):
                # Run the query but alert the user about the potential danger of the action.
                self.execute_sparql_query(query)
                return "The query executed successfully, but it appears to modify the database. Are you sure that was intended?"
            else:
                # Safe read-only query
                results = self.execute_sparql_query(query)
                return self.format_sparql_results(results)
        except Exception as e:
            return f"Error processing SPARQL query: {str(e)}"

    def execute_sparql_query(self, query: str):
        # Execute the query on the graph
        results = self.graph.query(query)
        return results

    @staticmethod
    def format_sparql_results(results) -> str:
        formatted_result = ""
        for row in results:
            formatted_result += "\t".join(str(value) for value in row) + "\n"
        return formatted_result if formatted_result else "No results found."

    def is_modifying_query(self, query: str) -> bool:
        # Check if the query modifies the graph with INSERT, DELETE, etc.
        modifying_keywords = ["INSERT", "DELETE", "UPDATE", "MODIFY"]
        return any(keyword in query.upper() for keyword in modifying_keywords)