import json

def get_distinct_types():
    # Lade die JSON-Datei
    with open('./datasources/images.json', 'r') as file:
        images_data = json.load(file)
    
    # Sammle alle unterschiedlichen Typen
    types = set()
    for entry in images_data:
        if "type" in entry:
            types.add(entry["type"])
    
    # Drucke die Ergebnisse
    print("Gefundene Typen:")
    for t in sorted(types):
        # Zähle wie oft jeder Typ vorkommt
        count = sum(1 for entry in images_data if entry.get("type") == t)
        print(f"- {t}: {count} Bilder")

if __name__ == "__main__":
    get_distinct_types() 