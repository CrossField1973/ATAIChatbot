from rdflib import Graph, Namespace
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        labels = self.data[idx]["labels"]
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2id[labels[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids)
        }

class FineTunedBert:
    def __init__(self):
        print("Initialisiere FineTunedBert...")
        self.g = Graph()
        try:
            self.g.parse("./datasources/14_graph.nt", format="turtle")
            print(f"Graph geladen. Anzahl Tripel: {len(self.g)}")
        except Exception as e:
            print(f"Fehler beim Laden des Graphen: {e}")
            raise

        self.label2id = {
            "O": 0,
            "B-MOVIE": 1,
            "B-REL": 2,
            "B-ENTITY": 3
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = None
        
        self.extract_movies()
        self.extract_predicates()
        self.extract_training_data()
        self.prepare_ner_data()

    def extract_movies(self):
        print("\nExtrahiere Filme...")
        MOVIE_INSTANCE_PREDICATE = "http://www.wikidata.org/prop/direct/P31"
        MOVIE_TYPE = "http://www.wikidata.org/entity/Q11424"

        self.movies = set()
        for subj, pred, obj in self.g:
            if str(pred) == MOVIE_INSTANCE_PREDICATE and str(obj) == MOVIE_TYPE:
                self.movies.add(str(subj))

        print(f"Gefundene Filme: {len(self.movies)}")
        if len(self.movies) == 0:
            print("WARNUNG: Keine Filme gefunden!")
            print("Verfügbare Prädikate:", set(str(p) for _, p, _ in self.g))
            print("Verfügbare Objekte:", set(str(o) for _, _, o in self.g))

    def extract_predicates(self):
        print("\nExtrahiere Prädikate...")
        self.predicates = set(str(pred) for _, pred, _ in self.g)
        print(f"Gefundene Prädikate: {len(self.predicates)}")
        if len(self.predicates) == 0:
            print("WARNUNG: Keine Prädikate gefunden!")

    def extract_training_data(self):
        print("\nExtrahiere Trainingsdaten...")
        self.training_data = []
        for subj, pred, obj in self.g:
            if str(subj) in self.movies:
                self.training_data.append((str(subj), str(pred), str(obj)))

        print(f"Extrahierte Trainingsdaten: {len(self.training_data)}")
        if len(self.training_data) == 0:
            print("WARNUNG: Keine Trainingsdaten gefunden!")
        else:
            print("Beispiel-Trainingsdaten:")
            for i, data in enumerate(self.training_data[:3]):
                print(f"  {i+1}. {data}")

    def prepare_ner_data(self):
        print("\nBereite NER-Daten vor...")
        self.ner_data = []
        self.relation_data = []

        for subj, pred, obj in self.training_data:
            # Erstelle einen lesbaren Satz
            sentence = f"The movie {subj} has the relation {pred} with {obj}."
            tokens = sentence.split()
            
            # Erstelle Labels
            labels = []
            for token in tokens:
                if token == subj:
                    labels.append("B-MOVIE")
                elif token == pred:
                    labels.append("B-REL")
                elif token == obj:
                    labels.append("B-ENTITY")
                else:
                    labels.append("O")

            self.ner_data.append({
                "tokens": tokens,
                "labels": labels
            })

            self.relation_data.append({
                "sentence": sentence,
                "subject": subj,
                "object": obj,
                "relation": pred
            })

        print(f"Erstellte NER-Datensätze: {len(self.ner_data)}")
        print(f"Erstellte Relations-Datensätze: {len(self.relation_data)}")
        
        if len(self.ner_data) == 0:
            print("WARNUNG: Keine NER-Daten erstellt!")
        else:
            print("\nBeispiel NER-Daten:")
            for i, data in enumerate(self.ner_data[:2]):
                print(f"\nBeispiel {i+1}:")
                print("Tokens:", data["tokens"])
                print("Labels:", data["labels"])

    def train_ner_model(self):
        print("\nStarte Training...")
        if len(self.ner_data) == 0:
            raise ValueError("Keine Trainingsdaten verfügbar!")

        dataset = NERDataset(self.ner_data, self.tokenizer, self.label2id)
        print(f"Dataset erstellt mit {len(dataset)} Beispielen")
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        print(f"Training-Set: {len(train_dataset)} Beispiele")
        print(f"Test-Set: {len(test_dataset)} Beispiele")

        self.model = AutoModelForTokenClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        print("Starte Training-Loop...")
        trainer.train()
        print("Training abgeschlossen")

    def analyze_question(self, sentence):
        if self.model is None:
            raise ValueError("Model muss zuerst trainiert werden!")
            
        # Tokenisierung
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        
        # Modellvorhersage
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Token-IDs zurück in Wörter umwandeln
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Entitäten und Relationen extrahieren
        entity_id = None
        relation_id = None
        
        for token, pred_id in zip(tokens, predictions[0]):
            label = self.id2label[pred_id.item()]
            if label == "B-MOVIE":
                entity_id = token
            elif label == "B-REL":
                relation_id = token
                
        return entity_id, relation_id

    def query_graph(self, entity_id, relation_id):
        if not entity_id or not relation_id:
            return []
            
        results = []
        # Normalisiere die IDs für den Vergleich
        entity_id = str(entity_id).lower()
        relation_id = str(relation_id).lower()
        
        for s, p, o in self.g:
            if str(s).lower().find(entity_id) != -1 and str(p).lower().find(relation_id) != -1:
                results.append(str(o))
        
        return results

    def process_question(self, question):
        try:
            print(f"\nAnalysiere Frage: {question}")
            entity_id, relation_id = self.analyze_question(question)
            
            if entity_id and relation_id:
                print(f"Gefundene Entität: {entity_id}")
                print(f"Gefundene Relation: {relation_id}")
                results = self.query_graph(entity_id, relation_id)
                return results
            else:
                return "Konnte keine Entitäten oder Relationen in der Frage finden."
                
        except Exception as e:
            return f"Fehler bei der Verarbeitung: {str(e)}"

if __name__ == "__main__":
    try:
        print("Programm startet...")
        ftb = FineTunedBert()
        ftb.train_ner_model()

        example_sentences = [
            "Wer ist der Regisseur von Inception?",
            "Welches ist das Genre von The Matrix?",
            "Wann wurde Avatar veröffentlicht?",
            "Wo wurde Herr der Ringe gedreht?"
        ]

        for sentence in example_sentences:
            results = ftb.process_question(sentence)
            print(f"Ergebnisse für '{sentence}': {results}")
            
    except Exception as e:
        print(f"Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()