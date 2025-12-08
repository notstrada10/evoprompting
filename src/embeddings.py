import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# Inizializza il client
client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

def compute_similarity(source_sentence: str, sentences: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Calcola la similarità tra una frase sorgente e più frasi target

    Args:
        source_sentence: La frase di riferimento
        sentences: Lista di frasi da comparare
        model: Modello da usare per gli embedding

    Returns:
        Lista di score di similarità
    """
    try:
        result = client.sentence_similarity(
            sentence=source_sentence,
            other_sentences=sentences,
            model=model
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# Esempio d'uso
if __name__ == "__main__":
    source = "That is a happy person"
    sentences = [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    print(f"Source sentence: {source}\n")

    # Prova con modello standard (più affidabile)
    similarities = compute_similarity(source, sentences)

    if similarities:
        print("Similarity scores:")
        for i, (sentence, score) in enumerate(zip(sentences, similarities)):
            print(f"{i+1}. '{sentence}' -> {score:.4f}")
