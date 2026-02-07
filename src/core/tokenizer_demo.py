from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

text = "Hello World!"

# Tokenizza il testo
token_ids = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print(f"Testo: {text}")
print(f"Token IDs: {token_ids}")
print(f"Tokens: {tokens}")
print(f"Numero di token: {len(tokens)}")
