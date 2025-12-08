import os

from dotenv import load_dotenv
from groq import Groq

# Carica le variabili d'ambiente dal file .env
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Parlami dei Beatles",
        }
    ],
    model="llama-3.1-8b-instant",
)

print(chat_completion.choices[0].message.content)
