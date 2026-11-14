
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": "You are an independent semantic auditor. Identify important instructions and constraints in a prompt."
        },
        {
            "role": "user",
            "content": """
            Original prompt:
            Explain machine learning.
            Do not use mathematical equations.
            Include an example from healthcare.

            Compressed prompt:
            Explain machine learning. Use mathematical equations.
            Include a healthcare example.

            Identify any instructions that were lost or changed.
            """
        }
    ]
)

print(response["message"]["content"])
