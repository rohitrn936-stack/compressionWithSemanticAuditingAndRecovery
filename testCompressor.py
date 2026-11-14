
from llmlingua import PromptCompressor

# Load the LLMLingua-2 compression model
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="mps"
)

# Original prompt
prompt = """
Explain machine learning in simple language.
Do not use mathematical equations.
Include an example from healthcare.

Machine learning is a field of artificial intelligence
that enables computers to learn patterns from data.
It is used in healthcare to detect diseases,
analyse medical images and predict patient outcomes.
"""

# Compress the prompt
result = compressor.compress_prompt(
    prompt,
    rate=0.5,
    force_tokens=["not"]
)

# Display results
print("\n--- ORIGINAL PROMPT ---")
print(prompt)

print("\n--- COMPRESSED PROMPT ---")
print(result["compressed_prompt"])

print("\n--- COMPRESSION STATISTICS ---")
print("Original tokens:", result["origin_tokens"])
print("Compressed tokens:", result["compressed_tokens"])
print("Compression ratio:", result["ratio"])
