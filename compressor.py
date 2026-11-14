
from llmlingua import PromptCompressor


# LLMLingua-2 configuration
modelName = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

compressor = PromptCompressor(
    model_name=modelName,
    use_llmlingua2=True,
    device_map="mps"
)


def compressPrompt(originalPrompt, compressionRate=0.5, forceTokens=None):

    # Compress the original prompt using LLMLingua-2
    result = compressor.compress_prompt(
        originalPrompt,
        rate=compressionRate,
        force_tokens=forceTokens or []
    )

    compressedPrompt = result["compressed_prompt"]

    originalTokens = result.get("origin_tokens", 0)
    compressedTokens = result.get("compressed_tokens", 0)

    # Calculate the actual compression ratio
    if originalTokens > 0:
        actualCompressionRatio = compressedTokens / originalTokens
    else:
        actualCompressionRatio = 0

    return {
        "compressed_prompt": compressedPrompt,
        "original_tokens": originalTokens,
        "compressed_tokens": compressedTokens,
        "compression_ratio": actualCompressionRatio
    }


# Run this test only when the file is executed directly.
# Importing it into main.py will not run the test.


if __name__ == "__main__":

    originalPrompt = """
    Explain machine learning in simple language.
    Do not use mathematical equations.
    Include an example from healthcare.

    Machine learning is a field of artificial intelligence
    that enables computers to learn patterns from data.
    It is used in healthcare to detect diseases,
    analyse medical images and predict patient outcomes.
    """

    result = compressPrompt(
        originalPrompt,
        compressionRate=0.5,
        forceTokens=["not"]
    )

    print("\nORIGINAL PROMPT ")
    print(originalPrompt)

    print("\nCOMPRESSED PROMPT ")
    print(result["compressed_prompt"])

    print("\nCOMPRESSION STATISTICS ")
    print("Original tokens:", result["original_tokens"])
    print("Compressed tokens:", result["compressed_tokens"])
    print(
        "Actual compression ratio:",
        round(result["compression_ratio"] * 100, 2), "%"
    )
