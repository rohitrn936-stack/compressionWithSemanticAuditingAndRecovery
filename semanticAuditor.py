
import ollama
import json


modelName = "gemma4:12b"


def extractRequirements(originalPrompt):

    response = ollama.chat(
        model=modelName,
        format="json",
        keep_alive="5m",
        messages=[
            {
                "role": "system",
                "content": """
You are a requirement extraction specialist.

Extract every meaningful requirement from the original prompt.

Include:
- Instructions
- Restrictions
- Prohibitions
- Negations
- Numerical constraints
- Required examples
- Output format requirements
- Conditions and exceptions
- Style and tone requirements

Break compound instructions into smaller, individual
requirements wherever appropriate.

For every requirement:
1. Assign a unique ID.
2. Copy the exact original text into original_span.
3. Briefly explain what the requirement means.

Do not invent requirements or change the original wording.

Return only JSON in this structure:

{
    "requirements": [
        {
            "id": "R1",
            "original_span": "Exact text from original prompt",
            "meaning": "What this instruction requires"
        }
    ]
}
"""
            },
            {
                "role": "user",
                "content": f"ORIGINAL PROMPT:\n{originalPrompt}"
            }
        ]
    )

    return json.loads(response["message"]["content"])


def verifyRequirements(originalPrompt, compressedPrompt, requirements):

    response = ollama.chat(
        model=modelName,
        format="json",
        keep_alive="5m",
        messages=[
            {
                "role": "system",
                "content": """
You are an independent semantic preservation evaluator.

You will receive:
1. The original prompt.
2. Its compressed version.
3. Requirements extracted from the original prompt.

Evaluate EVERY requirement individually.

Use only these classifications:

PRESERVED:
The complete intended meaning is retained, even if
the wording is different.

MISSING:
The requirement is absent from the compressed prompt.

WEAKENED:
Some of the requirement survives, but an important
part of its meaning is lost.

REVERSED:
The compressed prompt communicates the opposite
of the original requirement.

Important evaluation instructions:

- Compare meaning, not just matching words.
- Pay particular attention to negations.
- Distinguish missing instructions from reversed ones.
- Equivalent wording must count as preserved.
- Do not assume a requirement is missing just because
  its original wording has changed.
- Evaluate every requirement separately.
- Use the original_span provided during extraction.
- Never invent or rewrite the original_span.
- Provide evidence from the compressed prompt when available.

Return only JSON in this structure:

{
    "evaluations": [
        {
            "id": "R1",
            "status": "PRESERVED",
            "reason": "Explanation of the classification",
            "compressed_evidence": "Relevant text or empty string"
        }
    ],
    "audit_passed": false
}

audit_passed must be false if any requirement is
missing, weakened or reversed.
"""
            },
            {
                "role": "user",
                "content": f"""
ORIGINAL PROMPT:
{originalPrompt}

COMPRESSED PROMPT:
{compressedPrompt}

EXTRACTED REQUIREMENTS:
{json.dumps(requirements, indent=2)}

Evaluate every extracted requirement.
"""
            }
        ]
    )

    return json.loads(response["message"]["content"])


def auditPrompt(originalPrompt, compressedPrompt):

    # First LLM call: extract the requirements
    requirements = extractRequirements(originalPrompt)

    print("\nEXTRACTED REQUIREMENTS")
    print(json.dumps(requirements, indent=4))

    extractedRequirements = requirements.get("requirements", [])

    if not extractedRequirements:
        raise ValueError(
            "No requirements were extracted. "
            "The semantic audit cannot continue."
        )

    # Second LLM call: check whether the requirements survived
    verification = verifyRequirements(
        originalPrompt,
        compressedPrompt,
        extractedRequirements
    )

    # Create a lookup so each evaluation can be matched
    # with its corresponding original requirement
    requirementLookup = {
        item["id"]: item
        for item in extractedRequirements
    }

    finalReport = {
        "preserved": [],
        "missing_or_changed": [],
        "audit_passed": False
    }

    evaluatedIds = set()

    for evaluation in verification.get("evaluations", []):

        requirementId = evaluation.get("id")
        requirement = requirementLookup.get(requirementId)

        if requirement is None:
            continue

        evaluatedIds.add(requirementId)

        # Convert the status to lowercase so that
        # uppercase and lowercase responses both work
        status = evaluation.get("status", "").strip().lower()

        reportItem = {
            "id": requirementId,
            "original_instruction": requirement["original_span"],
            "status": status.upper(),
            "problem": evaluation.get("reason", ""),
            "compressed_evidence": evaluation.get(
                "compressed_evidence", ""
            ),
            "critical_span": requirement["original_span"]
        }

        if status == "preserved":
            finalReport["preserved"].append(reportItem)

        elif status in ["missing", "weakened", "reversed"]:
            finalReport["missing_or_changed"].append(reportItem)

        else:
            # An unknown status should never be treated as preserved
            reportItem["status"] = "UNKNOWN"
            reportItem["problem"] = (
                "The auditor returned an unrecognized status. "
                + reportItem["problem"]
            )

            finalReport["missing_or_changed"].append(reportItem)

    # Check whether the verification model skipped any requirements
    for requirement in extractedRequirements:

        if requirement["id"] not in evaluatedIds:

            finalReport["missing_or_changed"].append({
                "id": requirement["id"],
                "original_instruction": requirement["original_span"],
                "status": "NOT_EVALUATED",
                "problem": "The verification model did not evaluate this requirement.",
                "compressed_evidence": "",
                "critical_span": requirement["original_span"]
            })

    # The audit passes only if every extracted requirement is preserved
    finalReport["audit_passed"] = (
        len(finalReport["missing_or_changed"]) == 0
    )

    return finalReport


# Run this test only when this file is executed directly.
# Importing this file into main.py will not run the test.

if True:
    if __name__ == "__main__":

        originalPrompt = """
        Explain machine learning in simple language.
        Do not use mathematical equations.
        Include an example from healthcare.
        """

        compressedPrompt = """
        Explain machine learning.
        Use mathematical equations.
        Include a healthcare example.
        """

        result = auditPrompt(originalPrompt, compressedPrompt)

        print("\nFINAL SEMANTIC AUDIT REPORT")
        print(json.dumps(result, indent=4))
