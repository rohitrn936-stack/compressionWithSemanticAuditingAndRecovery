
import ollama
import json
import time
import re
from collections import defaultdict


# ==========================================
# CONFIGURATION
# ==========================================

MODELS = [
    "llama3.2",
    "gemma4:12b",
    "qwen3.5:9b"
]

LOST_STATUSES = {"missing", "weakened", "reversed"}
VALID_STATUSES = {"preserved", "missing", "weakened", "reversed"}


# ==========================================
# BENCHMARK DATASET
# ==========================================

TEST_CASES = [
    {
        "name": "Negation and paraphrasing",
        "original": """
Explain machine learning in simple language.
Do not use mathematical equations.
Include an example from healthcare.
""",
        "compressed": """
Explain machine learning.
Use mathematical equations.
Include a healthcare example.
""",
        "gold": [
            {
                "span": "Explain machine learning in simple language.",
                "status": "weakened"
            },
            {
                "span": "Do not use mathematical equations.",
                "status": "reversed"
            },
            {
                "span": "Include an example from healthcare.",
                "status": "preserved"
            }
        ]
    },

    {
        "name": "Numerical and formatting constraints",
        "original": """
Keep the answer under 100 words.
Use exactly three bullet points.
Do not mention patient names.
""",
        "compressed": """
Keep the answer under 150 words.
Use three bullet points.
Do not mention patient names.
""",
        "gold": [
            {
                "span": "Keep the answer under 100 words.",
                "status": "weakened"
            },
            {
                "span": "Use exactly three bullet points.",
                "status": "preserved"
            },
            {
                "span": "Do not mention patient names.",
                "status": "preserved"
            }
        ]
    },

    {
        "name": "Conditions and source restrictions",
        "original": """
If information is uncertain, explicitly state the uncertainty.
Use only the provided document as your source.
End with a one-sentence summary.
""",
        "compressed": """
State uncertainty when information is uncertain.
Use the provided document and relevant websites.
End with a one-sentence summary.
""",
        "gold": [
            {
                "span": "If information is uncertain, explicitly state the uncertainty.",
                "status": "preserved"
            },
            {
                "span": "Use only the provided document as your source.",
                "status": "weakened"
            },
            {
                "span": "End with a one-sentence summary.",
                "status": "preserved"
            }
        ]
    },

    {
        "name": "Confidentiality and examples",
        "original": """
Explain the result for a beginner.
Never reveal confidential account numbers.
Include one example from finance.
""",
        "compressed": """
Explain the result for a beginner.
Reveal confidential account numbers when relevant.
Include one example from finance.
""",
        "gold": [
            {
                "span": "Explain the result for a beginner.",
                "status": "preserved"
            },
            {
                "span": "Never reveal confidential account numbers.",
                "status": "reversed"
            },
            {
                "span": "Include one example from finance.",
                "status": "preserved"
            }
        ]
    }
]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize(text):
    """Normalize text for exact-span matching."""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?]+$", "", text)

    return text


def call_model(model, messages):

    response = ollama.chat(
        model=model,
        messages=messages,
        format="json",
        options={
            "temperature": 0
        }
    )

    return json.loads(response["message"]["content"])


# ==========================================
# STAGE 1: REQUIREMENT EXTRACTION
# ==========================================

def extract_requirements(model, original):

    messages = [
        {
            "role": "system",
            "content": """
You are a requirement extraction specialist.

Extract every meaningful instruction from the original prompt.

Include:
- Instructions
- Restrictions
- Prohibitions
- Negations
- Numerical constraints
- Required examples
- Output formats
- Conditions and exceptions
- Style and tone requirements

Break compound instructions into atomic requirements.

For every requirement:
- Assign a unique ID.
- Copy its exact original text into original_span.
- Explain its intended meaning.

Do not invent requirements.

Return only JSON:

{
    "requirements": [
        {
            "id": "R1",
            "original_span": "Exact original text",
            "meaning": "Intended meaning"
        }
    ]
}
"""
        },
        {
            "role": "user",
            "content": f"ORIGINAL PROMPT:\n{original}"
        }
    ]

    return call_model(model, messages)


# ==========================================
# STAGE 2: SEMANTIC VERIFICATION
# ==========================================

def verify_requirements(model, original, compressed, requirements):

    messages = [
        {
            "role": "system",
            "content": """
You are an independent semantic preservation evaluator.

Evaluate every extracted requirement against the compressed prompt.

Allowed classifications:

PRESERVED:
The complete intended meaning survives.

MISSING:
The requirement is absent.

WEAKENED:
Some meaning survives, but an important constraint is lost.

REVERSED:
The compressed prompt communicates the opposite meaning.

Rules:
- Compare meaning, not just matching words.
- Equivalent wording counts as preserved.
- Pay particular attention to negations.
- Do not confuse missing with reversed.
- Evaluate every requirement individually.
- Do not invent information.

Return only JSON:

{
    "evaluations": [
        {
            "id": "R1",
            "status": "preserved",
            "reason": "Explanation",
            "compressed_evidence": "Relevant text or empty string"
        }
    ]
}
"""
        },
        {
            "role": "user",
            "content": f"""
ORIGINAL PROMPT:
{original}

COMPRESSED PROMPT:
{compressed}

EXTRACTED REQUIREMENTS:
{json.dumps(requirements, indent=2)}

Evaluate every requirement.
"""
        }
    ]

    return call_model(model, messages)


# ==========================================
# COMPLETE AUDIT
# ==========================================

def audit_case(model, test_case):

    start_time = time.perf_counter()

    extraction = extract_requirements(
        model,
        test_case["original"]
    )

    verification = verify_requirements(
        model,
        test_case["original"],
        test_case["compressed"],
        extraction["requirements"]
    )

    elapsed = time.perf_counter() - start_time

    extracted = extraction["requirements"]
    evaluations = verification["evaluations"]

    # Match model-extracted spans against gold spans.
    extracted_lookup = {
        normalize(item["original_span"]): item
        for item in extracted
    }

    evaluation_lookup = {
        item["id"]: item
        for item in evaluations
    }

    predictions = []

    for gold in test_case["gold"]:

        gold_span = gold["span"]
        gold_status = gold["status"]

        extracted_item = extracted_lookup.get(
            normalize(gold_span)
        )

        if extracted_item is None:

            predicted_status = "not_extracted"
            span_correct = False

        else:

            span_correct = True

            evaluation = evaluation_lookup.get(
                extracted_item["id"]
            )

            if evaluation is None:
                predicted_status = "not_evaluated"
            else:
                predicted_status = (
                    evaluation.get("status", "")
                    .lower()
                    .strip()
                )

                if predicted_status not in VALID_STATUSES:
                    predicted_status = "invalid"

        predictions.append({
            "gold_span": gold_span,
            "gold_status": gold_status,
            "predicted_status": predicted_status,
            "span_correct": span_correct
        })

    return {
        "test_name": test_case["name"],
        "latency": elapsed,
        "predictions": predictions,
        "extracted_count": len(extracted),
        "gold_count": len(test_case["gold"])
    }


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(results):

    tp = 0
    fp = 0
    fn = 0

    correct_classifications = 0
    total = 0

    correct_spans = 0
    total_spans = 0

    for result in results:

        for item in result["predictions"]:

            gold = item["gold_status"]
            predicted = item["predicted_status"]

            actual_loss = gold in LOST_STATUSES
            predicted_loss = predicted in LOST_STATUSES

            if actual_loss and predicted_loss:
                tp += 1

            elif not actual_loss and predicted_loss:
                fp += 1

            elif actual_loss and not predicted_loss:
                fn += 1

            if predicted == gold:
                correct_classifications += 1

            total += 1

            if item["span_correct"]:
                correct_spans += 1

            total_spans += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    accuracy = (
        correct_classifications / total
        if total else 0
    )

    span_accuracy = (
        correct_spans / total_spans
        if total_spans else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_accuracy": accuracy,
        "span_accuracy": span_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_requirements": total
    }


# ==========================================
# MAIN BENCHMARK
# ==========================================

def main():

    all_results = {}
    summary = {}

    for model in MODELS:

        print("\n" + "=" * 55)
        print("EVALUATING:", model)
        print("=" * 55)

        model_results = []

        for test_case in TEST_CASES:

            print("\nTest:", test_case["name"])

            try:

                result = audit_case(model, test_case)

                model_results.append(result)

                print("Latency:", round(result["latency"], 2), "seconds")

                for item in result["predictions"]:
                    print(
                        item["gold_status"],
                        "->",
                        item["predicted_status"]
                    )

            except Exception as error:

                print("ERROR:", str(error))
                print("Skipping this test for this model.")

        if not model_results:
            print("\nNo successful tests for", model)
            continue

        metrics = calculate_metrics(model_results)

        average_latency = (
            sum(item["latency"] for item in model_results)
            / len(model_results)
        )

        summary[model] = {
            **metrics,
            "average_latency_seconds": average_latency,
            "successful_tests": len(model_results)
        }

        all_results[model] = model_results

    # ======================================
    # PRINT COMPARISON
    # ======================================

    print("\n\n")
    print("=" * 85)
    print("FINAL MODEL COMPARISON")
    print("=" * 85)

    print(
        f"{'MODEL':<20}"
        f"{'PRECISION':<12}"
        f"{'RECALL':<12}"
        f"{'F1':<12}"
        f"{'ACCURACY':<12}"
        f"{'LATENCY':<12}"
    )

    for model, metrics in summary.items():

        print(
            f"{model:<20}"
            f"{metrics['precision']:<12.3f}"
            f"{metrics['recall']:<12.3f}"
            f"{metrics['f1']:<12.3f}"
            f"{metrics['classification_accuracy']:<12.3f}"
            f"{metrics['average_latency_seconds']:<12.2f}"
        )

    # ======================================
    # SAVE RESULTS
    # ======================================

    report = {
        "summary": summary,
        "detailed_results": all_results
    }

    with open("model_evaluation_results.json", "w") as file:
        json.dump(report, file, indent=4)

    print("\nResults saved to model_evaluation_results.json")


if __name__ == "__main__":
    main()
