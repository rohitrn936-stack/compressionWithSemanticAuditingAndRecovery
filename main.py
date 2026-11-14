
import json

from compressor import compressPrompt
from semanticAuditor import auditPrompt


# Compression settings
compressionRate = 0.5


def printCompressionStats(result, attemptName):

    print(f"\n{attemptName} COMPRESSION STATISTICS")
    print("Original tokens:", result["original_tokens"])
    print("Compressed tokens:", result["compressed_tokens"])
    print(
        "Actual compression ratio:",
        round(result["compression_ratio"] * 100, 2),
        "%"
    )


def countAuditFailures(auditResult):

    return len(auditResult["missing_or_changed"])


def runPipeline(originalPrompt):


    print("\nPROMPT COMPRESSION PIPELINE")


#compression 1

    print("\n[1] Running first compression...")

    firstCompression = compressPrompt(
        originalPrompt,
        compressionRate=compressionRate
    )

    firstCompressedPrompt = firstCompression["compressed_prompt"]

    print("\nFIRST COMPRESSED PROMPT")
    print(firstCompressedPrompt)

    printCompressionStats(firstCompression, "FIRST")

#2 symantic auditing



    print("\n[2] Running first semantic audit...")

    firstAudit = auditPrompt(
        originalPrompt,
        firstCompressedPrompt
    )

    print("\nFIRST AUDIT RESULT")
    print(json.dumps(firstAudit, indent=4))

#3 recovery or final compressoin output 

    if firstAudit["audit_passed"]:

        print("\n[RESULT] First compression passed.")
        print("No second compression is required.")

        return {
            "final_prompt": firstCompressedPrompt,
            "final_compression": firstCompression,
            "first_audit": firstAudit,
            "second_audit": None,
            "recovery_attempted": False,
            "final_status": "PASSED_FIRST_ATTEMPT"
        }

#recall comes here if the first compression needs looping

    print("\n[RESULT] First compression failed.")
    print("Preparing a second compression attempt...")

    criticalSpans = list(dict.fromkeys(
        item["critical_span"]
        for item in firstAudit["missing_or_changed"]
        if item.get("critical_span")
    ))

    print("\nCRITICAL INSTRUCTIONS TO PRESERVE")

    for span in criticalSpans:
        print("-", span)

    if not criticalSpans:
        print("No critical spans were returned.")
        print("Cannot perform guided recovery.")

        return {
            "final_prompt": firstCompressedPrompt,
            "final_compression": firstCompression,
            "first_audit": firstAudit,
            "second_audit": None,
            "recovery_attempted": False,
            "final_status": "RECOVERY_NOT_POSSIBLE"
        }

#compression no 2 

    print("\n[3] Running second compression...")
    print("Using the original prompt and critical spans.")

    secondCompression = compressPrompt(
        originalPrompt,
        compressionRate=compressionRate,
        forceTokens=criticalSpans
    )

    secondCompressedPrompt = secondCompression["compressed_prompt"]

    print("\nSECOND COMPRESSED PROMPT")
    print(secondCompressedPrompt)

    printCompressionStats(secondCompression, "SECOND")

#2 semantic auditing 

    print("\n[4] Running second semantic audit...")

    secondAudit = auditPrompt(
        originalPrompt,
        secondCompressedPrompt
    )

    print("\nSECOND AUDIT RESULT")
    print(json.dumps(secondAudit, indent=4))



    if secondAudit["audit_passed"]:

        print("\n[RESULT] Recovery successful!")
        print("The second compression passed the audit.")

        finalStatus = "RECOVERED"

    else:

        print("\n[RESULT] Recovery was unsuccessful.")
        print("Some requirements are still missing or changed.")

        finalStatus = "RECOVERY_FAILED"

    return {
        "final_prompt": secondCompressedPrompt,
        "final_compression": secondCompression,
        "first_audit": firstAudit,
        "second_audit": secondAudit,
        "recovery_attempted": True,
        "final_status": finalStatus
    }



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

    result = runPipeline(originalPrompt)

    print("\n")
    print("FINAL REPORT")


    print("\nFinal status:", result["final_status"])
    print("Recovery attempted:", result["recovery_attempted"])

    print("\nFINAL COMPRESSED PROMPT")
    print(result["final_prompt"])

    print("\nFINAL COMPRESSION STATISTICS")
    print(
        json.dumps(
            result["final_compression"],
            indent=4
        )
    )
