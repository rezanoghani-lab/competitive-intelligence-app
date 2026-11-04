"""
Golden-set evaluator for IntelAgent. Run OFFLINE (not inside the demo UI):
    python evals/run_evals.py
Requires GEMINI_API_KEY and EXA_API_KEY in your environment.
"""
import os, json, csv, datetime

GOLDEN = json.load(open("evals/golden_set.json"))
LOG = "evals/results_log.csv"

def competitor_precision_recall(found_names, case):
    f = {n.lower() for n in found_names}
    must = {n.lower() for n in case["must_include"]}
    hits = len(f & must)
    recall = hits / len(must)
    precision = hits / max(len(f), 1)
    return precision, recall

def check_profiles(profiles):
    flags = []
    for p in profiles:
        for k in ["one_liner", "pricing_model", "positioning"]:
            if not p.get(k):
                flags.append(f"missing field {k} in {p.get('name','?')}")
        for feat in p.get("key_features", []):
            if isinstance(feat, dict) and not feat.get("source"):
                flags.append(f"feature without source in {p.get('name','?')}")
    return flags

def run_case(case, run_pipeline, config_version="v2"):
    result = run_pipeline(case["company"], case["product"])
    profiles, critique = result["profiles"], result["critique"]
    names = [p["name"] for p in profiles]
    p, r = competitor_precision_recall(names, case)
    struct_flags = check_profiles(profiles)
    critic_flags = len(critique.get("flags", []))
    conf = result.get("report_confidence")
    score = 0.4 * p + 0.4 * r + (0.1 if not struct_flags else 0) + (0.1 if critic_flags == 0 else 0)
    metrics = {"precision": round(p, 2), "recall": round(r, 2), "critic_flags": critic_flags,
               "struct_flags": len(struct_flags), "confidence": conf, "score": round(min(score, 1), 2)}
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "case_id", "config_version", "precision", "recall",
                        "critic_flags", "struct_flags", "confidence", "score"])
        w.writerow([datetime.datetime.now().isoformat(), case["id"], config_version,
                    metrics["precision"], metrics["recall"], metrics["critic_flags"],
                    metrics["struct_flags"], metrics["confidence"], metrics["score"]])
    print(f"{case['id']}: {metrics}")
    return metrics

if __name__ == "__main__":
    from app import run_pipeline  # headless import; set env keys first
    for case in GOLDEN:
        try:
            run_case(case, run_pipeline)
        except Exception as e:
            print(f"{case['id']}: FAILED - {e}")
