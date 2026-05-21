import json
import csv
import time
import requests
import os
import random
from collections import Counter

# ── config ────────────────────────────────────────────────────────────────────
URL           = "http://localhost:11434/api/generate"
MODEL         = "llama3.2:3b"
PERSONAS_FILE = "gym_personas.json"
TRN_OUT       = "training_data.csv"
EVAL_OUT      = "gym_members_llm.csv"
TRAIN_PCT     = 0.8
MAX_TRIES     = 3

# ── llm helpers ───────────────────────────────────────────────────────────────
def ask_llm(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300},
    }
    for i in range(MAX_TRIES):
        try:
            res = requests.post(URL, json=payload, timeout=60)
            if res.status_code == 200:
                return res.json()["response"].strip()
        except Exception as e:
            print(f"  api err (try {i+1}): {e}")
        time.sleep(1)
    return None

def extract_json(txt):
    start, end = txt.find("{"), txt.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(txt[start : end + 1])
        except Exception:
            pass
    return None

# ── CORE FIX: hybrid churn scoring ───────────────────────────────────────────
def hybrid_churn_prob(stats: dict, llm_prob: int) -> int:
    """
    Combine deterministic feature-based rules (55 %) with LLM persona
    insight (45 %).  This guarantees observable correlations that a
    logistic-regression / tree model can actually learn.

    Rule contributions (max 100 before blend):
      visits_per_week      → up to +45 pts  (strongest real-world predictor)
      payment_delay_days   → up to +30 pts  (financial friction = churn risk)
      join_month (seasonal)→ up to +15 pts  (resolution / summer joiners)
      honeymoon trap       → up to +20 pts  (joined recently, already disengaged)
      loyalty bonus        → up to -20 pts  (long-term + frequent = committed)
    """
    visits     = float(stats.get("visits_per_week", 2.0))
    delay      = int(stats.get("payment_delay_days", 0))
    months     = float(stats.get("months_since_joined", 6.0))
    join_month = int(stats.get("join_month", 6))
    age        = int(stats.get("age", 30))

    rule = 0

    # 1. Visit frequency — the most predictive feature
    if visits == 0:
        rule += 45
    elif visits < 1.0:
        rule += 35
    elif visits < 2.0:
        rule += 20
    elif visits < 3.0:
        rule += 8
    # 3+ visits/week → no addition (strong retention signal)

    # 2. Payment behaviour
    if delay > 20:
        rule += 30
    elif delay > 10:
        rule += 18
    elif delay > 5:
        rule += 8

    # 3. Seasonal joiners (known gym churn phenomenon)
    if join_month in [1, 2]:        # New Year's resolutioners
        rule += 15
    elif join_month in [6, 7, 8]:   # Summer body seekers
        rule += 10

    # 4. Honeymoon trap: joined recently AND already barely showing up
    if months < 2 and visits < 2:
        rule += 20

    # 5. Loyalty bonus: long-term AND engaged members
    if months > 18 and visits >= 3:
        rule = max(0, rule - 20)

    rule = min(100, max(0, rule))

    # Weighted blend
    final = round(0.55 * rule + 0.45 * llm_prob)
    return int(min(100, max(0, final)))


def churn_label(prob: int, noise: float = 0.08) -> int:
    """
    Deterministic threshold (prob > 50 → churned) with a small random
    label-flip to simulate real-world measurement noise.
    Keeps a clean signal while avoiding a perfectly separable dataset.
    """
    base = 1 if prob > 50 else 0
    if random.random() < noise:
        return 1 - base       # flip ~8% of labels
    return base


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(PERSONAS_FILE):
        print(f"need {PERSONAS_FILE} first — run persona generation script.")
        return

    with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
        peeps = json.load(f)

    random.shuffle(peeps)
    total     = len(peeps)
    split_idx = int(total * TRAIN_PCT)

    print(f"loaded {total} personas.")
    print(f"split → {split_idx} train | {total - split_idx} eval\n")

    trn_cols  = ["age", "months_since_joined", "join_month",
                 "visits_per_week", "payment_delay_days", "churned"]
    eval_cols = ["member_id", "age", "months_since_joined", "join_month",
                 "visits_per_week", "payment_delay_days", "churned",
                 "llm_prob", "rule_prob", "hybrid_prob", "churn_reason"]

    gen_trn = gen_eval = 0
    label_counts = Counter()          # for class-balance diagnostics

    with open(TRN_OUT,  "w", encoding="utf-8", newline="") as f_trn, \
         open(EVAL_OUT, "w", encoding="utf-8", newline="") as f_eval:

        w_trn  = csv.DictWriter(f_trn,  fieldnames=trn_cols)
        w_eval = csv.DictWriter(f_eval, fieldnames=eval_cols)
        w_trn.writeheader()
        w_eval.writeheader()

        for idx, p in enumerate(peeps):
            is_trn  = idx < split_idx
            ds_name = "TRN " if is_trn else "EVAL"

            prompt = f"""You are simulating real gym member data. Look at this member persona:
{json.dumps(p, indent=2)}

Generate realistic gym usage statistics based on their personality and lifestyle.
Remember:
- SLEEPER MEMBERS are real: many people pay but rarely go.
- Use WIDE VARIANCE for churn probability (avoid clustering around 50%).
  Very motivated → 5-20 %, Average/Busy → 25-50 %, Unmotivated → 50-85 %.
- January/February joiners and Summer joiners churn more.
- Busy people visit less often.

Respond ONLY with a JSON object with these exact keys:
- "join_month": integer 1-12
- "months_since_joined": float 0.5-36.0
- "visits_per_week": float 0.0-7.0
- "payment_delay_days": integer 0-30
- "expected_churn_probability": integer 0-100
- "churn_reason": one sentence explaining this specific probability

JSON only, no extra text."""

            res = ask_llm(prompt)
            if not res:
                continue

            stats = extract_json(res)
            if not stats:
                print(f"  [WARN] could not parse json for persona {idx}")
                continue

            try:
                llm_prob  = int(stats.get("expected_churn_probability", 20))
                stats["age"] = p.get("age", 30)       # inject age for rule calc

                # ── hybrid scoring (THE KEY CHANGE) ──────────────────────────
                rule_prob  = hybrid_churn_prob(stats, llm_prob)   # blended
                # For diagnostics we also expose a pure rule score
                stats_no_llm = dict(stats)
                pure_rule  = hybrid_churn_prob(stats, 0)          # rules only

                actual = churn_label(rule_prob)
                label_counts[actual] += 1

                base_data = {
                    "age":                int(stats["age"]),
                    "months_since_joined": float(stats.get("months_since_joined", 6.0)),
                    "join_month":          int(stats.get("join_month", 1)),
                    "visits_per_week":     float(stats.get("visits_per_week", 2.0)),
                    "payment_delay_days":  int(stats.get("payment_delay_days", 0)),
                    "churned":             actual,
                }

                if is_trn:
                    w_trn.writerow(base_data)
                    gen_trn += 1
                    curr = gen_trn
                else:
                    eval_row = base_data.copy()
                    eval_row["member_id"]   = gen_eval + 1
                    eval_row["llm_prob"]    = llm_prob
                    eval_row["rule_prob"]   = pure_rule
                    eval_row["hybrid_prob"] = rule_prob
                    eval_row["churn_reason"] = stats.get("churn_reason", "")
                    w_eval.writerow(eval_row)
                    gen_eval += 1
                    curr = gen_eval

                status = "🔴 CHURNED" if actual == 1 else "🟢 STAYED "
                print(
                    f"[{ds_name}] #{curr:>3}  {status} "
                    f"| llm={llm_prob:>3}% rule={pure_rule:>3}% hybrid={rule_prob:>3}% "
                    f"| visits={base_data['visits_per_week']:.1f} "
                    f"delay={base_data['payment_delay_days']:>2}d"
                )

            except (ValueError, KeyError) as e:
                print(f"  [WARN] skipped persona {idx}: {e}")

    # ── class balance report ──────────────────────────────────────────────────
    total_written = label_counts[0] + label_counts[1]
    churn_rate = label_counts[1] / total_written * 100 if total_written else 0

    print("\n" + "═" * 60)
    print("✅  generation complete!")
    print(f"   {TRN_OUT}  → {gen_trn} rows (training)")
    print(f"   {EVAL_OUT} → {gen_eval} rows (eval, includes hybrid_prob & churn_reason)")
    print(f"\n📊 overall class balance:")
    print(f"   stayed  (0): {label_counts[0]} ({100-churn_rate:.1f}%)")
    print(f"   churned (1): {label_counts[1]} ({churn_rate:.1f}%)")
    if churn_rate < 20 or churn_rate > 80:
        print("   ⚠️  class imbalance detected — train_model.py uses class_weight='balanced' to handle this")
    else:
        print("   ✅ healthy class balance — model should learn well")
    print("═" * 60)


if __name__ == "__main__":
    main()