import csv
from collections import Counter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- cards -> embeddings (same as Piece 2) ----
with open("data/agent_cards.csv", encoding="utf-8") as f:
    cards = list(csv.DictReader(f))
names = [c["agent"] for c in cards]
card_texts = [c["description"] + " " + c["use_when"] + " " + c["keywords"] for c in cards]
card_embeddings = model.encode(card_texts)

# ---- test questions -> embeddings ----
with open("data/test.csv", encoding="utf-8") as f:
    test = list(csv.DictReader(f))
questions = [r["question"] for r in test]
answers = [r["agent"] for r in test]
q_embeddings = model.encode(questions)

# ---- compare every question with every card ----
scores = model.similarity(q_embeddings, card_embeddings)    # 267 x 25 table

top1_correct = 0
top3_correct = 0
mistakes = Counter()

for i in range(len(test)):
    ranked = scores[i].argsort(descending=True)            # agent positions, best first
    top3 = [names[j] for j in ranked[:3]]

    if top3[0] == answers[i]:
        top1_correct += 1
    else:
        mistakes[(answers[i], top3[0])] += 1               # (real agent, wrongly predicted)
    if answers[i] in top3:
        top3_correct += 1

print(f"Top-1 accuracy: {top1_correct / len(test):.1%}")
print(f"Top-3 accuracy: {top3_correct / len(test):.1%}")

print("\nMost common mistakes (real -> predicted):")
for (real, pred), count in mistakes.most_common(10):
    print(f"  {real} -> {pred}: {count}")
