import csv
from sklearn.model_selection import train_test_split

# read all questions
with open("data/questions.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# split 80/20, keeping each agent's share equal
train, test = train_test_split(
    rows,
    test_size=0.2,                              # 20% for testing
    stratify=[r["agent"] for r in rows],        # split each agent 80/20
    random_state=42,                            # same split every time you run it
)

# save both files
for name, data in [("train", train), ("test", test)]:
    with open(f"data/{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "agent"])
        writer.writeheader()
        writer.writerows(data)
    print(name, ":", len(data), "questions")
