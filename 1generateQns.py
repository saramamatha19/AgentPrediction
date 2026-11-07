import csv
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ---- read agent cards ----
with open("data/agent_cards.csv", encoding="utf-8") as f:
    cards = list(csv.DictReader(f))

rows = []

# ---- generate questions for every agent ----
for card in cards:
    print("Generating for", card["agent"])

    prompt = f"""Here is an AI agent:
- What it does: {card['description']}
- User provides: {card['inputs']}
- Use it when: {card['use_when']}
- Keywords: {card['keywords']}
- NOT for: {card['not_for']}

Write 50 different requests a real employee might type when they need this agent.
Mix styles: formal, casual, very short (2-5 words), vague, describing a problem,
starting with a job role, and some with typos.
Never mention the agent's name.
Return JSON like this: {{"questions": ["request 1", "request 2"]}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    questions = json.loads(response.choices[0].message.content)["questions"]

    for q in questions:
        rows.append({"question": q, "agent": card["agent"]})

# ---- save agent questions ----
with open("data/questions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "agent"])
    writer.writeheader()
    writer.writerows(rows)

print("Saved", len(rows), "questions to data/questions.csv")

# ---- no-match questions ----
print("Generating no-match questions")

prompt = """Write 50 realistic workplace requests that NO AI agent on our platform can do.
Our agents handle: court website scraping, PACER case classification, BRDs/PRDs/documents,
training guides, meeting summaries, UI test running and test writing, resume parsing and screening,
Jira reports and prioritization, supplier risk, invoice extraction, data breach checks,
email writing, news digests, birthday cards on Teams, image generation, and general chat.

Write TASKS that need an action none of these can do (e.g. booking travel, resetting a password,
applying for leave, ordering a laptop). Not general knowledge questions.
Mix casual, short and detailed styles.
Return JSON like this: {"questions": ["request 1", "request 2"]}"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)
no_match = json.loads(response.choices[0].message.content)["questions"]

with open("data/no_match.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question"])
    for q in no_match:
        writer.writerow([q])

print("Saved", len(no_match), "no-match questions to data/no_match.csv")
