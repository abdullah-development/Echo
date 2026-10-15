"""Builds a vocabulary hint for Whisper from the user's docs, so names and jargon
(e.g. "Roomsss", "Kubernetes", "NestJS") are transcribed with the right spelling
instead of the nearest common word."""

import re
from collections import Counter

# Common tech terms Whisper often mishears, used even when docs don't mention them.
TECH_TERMS = [
    "Kubernetes", "Docker", "PostgreSQL", "MySQL", "MongoDB", "Redis", "GraphQL", "gRPC", "REST API",
    "TypeScript", "JavaScript", "Python", "Golang", "Node.js", "Next.js", "React", "NestJS", "Prisma",
    "AWS", "GCP", "Azure", "CI/CD", "DevOps", "Terraform", "Kafka", "Elasticsearch", "Kibana",
    "microservices", "LLM", "RAG", "API", "SQL", "NoSQL", "Linux", "Nginx", "OAuth", "JWT",
]

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.+#/-][A-Za-z0-9+#]+)*")
STOP = {
    "The", "This", "That", "With", "And", "For", "From", "Into", "Over", "Built", "Led", "Owned",
    "Ran", "Used", "Wrote", "Added", "Defined", "Designed", "Delivered", "Engineered", "Generated",
    "Maintained", "Prototyped", "Shipped", "Architected", "Remote", "Present", "Skills", "Experience",
    "Education", "Key", "Achievements", "Name", "Role", "Background", "How", "Answer", "Give", "Don",
    "About", "Software", "Engineer", "Engineering", "Manager", "Lead", "Languages", "Frontend",
    "Backend", "Cloud", "Infra", "Observability", "Multiple", "Clients", "Stack", "Go",
}


def build_vocabulary(text: str, limit: int = 80) -> list[str]:
    tokens = TOKEN.findall(text)
    lower_seen = {t for t in tokens if t.islower()}
    counts: Counter[str] = Counter()
    for t in tokens:
        if t in STOP or len(t) < 2:
            continue
        unusual = (
            any(c.isupper() for c in t[1:])  # NestJS, gRPC, AWS
            or any(c.isdigit() for c in t)  # S3, EC2
            or any(c in ".+#/-" for c in t)  # Next.js, CI/CD
            or (t[0].isupper() and t.lower() not in lower_seen)  # proper nouns: Roomsss, Kibana
        )
        if unusual:
            counts[t] += 1
    terms = [t for t, _ in counts.most_common(limit)]
    for t in TECH_TERMS:
        if t not in terms:
            terms.append(t)
    return terms


def whisper_prompt(terms: list[str], max_words: int = 60) -> str:
    # Whisper reads the prompt as preceding speech; a glossary-style sentence
    # steers spelling without making it hallucinate the words.
    return "Glossary: " + ", ".join(terms[:max_words]) + "."
