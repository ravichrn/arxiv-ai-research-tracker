#!/usr/bin/env python3
from ingestion.arxiv_fetcher import fetch_and_summarize_papers, TOPICS
from agents.runner import launch_agent

_TOPIC_LABELS = {
    "1": ("cs.AI",  "Artificial Intelligence"),
    "2": ("cs.LG",  "Machine Learning"),
    "3": ("cs.CL",  "Computation & Language (NLP / LLMs)"),
    "4": ("cs.RO",  "Robotics"),
}


def _prompt_topics() -> list[str]:
    print("\nAvailable topics:")
    for key, (code, label) in _TOPIC_LABELS.items():
        print(f"  [{key}] {code} — {label}")
    print("  [5] All of the above")

    raw = input("\nEnter topic numbers (e.g. 1 3) or press Enter for all: ").strip()
    if not raw or raw == "5":
        return list(TOPICS)

    chosen = []
    for char in raw.split():
        entry = _TOPIC_LABELS.get(char)
        if entry:
            chosen.append(entry[0])
        else:
            print(f"  Ignoring unknown option: {char!r}")

    if not chosen:
        print("  No valid topics selected — fetching all.")
        return list(TOPICS)

    return chosen


def main() -> None:
    topics = _prompt_topics()
    print(f"\nFetching topics: {', '.join(topics)}")
    fetch_and_summarize_papers(topics=topics)
    launch_agent()


if __name__ == "__main__":
    main()
