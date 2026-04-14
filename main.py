#!/usr/bin/env python3
import os

from agents.supervisor import launch_supervisor

# Group LangSmith traces under a named project — auto-activates when
# LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set in .env.
os.environ.setdefault("LANGCHAIN_PROJECT", "arxiv-ai-research-tracker")


def main() -> None:
    launch_supervisor()


if __name__ == "__main__":
    main()
