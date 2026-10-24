"""Daily long-memory summarization job.

Gathers every Project and Session node, asks the small general model for a
3-5 sentence summary, and stores it as a LongMemory node. Does nothing (and
never calls the model) when there is nothing to summarize.

Run by hand, or via Task Scheduler (see install_scheduled_task.ps1). Ollama
must already be running.
"""
import sys

import config
import ollama_client
from graph_memory import MemoryManager
from ollama_client import OllamaError

MAX_INPUT_CHARS = 6000

PROMPT = """Summarize the following notes about a user's projects and chat sessions in 3 to 5 sentences.
Keep concrete facts: names, goals, technologies, decisions, open problems. No preamble.

{notes}

Summary:"""


def main() -> int:
    memory = MemoryManager(config.DB_PATH)
    items = memory.gather_for_summary()
    if not items:
        print("Nothing to summarize.")
        return 0

    if not ollama_client.ensure_running():
        print("Ollama is not running and could not be started.", file=sys.stderr)
        return 1
    notes = "\n\n".join(text for _, text in items)
    if len(notes) > MAX_INPUT_CHARS:
        notes = notes[-MAX_INPUT_CHARS:]  # keep the most recent material
    try:
        summary = ollama_client.generate(
            config.SUMMARIZER_MODEL, PROMPT.format(notes=notes), num_ctx=config.DEFAULT_NUM_CTX)
    except OllamaError as exc:
        print(f"Summarization failed: {exc}", file=sys.stderr)
        return 1
    if not summary:
        print("The model returned an empty summary; nothing stored.", file=sys.stderr)
        return 1

    node_id = memory.add_long_memory(summary, [node_id for node_id, _ in items])
    print(f"Stored {node_id} covering {len(items)} nodes:\n{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
