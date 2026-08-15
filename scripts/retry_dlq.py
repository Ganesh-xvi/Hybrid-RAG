"""CLI: retry failed ingestion jobs from DLQ."""

from hybrid_rag.reliability.dlq import DLQManager


def main() -> None:
    dlq = DLQManager()
    while True:
        job = dlq.pop_for_retry()
        if not job:
            print("No failed jobs in DLQ.")
            break
        print(f"Retrying: {job}")


if __name__ == "__main__":
    main()
