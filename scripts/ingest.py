"""CLI: run ingestion pipeline synchronously."""

from hybrid_rag.ingestion.job_manager import IngestJobManager
from hybrid_rag.ingestion.pipeline import run_ingest_pipeline


def main() -> None:
    jobs = IngestJobManager()
    job_id = jobs.create_job()
    run_ingest_pipeline(job_id)
    job = jobs.get_job(job_id)
    print(job)


if __name__ == "__main__":
    main()
