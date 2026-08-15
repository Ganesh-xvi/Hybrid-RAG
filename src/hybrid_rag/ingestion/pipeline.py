from pathlib import Path

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.job_manager import IngestJobManager
from hybrid_rag.ingestion.loader import load_pdfs_from_folder
from hybrid_rag.reliability.dlq import DLQManager
from hybrid_rag.retrieval.qdrant_store import QdrantStore
from hybrid_rag.utils.logging import logger


def run_ingest_pipeline(
    job_id: str,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> None:
    cfg = settings or get_settings()
    jobs = IngestJobManager(cfg)
    dlq = DLQManager(cfg)
    store = QdrantStore(cfg)
    jobs.mark_running(job_id, stage="starting", progress=1)

    try:
        folder = data_dir or cfg.data_dir
        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {folder.resolve()}")

        jobs.update_job(job_id, pdfs_total=len(pdf_files))
        jobs.mark_running(job_id, stage="loading_pdfs", progress=10)

        documents = load_pdfs_from_folder(folder, cfg, dlq)
        jobs.mark_running(job_id, stage="chunking", progress=30)

        chunks = chunk_documents(documents, cfg)
        jobs.mark_running(job_id, stage="upserting_chunks", progress=50)

        upserted = store.upsert_chunks(chunks)
        jobs.update_job(job_id, chunks_upserted=upserted, pdfs_processed=len(pdf_files))

        jobs.mark_completed(
            job_id,
            pdfs=len(pdf_files),
            chunks=upserted,
            dlq_count=dlq.count(),
        )
        logger.info("Ingest job %s completed: %s chunks", job_id, upserted)
    except Exception as exc:
        logger.exception("Ingest job %s failed", job_id)
        jobs.mark_failed(job_id, str(exc))
