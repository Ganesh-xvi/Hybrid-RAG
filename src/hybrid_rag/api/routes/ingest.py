from fastapi import APIRouter, BackgroundTasks, Depends, status

from hybrid_rag.api.dependencies import get_ingest_job_manager
from hybrid_rag.api.schemas import IngestStartResponse, IngestStatusResponse
from hybrid_rag.api.security import verify_ingest_api_key
from hybrid_rag.config.settings import get_settings
from hybrid_rag.ingestion.job_manager import IngestJobManager
from hybrid_rag.ingestion.pipeline import run_ingest_pipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestStartResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_ingest(
    background_tasks: BackgroundTasks,
    jobs: IngestJobManager = Depends(get_ingest_job_manager),
    _: None = Depends(verify_ingest_api_key),
) -> IngestStartResponse:
    settings = get_settings()
    pdf_count = len(list(settings.data_dir.glob("*.pdf")))
    job_id = jobs.create_job(pdfs_total=pdf_count)
    background_tasks.add_task(run_ingest_pipeline, job_id)
    return IngestStartResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion job queued",
    )


@router.get("/status/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(
    job_id: str,
    jobs: IngestJobManager = Depends(get_ingest_job_manager),
) -> IngestStatusResponse:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return IngestStatusResponse(**job)
