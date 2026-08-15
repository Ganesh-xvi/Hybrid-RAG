from pathlib import Path

import pdfplumber
from langchain_core.documents import Document

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.reliability.dlq import DLQManager
from hybrid_rag.utils.company import extract_company
from hybrid_rag.utils.logging import logger
from hybrid_rag.utils.retry import retry


@retry()
def _load_single_pdf(path: Path) -> list[Document]:
    documents: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "company": extract_company(path.name),
                        "page": page_num,
                        "type": "employment_agreement",
                    },
                )
            )
    return documents


def load_pdfs_from_folder(
    data_dir: Path | None = None,
    settings: Settings | None = None,
    dlq: DLQManager | None = None,
) -> list[Document]:
    cfg = settings or get_settings()
    folder = data_dir or cfg.data_dir
    folder.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder.resolve()}")

    dead_letter = dlq or DLQManager(cfg)
    all_documents: list[Document] = []

    for pdf_path in pdf_files:
        try:
            docs = _load_single_pdf(pdf_path)
            all_documents.extend(docs)
            logger.info("Loaded %s pages from %s", len(docs), pdf_path.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", pdf_path.name, exc)
            dead_letter.push(
                {
                    "file": str(pdf_path),
                    "error": str(exc),
                    "stage": "pdf_load",
                }
            )

    return all_documents
