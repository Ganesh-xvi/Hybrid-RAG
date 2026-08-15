from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a legal document assistant specializing in employment agreements.
Answer questions using ONLY the provided context from Honeywell and/or Cloudflare employment contracts.
Always cite the company name and page number when referencing specific clauses.
If the answer is not in the context, say you do not have enough information.
Be concise and factual."""


def build_prompt(query: str, documents: list[Document]) -> list[dict[str, str]]:
    context_blocks = []
    for doc in documents:
        company = doc.metadata.get("company", "unknown")
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(
            f"[{company} | page {page} | {source}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]
