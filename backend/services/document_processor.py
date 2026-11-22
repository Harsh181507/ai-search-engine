import os
from pypdf import PdfReader
import docx
import logging

from backend.services.embedding import embed_text
from backend.services.vector_db import store_document
from backend.services.chunking import semantic_chunk
from backend.services.logger import register_document_source

# Logger :- 
logger = logging.getLogger(__name__)

# TEXT EXTRACTION

def _extract_text_from_txt(file_path: str) -> str:
    """
    Extracts text from a .txt file.
    """
    with open(file_path, "r", encoding = "utf-8") as f:
        return f.read()
    

def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a .pdf file.
    """
    reader = PdfReader(file_path)
    text: str = ""
    
    for page in reader.pages:
        page_text: str = page.extract_text()
        if page_text:
            text += page_text + "\n"
        
    return text

def _extract_text_from_docx(file_path: str) -> str:
    """
    Extracts ALL text content from a .docx Word document.
 
    A real .docx file can store text in multiple locations:
 
        PARAGRAPHS — the main body text.
            Includes: headings, normal text, bullet points, numbered lists.
            Accessed via: doc.paragraphs
            Original code only read this — causing 0 chunks for table-heavy docs.
 
        TABLES — grid-structured content.
            Includes: syllabi, CVs, reports, data sheets, course outlines.
            Accessed via: doc.tables → table.rows → row.cells → cell.text
            FIX: Now extracted and joined as readable lines.
 
        HEADERS — text that appears at the top of every page.
            Includes: document title, course name, company name.
            Accessed via: doc.sections → section.header → header.paragraphs
            Often contains key identifying information about the document.
 
        FOOTERS — text that appears at the bottom of every page.
            Includes: page numbers, dates, confidentiality notices.
            Accessed via: doc.sections → section.footer → footer.paragraphs
            Usually less critical but included for completeness.
 
        TEXT BOXES — floating text containers.
            Includes: callout boxes, sidebars, highlighted notes.
            Stored in the XML as drawing elements with text runs.
            Accessed by parsing the underlying XML directly since
            python-docx does not expose text boxes through its high-level API.
 
    Args:
        file_path (str): Path to the .docx file.
 
    Returns:
        str: All extracted text joined with newlines.
             Empty lines are skipped to avoid polluting the chunks.
    """
    doc = docx.Document(file_path)
 
    lines: list[str] = []
 
    # 1. Paragraphs 
    for paragraph in doc.paragraphs:
 
        stripped: str = paragraph.text.strip()
 
        if stripped:
            lines.append(stripped)
 
    # 2. Tables 
    for table in doc.tables:
        for row in table.rows:
            cell_texts: list[str] = []
            for cell in row.cells:
                cell_content: str = cell.text.strip()
                if cell_content:
                    cell_texts.append(cell_content)
 
            if cell_texts:
                row_line: str = " | ".join(cell_texts)
                lines.append(row_line)
 
    # 3. Headers 
    for section in doc.sections:
 
        header = section.header
 
        if header:
            for paragraph in header.paragraphs:
                stripped = paragraph.text.strip()
                if stripped:
                    lines.append(stripped)
 
    # 4. Footers 
        footer = section.footer
 
        if footer:
            for paragraph in footer.paragraphs:
                stripped = paragraph.text.strip()
                if stripped:
                    lines.append(stripped)
 
    # 5. Text Boxes 
 
    try:
        from docx.oxml.ns import qn
 
        body = doc.element.body
 
        for text_element in body.iter(qn("w:t")):
            text_content: str = (text_element.text or "").strip()
            if text_content and len(text_content) > 10:
                lines.append(text_content)
 
    except Exception as e:
        logger.warning(
            "Text box extraction failed for '%s' — skipping: %s",
            file_path,
            e,
        )
 
    # Join and return 
    full_text: str = "\n".join(lines)
 
    return full_text



# MAIN PROCESSING FUNCTION

def process_document(file_path: str, workspace: str) -> int:
    """
    Processes a document by extracting text, chunking it, embedding the chunks,
    and storing them in the vector database.
    """
    filename: str = os.path.basename(file_path)
    logger.info(
        "Processing document | filename='%s' | workspace='%s'",
        filename,
        workspace,
    )

    # Step 1 : Extract text
    if filename.endswith(".txt"):
        text = _extract_text_from_txt(file_path)
    elif filename.endswith(".pdf"):
        text = _extract_text_from_pdf(file_path)    
    elif filename.endswith(".docx"):
        text = _extract_text_from_docx(file_path)
 
    else:
        raise ValueError(
            f"Unsupported file format: '{filename}'. "
            "Supported formats: .txt, .pdf, .docx"
        )
    
    logger.info(
        "Text extracted | filename='%s' | length=%d chars",
        filename,
        len(text),
    )

    if not text.strip():
        logger.warning(
            "No text extracted from '%s' - document may be empty,"
            "image-based, or password-protected.",
            filename
        )
        return 0
    
    # Step 2 : Semantic Chunking
    chunks: list[str] = semantic_chunk(text)
    logger.info(
        "Chunking complete | filename='%s' | chunks=%d",
        filename,
        len(chunks),
    )

    if not chunks:
        logger.warning(
            "No chunks produced from '%s' - text may be too short to chunk.",
            filename,
        )
        return 0
 
    # Step 3 + 4 : Embed and Store Each Chunk
    for i, chunk_text in enumerate(chunks):
        embedding: list[float] = embed_text(chunk_text)
 
        metadata: dict = {
            "text":     chunk_text,
            "source":   filename,
            "chunk_id": i,
        }
 
        store_document(embedding, metadata, workspace)
 
    logger.info(
        "All chunks indexed in Qdrant | filename='%s' | workspace='%s' | chunks=%d",
        filename,
        workspace,
        len(chunks),
    )
 
    #  Step 5: Register in SQLite (Phase D)
    register_document_source(filename, workspace)
    
    return len(chunks)