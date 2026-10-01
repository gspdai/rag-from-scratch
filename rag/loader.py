from pypdf import PdfReader
from dataclasses import dataclass

@dataclass(frozen=True)
class Page:
    number: int
    text: str

def load_pdf_pages(path: str) -> list[Page]:
    reader = PdfReader(path)
    pages = []
    for i,page in enumerate(reader.pages,start=1):
        text = page.extract_text()
        text = text.strip()
        if not text:
            continue
        pages.append(Page(number=i, text=text))

    return pages
