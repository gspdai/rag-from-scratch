from llm import Claude
from dotenv import load_dotenv
import os
from rag import load_pdf_pages

def try_claude()-> None:
    load_dotenv()
    key = os.getenv("ANTHROPIC_KEY")
    if key is None:
        raise ValueError("ANTHROPIC_KEY not found in .env")
    
    claude = Claude(key=key)
    response = claude.ask("How are you today")
    print(response)

def try_loader() -> None:
    pages = load_pdf_pages("data/Motor_Insurance_Comprehensive.pdf")
    print(len(pages))
    print(pages[0].text[:300])


if __name__ == "__main__":
    try_loader()