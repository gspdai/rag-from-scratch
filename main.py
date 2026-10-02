from llm import Claude
from dotenv import load_dotenv
import os
from rag import load_pdf_pages, KeywordRetriever, EmbeddingRetriever

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

def try_keyword_retriever(question, topk) -> None:
    pages = load_pdf_pages("data/Motor_Insurance_Comprehensive.pdf")
    retriever = KeywordRetriever(pages)
    top_pages = retriever.search(question = question, topk=topk)
    for page in top_pages:
        print(page.number, page.score, page.text)

def try_embedding_retriever(question, topk) -> None:
    pages = load_pdf_pages("data/Motor_Insurance_Comprehensive.pdf")
    retriever = EmbeddingRetriever(pages)
    top_pages = retriever.search(question = question, topk=topk)
    for page in top_pages:
        print(page.number, page.score, page.text)


if __name__ == "__main__":
    try_embedding_retriever("My car got stolen, is it covered?", topk = 2)