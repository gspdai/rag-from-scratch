from llm import Claude
from .retriever import KeywordRetriever, EmbeddingRetriever
class RagAnswerer():
    def __init__(self, claude_client: Claude, retriever:KeywordRetriever|EmbeddingRetriever):
        self.client = claude_client
        self.retriever = retriever

    def ask(self, question: str, topk: int = 3) -> str:
        top_pages = self.retriever.search(question, topk)
        joint_pages = ""
        for page in top_pages:
            joint_pages += page.text
        prompt = f'''Please answer to the question "{question}" using the pages {joint_pages}.
        make sure that the answer is given only from the pages provided and not from any other source'''
        response = self.client.ask(prompt)
        return response