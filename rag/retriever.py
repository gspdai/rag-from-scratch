from rag import Page
import re
from dataclasses import dataclass
import voyageai
from dotenv import load_dotenv
import os
import numpy as np
import json

load_dotenv()
key = os.getenv("EMBEDDING_KEY")

def _split(text: Page|str) -> dict[str,int]:
    STOP_WORDS = {"is", "an", "the", "to", "and", "how", "this", "out", "of", "a"}
    if type(text) is str:
        text_updated = re.sub(r'[^\w\s]','', text)
        words = text_updated.split()
        words = [w.lower() for w in words if w.lower() not in STOP_WORDS]
        words_count = set(words)
    else:
        text = text.text
        text_updated = re.sub(r'[^\w\s]','', text)
        words = text_updated.split()
        words = [w.lower() for w in words if w.lower() not in STOP_WORDS]
        words_count = {word: words.count(word) for word in set(words)}
    return words_count

    
def _embedding(text: Page|str, key = key) -> list[float]:
    vec = voyageai.Client(api_key = key)
    if type(text) is str:
        text_updated = re.sub(r'[^\w\s]','', text)
        response = vec.embed(texts=[text_updated], model="voyage-4-large", input_type="query")
        embedding = response.embeddings[0]
    else:
        text = text.text
        text_updated = re.sub(r'[^\w\s]','', text)
        response = vec.embed(texts=[text_updated], model="voyage-4-large", input_type="document")
        embedding = response.embeddings[0]
    return embedding

@dataclass(frozen = True)
class _PageWordCount:
    number: int
    text: str
    word_count: dict[str, int]

@dataclass(frozen = True)
class _PageEmbedding:
    number: int
    text: str
    embedding: list[float]

@dataclass()
class PageScore:
    number: int
    text: str
    score: float

class KeywordRetriever():
    def __init__(self, pages: list[Page]):
        self.Page_list = []
        for page in pages:
            page_word_count = _split(page)
            self.Page_list.append(_PageWordCount(number = page.number,
                                                text = page.text, 
                                                word_count = page_word_count)
                                                )

    def search(self, question: str, topk: int =3) -> list[PageScore]:
        if topk < 1 :
            raise ValueError("# of pages request should be atleast 1")
        question_words = _split(question)
        pages_scored = []
        for page in self.Page_list:
            score_list = [page.word_count.get(word,0) for word in set(question_words)]
            score = sum(score_list)
            if score == 0:
                continue
            pages_scored.append(PageScore(number=page.number, text= page.text, score=score))

        top_pages = sorted(pages_scored, key=lambda page: page.score, reverse=True)

        return top_pages[:topk]

class EmbeddingRetriever():
    def __init__(self, pages: list[Page]):
        embeddings_file = "cache/embeddings.json"
        self.Page_list = []
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r') as file:
                saved_embeddings = json.load(file)
            for page in pages:
                self.Page_list.append(_PageEmbedding(number = page.number,
                                        text = page.text, 
                                        embedding= saved_embeddings[str(page.number)])
                                        )
        else:
            embedding_dict = {}
            for page in pages:
                embedding = _embedding(page)
                self.Page_list.append(_PageEmbedding(number = page.number,
                                                    text = page.text, 
                                                    embedding= embedding)
                                                    )
                embedding_dict[page.number] = embedding
            with open(embeddings_file, 'w') as file:
                json.dump(embedding_dict, file)

    def search(self, question: str, topk: int =3) -> list[PageScore]:
        if topk < 1 :
            raise ValueError("# of pages request should be atleast 1")
        question_embed = _embedding(question)
        pages_scored = []
        for page in self.Page_list:
            dot_product = np.dot(question_embed, page.embedding)
            norm_q = np.linalg.norm(question_embed)
            norm_p = np.linalg.norm(page.embedding)
            cosine_similarity = dot_product/(norm_q*norm_p)
            pages_scored.append(PageScore(number=page.number, text= page.text, score=cosine_similarity))

        top_pages = sorted(pages_scored, key=lambda page: page.score, reverse=True)

        return top_pages[:topk]