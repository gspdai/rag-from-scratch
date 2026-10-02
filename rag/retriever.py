from rag import Page
import re
from dataclasses import dataclass

def _split(text: Page|str):
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

@dataclass(frozen = True)
class _PageWordCount:
    number: int
    text: str
    word_count: dict[str, int]

@dataclass()
class PageScore:
    number: int
    text: str
    score: int

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