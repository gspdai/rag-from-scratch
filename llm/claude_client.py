from anthropic import Anthropic

class Claude:
    def __init__(self, key: str, max_token: int = 1024, model: str = "claude-opus-5-5") -> None:
        self.max_token = max_token
        self.model = model
        self.client = Anthropic(api_key = key)

    def ask(self, question: str) -> str:
        message = self.client.messages.create(
            max_tokens = self.max_token,
            model = self.model,
            messages = [{"role": "user", "content": question}]
        )
        for block in message.content:
            if block.type == "text":
                return block.text
        raise ValueError("Claude Returned no response")