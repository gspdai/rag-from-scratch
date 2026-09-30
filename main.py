from llm import Claude
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("ANTHROPIC_KEY")
    if key is None:
        raise ValueError("ANTHROPIC_KEY not found in .env")
    
    claude = Claude(key=key)
    response = claude.ask("How are you today")
    print(response)
