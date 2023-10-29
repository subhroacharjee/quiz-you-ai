import os

from dotenv import load_dotenv

from src.chains import LLMChainConfig, StreamHandler

load_dotenv()

config = {
    "url": os.getenv("NEO4J_URL"),
    "username": os.getenv("NEO4J_USERNAME"),
    "password": os.getenv("NEO4J_PASSWORD"),
    "cache_folder": "/embedding_model",
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
}


def main():
    llm = LLMChainConfig(config)
    query = """
        Generate a multiple-choice questions. Make sure not to repeat any following questions.`
    """
    with open(os.path.join(os.getcwd(), "assets", "test.pdf"), "rb") as pdf:
        qa = llm.run(pdf)
        for _ in range(10):
            result = qa(
                {
                    "question": query,
                }
            )
            print(result["answer"])


if __name__ == "__main__":
    main()
