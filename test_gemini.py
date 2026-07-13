from dotenv import load_dotenv
from ai_client import chat_completion

load_dotenv()


def main():
    try:
        response = chat_completion(
            "Use the following context to answer the question.\n\nContext:\n- test\n\nQuestion: hello\nAnswer concisely using the context above.",
            system_instruction="You answer questions using only the provided context.",
            max_output_tokens=1024,
            temperature=0.2,
        )
        print("SUCCESS:", response)
    except Exception as e:
        print("ERROR:", str(e))


if __name__ == "__main__":
    main()
