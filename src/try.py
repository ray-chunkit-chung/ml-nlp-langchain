from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)


def main():
    if OPENAI_API_KEY is None:
        raise Exception('No API key found')
    print(OPENAI_API_KEY)


if __name__ == '__main__':
    main()
