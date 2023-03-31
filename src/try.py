from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai
import os
import requests

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator


load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID', None)

PWD = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(PWD, '..', 'local', '2005.14165.pdf')
TXT_PATH = os.path.join(PWD, '..', 'local', 'news_putin_20230331.txt')


def langchain():
    """ Test the langchain library """
    loader = TextLoader(TXT_PATH)
    index = VectorstoreIndexCreator().from_loaders([loader])

    return index


def load_model(model_engine='text-davinci-003'):
    """Load the GPT-3 language model. Defaults to text-davinci-003"""
    # Set up your OpenAI API key
    openai.api_key = OPENAI_API_KEY
    # embeddings = openai.Embedding.create(model=model_engine)

    response = openai.Embedding.create(
        input="Your text string goes here",
        # Not allowed to generate embeddings from text-davinci-003
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    completion = openai.Completion.create(
        model=model_engine,
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0
    )
    return completion, embeddings


def read_pdf():
    """ Read a PDF file and extract the text """
    # Open the PDF file in binary mode
    with open(PDF_PATH, 'rb') as pdf_file:
        # Create a PDF reader object
        reader = PdfReader(PDF_PATH)
        number_of_pages = len(reader.pages)
        page = reader.pages[0]
        text = '\n'.join([page.extract_text() for page in reader.pages])

        # print(page)
        print(text)
        print(number_of_pages)
        print(len(text))


def main():
    # test, embeddings = load_model()
    # print(test)
    # print(embeddings)

    read_pdf()

    # index = langchain()
    # query = "What did the president say about Ketanji Brown Jackson"
    # index.query(query)


if __name__ == '__main__':
    main()
