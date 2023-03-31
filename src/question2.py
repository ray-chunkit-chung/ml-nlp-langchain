from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)

PWD = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(PWD, '..', 'local', '2005.14165.pdf')


class PdfDocument():

    def __init__(self, path):
        self.path = path
        with open(PDF_PATH, 'rb') as pdf_file:
            # Create a PDF reader object
            reader = PdfReader(self.path)
        self.number_of_pages = len(reader.pages)
        self.text = '\n'.join([page.extract_text() for page in reader.pages])


def main():
    doc = PdfDocument(PDF_PATH)
    print(len(doc.text))


if __name__ == '__main__':
    main()
