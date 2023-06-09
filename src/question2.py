from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)

PWD = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(PWD, '..', 'local', '2005.14165.pdf')


class PdfDocument():
    def __init__(self, pdf_path):
        self.path = pdf_path
        with open(pdf_path, 'rb') as pdf_file:
            # Create a PDF reader object
            reader = PdfReader(self.path)
        self.number_of_pages = len(reader.pages)
        self.pages = reader.pages
        self.text = ' '.join([page.extract_text() for page in reader.pages])


def main():
    pdf_doc = PdfDocument(PDF_PATH)
    print('# of char', len(pdf_doc.text))
    print()

    text_splitter = SpacyTextSplitter(chunk_size=500)
    texts = text_splitter.split_text(pdf_doc.text)
    print()

    print('# of text', len(texts))
    print()

    query = 'On which datasets does GPT-3 struggle?'
    # query = 'What datasets are used on which GPT-3 struggles?'
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings, metadatas=[
                                 {'source': i} for i in range(len(texts))])
    docs = docsearch.similarity_search(query, k=5)

    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0), chain_type='stuff')

    response = chain.run(input_documents=docs, question=query,
                         model_engine='text-davinci-003')
    print('response:')
    print(response)
    print()

    # for i in range(10,20):
    # for i in [44, 45, 334, 303, 141]:
    # for i in [44, 179, 196, 334]:
    #     print('source ', i, texts[i])
    #     print()
    #     print()


    # for i in range(10,20):
    # for i in [44, 45, 334, 303, 141]:
    # for i in [44, 179, 196, 334]:
    # for i in [44, 6, 179, 304, 353]:
    #     print('source ', i, texts[i])
    #     print()
    #     print()



if __name__ == '__main__':
    main()
