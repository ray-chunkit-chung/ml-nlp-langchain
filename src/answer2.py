from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

load_dotenv()
PWD = os.path.dirname(os.path.abspath(__file__))
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
PDF_PATH = os.path.join(PWD, '..', 'local', '2005.14165.pdf')


class PdfDocument():
    """
    PDFを読み込み、テキストを取り出すObject
    """

    def __init__(self, pdf_path):
        self.path = pdf_path
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfReader(self.path)
        self.number_of_pages = len(reader.pages)
        self.pages = reader.pages
        self.text = ' '.join([page.extract_text() for page in reader.pages])


def main():
    # PDFを読み込み、Document objectを作る
    pdf_doc = PdfDocument(PDF_PATH)

    # Document objectを適切なチャンクサイズに分割
    text_splitter = SpacyTextSplitter(chunk_size=500)
    texts = text_splitter.split_text(pdf_doc.text)

    # 実際に投げかける質問文
    query = 'On which datasets does GPT-3 struggle?'

    # OpenAIEmbeddings と FAISS オブジェクトを作成し、作ったチャンクをベクトル化し、docsに保存
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings, metadatas=[
                                 {'source': i} for i in range(len(texts))])
    docs = docsearch.similarity_search(query, k=5)

    # 作ったプロンプトを text-davinci-003 へ投げる
    # PromptTemplate クラス使用なし、load_qa_with_sources_chainのinput_documentsを使用し、回答を取得
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0), chain_type='stuff')
    response = chain.run(input_documents=docs, question=query,
                         model_engine='text-davinci-003')

    # 回答をPrint
    print('response:')
    print(response)
    print()


if __name__ == '__main__':
    main()
