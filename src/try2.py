
from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import SequentialChain, LLMMathChain, PALChain
from langchain import LLMChain
from langchain.llms import OpenAI


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document


from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# import langchain
# from langchain.cache import SQLiteCache
# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)

PWD = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(PWD, '..', 'local', 'dragonball.txt')
TXT_PATH = os.path.join(PWD, '..', 'local', 'news_putin_20230331.txt')


def example1():
    """
    https://note.com/npaka/n/n716dfd26094d

    Getting Started

    model_name: str = "text-davinci-003" - model name
    temperature: float = 0.7 - sampling temperature
    max_tokens: int = 256 - maximum tokens (-1: as many as possible)
    top_p: float = 1 - total probability mass of tokens considered per step
    frequency_penalty: float = 0 - Penalty for the repetition frequency of the tokens.
    presence_penalty: float = 0 - penalty for token repetition
    n: int = 1 - Number of completions generated per prompt.
    best_of: int = 1 - return best Completion
    model_kwargs: Dict[str, Any] = Field(default_factory=dict) - unspecified 'create' model parameter
    OpenAI_api_key: Optional[str] = None - OpenAI API key
    batch_size: int = 20 - batch size
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None - timeout (default 600 seconds)

    """

    llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2,
                 openai_api_key=OPENAI_API_KEY)

    result = llm("ネコの鳴き声は？")
    print(result)

    result = llm.generate(["猫の鳴き声は？", "カラスの鳴き声は？"])
    print(result)

    # 出力テキスト
    print("response0:", result.generations[0][0].text)
    print("response1:", result.generations[1][0].text)

    # 使用したトークン数
    print("llm_output:", result.llm_output)


def example2():
    """
    https://note.com/npaka/n/n97aac2da03f4
    """
    # 入力変数のないプロンプトテンプレートの準備
    no_input_prompt = PromptTemplate(
        input_variables=[],
        template="かっこいい動物といえば？"
    )
    print(no_input_prompt.format())

    # 1つの入力変数を持つプロンプトテンプレートの準備
    one_input_prompt = PromptTemplate(
        input_variables=["adjective"],
        template="{adjective}動物といえば？"
    )
    print(one_input_prompt.format(adjective="くそかっこいい"))

    # 複数の入力変数を持つプロンプトテンプレートの準備
    multiple_input_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="{adjective}{content}といえば？"
    )
    print(multiple_input_prompt.format(adjective="かっこいい", content="車"))

    ########################## jinja2 ##########################
    # テンプレートの準備 (jinja2)
    template = """
    {% for item in items %}
    Q: {{ item.question }}
    A: {{ item.answer }}
    {% endfor %}
    """

    # 入力変数の準備
    items = [
        {"question": "foo", "answer": "bar"},
        {"question": "1", "answer": "2"}
    ]

    # jinja2を使ったプロンプトテンプレートの準備
    jinja2_prompt = PromptTemplate(
        input_variables=["items"],
        template=template,
        template_format="jinja2"
    )
    print(jinja2_prompt.format(items=items))

    ########################## fewshot ##########################
    # fewshot 例の準備
    examples = [
        {"input": "明るい", "output": "暗い"},
        {"input": "おもしろい", "output": "つまらない"},
    ]

    # fewshot 例のプロンプト
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="入力: {input}\n出力: {output}",
    )

    # FewShotPromptTemplateの準備
    prompt_from_string_examples = FewShotPromptTemplate(
        examples=examples,  # 例
        example_prompt=example_prompt,  # 例のフォーマット
        prefix="すべての入力の反意語を与えてください",  # プレフィックス
        suffix="入力: {adjective}\n出力:",  # サフィックス
        input_variables=["adjective"],  # 入力変数
        example_separator="\n\n"  # プレフィックスと例とサフィックスを結合する文字

    )
    print(prompt_from_string_examples.format(adjective="大きい"))


def example3():
    """
    https://note.com/npaka/n/n886960b89de1
    https://note.com/npaka/n/n61ad59380a43

    ジェネリックチェーン
    ドキュメントチェーン
    ユーティリティチェーン

    """
    ##########################################################
    # テンプレートの準備
    template = """Q: {question}

    A: 一歩一歩考えてみましょう。"""

    # プロンプトテンプレートの準備
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"]
    )

    # LLMChainの準備
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
        verbose=True
    )

    # LLMChainの実行
    question = "ギターを上達するには？"
    print(llm_chain.predict(question=question))
    ##########################################################

    ##########################################################
    # テンプレートの準備
    template = """{subject}を題材に{adjective}ポエムを書いてください。"""

    # プロンプトテンプレートの準備
    prompt = PromptTemplate(
        template=template,
        input_variables=["adjective", "subject"]
    )

    # LLMChainの準備
    llm_chain = LLMChain(
        prompt=prompt,
        llm=OpenAI(temperature=0),
        verbose=True
    )

    # LLMChainの実行
    print(llm_chain.predict(adjective="かわいい", subject="猫"))
    ##########################################################

    ##########################################################
    # 1つ目のチェーン : 劇のタイトルと時代からあらすじを生成
    llm = OpenAI(temperature=.7)

    # テンプレートの準備
    template = """あなたは劇作家です。劇のタイトルと時代が与えられた場合、そのタイトルのあらすじを書くのがあなたの仕事です。

    タイトル:{title}
    時代:{era}
    あらすじ:"""

    # プロンプトテンプレートの生成
    prompt_template = PromptTemplate(
        input_variables=["title", 'era'],
        template=template
    )

    # LLMChainの準備
    synopsis_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="synopsis"
    )

    # 2つ目のチェーン : 劇のあらすじからレビューを生成

    # LLMの準備準備
    llm = OpenAI(temperature=.7)

    # テンプレートの準備
    template = """あなたは演劇評論家です。 劇のあらすじが与えられた場合、そのあらすじのレビューを書くのがあなたの仕事です。

    あらすじ:
    {synopsis}
    レビュー:"""

    # プロンプトテンプレートの準備
    prompt_template = PromptTemplate(
        input_variables=["synopsis"],
        template=template
    )

    # LLMChainの準備
    review_chain = LLMChain(
        llm=llm, prompt=prompt_template,
        output_key="review"
    )

    # SequentialChainで2つのチェーンを繋ぐ
    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        output_variables=["synopsis", "review"],
        verbose=True
    )

    review = overall_chain({"title": "浜辺の夕暮れ時の悲劇", "era": "戦国時代"})
    print(review)
    ##########################################################

    ##########################################################
    # LLMの準備
    llm = OpenAI(temperature=0)

    # LLMMathChainの準備
    llm_math = LLMMathChain(llm=llm, verbose=True)

    # LLMMathChainの実行
    print(llm_math.run(
        "How many of the integers between 0 and 99 inclusive are divisible by 8?"))

    ##########################################################

    ##########################################################
    # LLMの準備
    llm = OpenAI(
        # model_name='code-davinci-002',
        temperature=0,
        max_tokens=512)

    # PALChainの準備
    pal_chain = PALChain.from_math_prompt(
        llm,
        verbose=True
    )

    # PALChainの実行
    # (JanはMarciaの3倍のペットを飼っています。 MarciaはCindy より2匹多くペットを飼っています。シンディが4匹のペットを飼っている場合、3 人が飼っているペットの総数は?)
    question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"
    print(pal_chain.run(question))


def example4():
    """
    https://note.com/npaka/n/nb9b70619939a

    faiss
    question and answer on a document

    Document Question Answering
    Question answering involves fetching multiple documents, and then asking a question of them. The LLM response will contain the answer to your question, based on the content of the documents.
    """

    # 3. 質問応答

    # 3-1. 関連するチャンクの準備
    with open(TXT_PATH) as f:
        txt = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    # text_splitter = SpacyTextSplitter(chunk_size=1000)
    texts = text_splitter.split_text(txt)
    print(len(texts))

    query = "what did Putin do?"

    # 関連するチャンクの抽出
    embeddings = OpenAIEmbeddings()
    # docsearch = FAISS.from_texts(texts, embeddings)
    docsearch = FAISS.from_texts(texts, embeddings, metadatas=[
                                 {"source": i} for i in range(len(texts))])
    docs = docsearch.similarity_search(query)

    # print(docs)

    # stuffのload_qa_chainを準備
    chain_stuff = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    chain_refine = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0), chain_type="stuff")
    

    chain()

    # 質問応答の実行
    # res_stuff = chain_stuff.run(input_documents=docs, question=query, return_only_outputs=True)
    # print("res_stuff:")
    # print("。\n".join(res_stuff.split(".")))

    # res_refine = chain_refine.run(input_documents=docs, question=query)
    # print("res_refine:")
    # print("。\n".join(res_refine.split(".")))

    response = chain.run(input_documents=docs, question=query)
    print("response:")
    print("。\n".join(response.split(".")))


if __name__ == "__main__":
    # example1()
    # example2()
    # example3()

    example4()
