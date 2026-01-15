import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

# Pydantic V1の警告を抑制（Python 3.14との互換性警告）
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

# 環境変数の読み込み
load_dotenv()

# APIキーの確認
if not os.getenv("OPENAI_API_KEY"):
    print("エラー: OPENAI_API_KEYが設定されていません。")
    print(".envファイルにOPENAI_API_KEY=your_api_key_here を追加してください。")
    exit(1)

# PDFを読み込む
pdf_file = "sample.pdf"
if not os.path.exists(pdf_file):
    print(f"エラー: {pdf_file} が見つかりません。")
    print("同じディレクトリにPDFファイルを配置してください。")
    exit(1)

try:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    if not documents:
        print(f"エラー: {pdf_file} からドキュメントを読み込めませんでした。")
        exit(1)
except Exception as e:
    print(f"エラー: PDFファイルの読み込みに失敗しました: {e}")
    exit(1)

# テキストの分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
texts = text_splitter.split_documents(documents)

# 埋め込みモデルの読み込み
try:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as e:
    print(f"エラー: 埋め込みモデルの読み込みに失敗しました: {e}")
    exit(1)

# FAISSベクトルストアの作成
try:
    vectorstore = FAISS.from_documents(texts, embedding)
except Exception as e:
    print(f"エラー: ベクトルストアの作成に失敗しました: {e}")
    exit(1)

# 質問の受け取り
query = "この文章について教えて"
results = vectorstore.similarity_search(query, k=3)

print("\n=== 検索結果 ===")
for i, doc in enumerate(results):
    print(f"\n--- 結果{i+1} ---")
    print(doc.page_content[:200])

# OpenAI APIの設定
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"エラー: OpenAIクライアントの初期化に失敗しました: {e}")
    exit(1)

# 検索結果を結合して文脈を作成
if not results:
    print("警告: 検索結果がありません。")
    context = ""
else:
    context = "\n\n".join([doc.page_content for doc in results])

print("\n=== AIによる回答 ===")
# ChatGPTに質問
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # またはgpt-4
        messages=[
            {
                "role": "system",
                "content": "あなたは親切なアシスタントです。提供された文脈に基づいて、正確に回答してください。"
            },
            {
                "role": "user",
                "content": f"以下の文脈を参考に質問に答えてください。\n\n文脈:\n{context}\n\n質問: {query}"
            }
        ],
        temperature=0.7,
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"エラー: OpenAI APIの呼び出しに失敗しました: {e}")
    print("APIキーが正しいか、ネットワーク接続を確認してください。")
    exit(1)