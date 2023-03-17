import openai
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from tqdm.auto import tqdm
from time import sleep


def split_text(scraped_data, chunk_size=500, chunk_overlap=20):
    tokenizer = tiktoken.get_encoding('p50k_base')

    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for idx, record in enumerate(tqdm(scraped_data)):
        texts = text_splitter.split_text(record['text'])
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[i],
            'chunk': i,
            'url': record['url']
        } for i in range(len(texts))])

    return chunks


def create_embeddings(texts, engine="text-embedding-ada-002"):
    try:
        res = openai.Embedding.create(input=texts, engine=engine)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=engine)
                done = True
            except:
                pass
    return [record['embedding'] for record in res['data']]
