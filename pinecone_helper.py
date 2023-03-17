import pinecone
from tqdm.auto import tqdm
import openai
import os

from embeddings import create_embeddings


def store_in_pinecone(chunks, embeddings, index_name):
    pinecone.init(
        api_key=os.environ['PINECODE_API_KEY'],
        environment=os.environ["PINECONE_ENV"]
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=len(embeddings[0]),
            metric='dotproduct'
        )

    index = pinecone.Index(index_name)
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size)):
        i_end = min(len(chunks), i + batch_size)
        meta_batch = chunks[i:i_end]
        ids_batch = [x['id'] for x in meta_batch]
        embeds_batch = embeddings[i:i_end]
        meta_batch = [{
            'text': x['text'],
            'chunk': x['chunk'],
            'url': x['url']
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds_batch, meta_batch))
        index.upsert(vectors=to_upsert)
    return index


def query_and_generate_response(index_name, api_key, environment, query, model="gpt-3.5-turbo", engine="text-embedding-ada-002", top_k=5):
    # Initialize Pinecone
    pinecone.init(
        api_key=api_key,
        environment=environment
    )

    # Connect to the Pinecone index
    index = pinecone.Index(index_name)

    res = openai.Embedding.create(
        input=[query],
        engine=engine
    )

    # Retrieve from Pinecone
    query_embedding = res['data'][0]['embedding']
    pinecone_results = index.query(query_embedding, top_k=top_k, include_metadata=True)

    contexts = [item['metadata']['text'] for item in pinecone_results['matches']]
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

    # Set up system rules
    system_rules = "You are a helpful assistant who helps with answering questions based on the provided information. If the information cannot be found in the text the user provides you, you truthfully say I don't know."

    res = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_rules},
            {"role": "user", "content": augmented_query}
        ]
    )

    return res['choices'][0]['message']['content']
