from dotenv import load_dotenv
import openai
import os
from time import sleep

from scraper import scrape_website
from embeddings import split_text, create_embeddings
from pinecone_helper import store_in_pinecone, query_and_generate_response


def scrape_and_store(url, domain, index_name):
    scraped_data = scrape_website(url, domain)
    processed_chunks = split_text(scraped_data)
    texts_to_embed = [x['text'] for x in processed_chunks]
    embeddings = create_embeddings(texts_to_embed)
    return store_in_pinecone(processed_chunks, embeddings, index_name)


def main(index_name, question):
    try:
        response = query_and_generate_response(index_name, question, pinecone_api_key, pinecone_environment)
        return response
    except:
        sleep(5)
        response = query_and_generate_response(index_name, question, pinecone_api_key, pinecone_environment)
        return response


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']
    pinecone_api_key = os.environ['PINECONE_API_KEY']
    pinecone_environment = os.environ["PINECONE_ENV"]
    print("\n")
    print(main(
        index_name="langchain-docs",
        question="How can i create a single prompt task without a conversational chat model?"
    ))