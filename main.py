from dotenv import load_dotenv
import openai
import os
from scraper import scrape_website
from embeddings import split_text, create_embeddings
from pinecone_helper import store_in_pinecone, query_and_generate_response




def main():
    example_url = "https://langchain.readthedocs.io/en/latest/"
    example_domain = "https://langchain.readthedocs.io/"
    example_index_name = "langchain-docs"
    scraped_data = scrape_website(example_url, example_domain)
    processed_chunks = split_text(scraped_data)
    texts_to_embed = [x['text'] for x in processed_chunks]
    embeddings = create_embeddings(texts_to_embed)
    index = store_in_pinecone(processed_chunks, embeddings, example_index_name)

    index_name = 'langchain-docs'
    pinecone_api_key = os.environ['PINECODE_API_KEY']
    pinecone_environment = os.environ["PINECONE_ENV"]
    question = "How do I use the LLMChain in LangChain?"

    response = query_and_generate_response(index_name, pinecone_api_key, pinecone_environment, question)

    return response


if __name__ == "__main__":
    # Initialize environment
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']

    print(main())