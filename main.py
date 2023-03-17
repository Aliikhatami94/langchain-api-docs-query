from dotenv import load_dotenv
import openai
import os
from time import sleep
import gradio as gr

from scraper import scrape_website
from embeddings import split_text, create_embeddings
from pinecone_helper import store_in_pinecone, query_and_generate_response


def scrape_and_store():
    scraped_data = scrape_website("https://langchain.readthedocs.io/en/latest/", "https://langchain.readthedocs.io/")
    processed_chunks = split_text(scraped_data)
    texts_to_embed = [x['text'] for x in processed_chunks]
    embeddings = create_embeddings(texts_to_embed)
    return store_in_pinecone(processed_chunks, embeddings, "langchain-docs")


def query_langchain(question):
    try:
        response = query_and_generate_response("langchain-docs", question, pinecone_api_key, pinecone_environment)
        return response
    except:
        sleep(5)
        response = query_and_generate_response("langchain-docs", question, pinecone_api_key, pinecone_environment)
        return response


def main():
    # Define the input and output components for the Gradio interface
    input_component = gr.inputs.Textbox(lines=2, placeholder="Enter your question here...", label="Question")
    output_component = gr.Markdown(label="Response")

    # Create the Gradio interface
    iface = gr.Interface(
        fn=query_langchain,  # Function to call
        inputs=input_component,  # Input component
        outputs=output_component,  # Output component
        title="Langchain documentation search",
        description="Enter a question and get a response.",
    )

    # Launch the Gradio interface
    iface.launch()


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.environ['OPENAI_API_KEY']
    pinecone_api_key = os.environ['PINECONE_API_KEY']
    pinecone_environment = os.environ["PINECONE_ENV"]
    main()