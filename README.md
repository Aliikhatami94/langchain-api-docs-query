# Langchain Documentation Search
This technical documentation describes a Python code for searching and retrieving Langchain API documentation. The code consists of four modules: scraper, embeddings, pinecone_helper, and main. The scraper module is used to scrape Langchain API documentation from the web. The embeddings module is used to split the scraped data into smaller chunks and create embeddings for each chunk. The pinecone_helper module is used to store the embeddings in Pinecone, a vector search engine. The main module uses Gradio to create a user interface for searching the Pinecone index.

### Prerequisites
The code requires Python 3.7 or higher and the following Python packages:

* dotenv
* openai
* os
* time
* gradio
* tqdm
* requests
* beautifulsoup4
* urllib3
* html
* re

To install these packages, run the following command in your terminal:
```
pip install -r requirements.txt
```
To use the code, you need to follow these steps:

Clone the repository from GitHub: git clone https://github.com/Aliikhatami94/langchain-api-docs-query.git.

Navigate to the repository directory: cd repo.
Create a .env file and add your OpenAI API key, Pinecone API key, and Pinecone environment name in the following format:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment_name
```
You have to first run the scrape and embedding and once you have stored the embeddings, then you can run the main module using the following command:
```
python main.py
```
The Gradio interface will be launched in your default web browser. Enter your question in the input textbox and click the "Submit" button. The response will be displayed in the output textbox.

## Modules

### scraper
The scraper module is used to scrape Langchain API documentation from the web. The scrape_website function takes two parameters: the URL of the Langchain API documentation and the domain name of the website. It returns a list of dictionaries, where each dictionary contains the URL of a page and its main content.

### embeddings
The embeddings module is used to split the scraped data into smaller chunks and create embeddings for each chunk. The split_text function takes the scraped data as input and splits it into chunks of a specified size. The size of the chunks and the overlap between them can be adjusted using the chunk_size and chunk_overlap parameters. The create_embeddings function takes the list of texts as input and returns a list of embeddings.

### pinecone_helper
The pinecone_helper module is used to store the embeddings in Pinecone, a vector search engine. The store_in_pinecone function takes the chunks, embeddings, and the name of the Pinecone index as input. It creates the Pinecone index if it doesn't exist and stores the embeddings in it. The query_and_generate_response function takes the name of the Pinecone index, the user's query, the Pinecone API key, and the Pinecone environment name as input. It retrieves the embeddings from Pinecone and generates a response to the user's query using OpenAI's GPT-3.5 model.

### main
The main module uses Gradio to create a user interface for searching the Pinecone index. It defines the input and output components for the Gradio interface, creates the Gradio interface, and launches it.

### Functions
scrape_and_store()
The scrape_and_store() function scrapes the Langchain API documentation using the scraper module, splits the scraped data into chunks using the embeddings module, creates embeddings for each chunk using the OpenAI API, and stores the embeddings in the Pinecone index using the pinecone_helper module. It returns the Pinecone index.

### query_langchain(question)
The query_langchain(question) function takes a user's question as input, retrieves the embeddings from the Pinecone index using the pinecone_helper module, and generates a response to the user's question using OpenAI's GPT-3.5 model. If there is an error while retrieving the embeddings from Pinecone, it waits for 5 seconds and tries again. It returns the response to the user's question.

### main()
The main() function defines the input and output components for the Gradio interface using the gradio module, creates the Gradio interface, and launches it.

### Gradio Interface
The Gradio interface is created using the gradio module in the main module. It consists of a textbox where the user can enter their question and a markdown output where the response to the user's question is displayed.

### Conclusion
This technical documentation provides a detailed explanation of the Python code for searching and retrieving Langchain API documentation. The code uses various Python modules such as dotenv, openai, os, time, gradio, tqdm, requests, beautifulsoup4, urllib3, html, and re to scrape the web, split the data into chunks, create embeddings, store the embeddings in Pinecone, and generate responses to user queries using GPT-3.5. The Gradio interface provides an easy-to-use platform for users to search the Langchain API documentation.