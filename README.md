# RAG Pipeline for Webpage Q&A

This project implements a Retrieval Augmented Generation (RAG) pipeline to answer questions based on content from specific webpages.

## Components

- **Data Ingestion**: Uses WebBaseLoader to fetch content from https://www.educosys.com/
- **Text Processing**: Splits text into chunks using RecursiveCharacterTextSplitter
- **Vectorization & Storage**: Converts chunks to embeddings with HuggingFaceEmbeddings and stores in Chroma DB
- **Language Model**: Uses HuggingFaceEndpoint with google/flan-t5-base
- **RAG Chain**: Orchestrates retrieval and generation using LangChain

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get a Hugging Face API token from https://huggingface.co/settings/tokens

3. Add your token to `.env`:
   ```
   HUGGINGFACE_API_TOKEN=your_token_here
   ```

4. Run the pipeline:
   ```
   python rag_pipeline.py
   ```

## Troubleshooting

If you encounter StopIteration errors, try switching to a different model in the code, such as 'microsoft/DialoGPT-medium' or another generative model.