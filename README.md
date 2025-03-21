## Installation Instructions

To set up the project with `faiss` you'll need `conda` as it is not available on poetry. The rest of the dependencies can be installed using `poetry`:

1. **Create a `conda` environment:**

Open your terminal and run the following command to create a new `conda` environment:

```sh
conda create -n chromadb-project python=3.12
```

2. **Activate the conda environment**

```sh
conda activate chromadb-project
```

3. **Install faiss using conda**

```sh
conda install -c pytorch faiss-cpu
```

4. **Install the rest of the dependencies using poetry**

```sh
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

## Running the LLM solo, without a vectorDB or inMemory vector clustering

```sh
poetry run python llm_raw_server.py
```

Choose between asking a question (with manual input and optional context to be provided) or reading a directory (serving as context and manual input). As a warning, last option won't ever finish if the directory is more than a couple files long, given we are chunking the Input into valid lengths and running them through the model.

## Running the LLM alognside a vectorDB (for data persistance) and inMemory vector clustering (for faster search and clustering of vectors)

```sh
poetry run python src/main.py
```

## Note

For this project we are using the lightweight variation of the DeepSeek-Coder LLM. It should be fine for an Apple Silicon MacBook.

This is an open-source model that excels in use cases like:

- Answering specific questions about code (e.g., “What does this function do?”).
- Explaining complex code snippets in detail.
- Following instructions like "Refactor this code to improve performance."
- Assisting with learning programming concepts, providing explanations and guidance.

Also, the -instruct variation is specifically optimized for tasks where clear instructions are given, like explaining code, answering questions about code behavior, or giving detailed responses in a conversational manner. This model is more adept at interpreting human instructions and providing more contextually accurate responses.

https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct

When running the `llm_server.py` or `src/main.py` for the 1st time, the LLM will be automatically added to your ~/.cache/huggingface/hub folder.
