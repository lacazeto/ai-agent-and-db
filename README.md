## Installation Instructions

To set up the project with `faiss` you'll need `conda` as it is not available on poetry. The rest of the dependencies can be done using `poetry`:

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

For this project we are using the lightweight variation of the DeepSeek-Coder LLM. It should be fine for an Apple Silicon MacBook.

Also, the -instruct variation is specifically optimized for tasks where clear instructions are given, like explaining code, answering questions about code behavior, or giving detailed responses in a conversational manner. This model is more adept at interpreting human instructions and providing more contextually accurate responses.

https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct

When running the llm_server.py for the 1st time, the LLM will be automatically added to your ~/.cache/huggingface/hub folder.
