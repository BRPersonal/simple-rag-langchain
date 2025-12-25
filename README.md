# krishnaik-simple-rag-langchain
Basic RAG using langchain

Setup the project
$ uv init --python 3.12.9
$ uv add --requirements requirements.txt

verify the python version
$ cat .python-version 
$ cat pyproject.toml
$ ls -alt # you should see uv.lock

Once pyproject.toml is created, you can delete requirements.txt
If you have pyproject.toml file , run
$ uv sync
It will create the virtual environment, install all the packages and create uv.lock file
You are all set


