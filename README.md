# Vector Store

A small implementation of a Vector Store using Numpy to understand the working of semantic search with support for multiple similarity metrics.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)# Vector Store

A small implementation of a Vector Store using Numpy to understand the working of semantic search with support for multiple similarity metrics.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Similarity Metrics](#similarity-metrics)
- [Saving and Loading the Vector Store](#saving-and-loading-the-vector-store)

## Introduction

This project provides a lightweight vector store for semantic search. It uses the `sentence-transformers` library to embed documents and queries, and `numpy` for efficient similarity calculations. The vector store supports multiple similarity metrics, including Euclidean, Cosine, and Manhattan distances.

## Features

- Embedding documents using `sentence-transformers`.
- Support for multiple similarity metrics.
- Efficient similarity calculations using `numpy`.
- Batch query support.
- Saving and loading the vector store to/from files.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. **Initialize the Vector Store:**

```python
from vectorstore.vectorstore import Vectorstore, SimMetric
from sentence_transformers import SentenceTransformer
from documents import document

docs = document.split('\n')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
store = Vectorstore.from_docs(docs, embedder, similarity_metric=SimMetric.MANHATTAN)

## OR
store = Vectorstore(docs, embedder, similarity_metric=SimMetric.MANHATTAN)
store.build_store()
```

2. **Single Query:**

```python
query = "What happens when someone steps into the circle of birch trees during the solstice?"
results, exectime = store.search(query, k=3)
print("Top results:", results[0])
print(f"Search time: {exectime} ms\n")
```

3. **Multiple Queries:**

```python
queries = ["What does Rachel discover in the library, and who presents it to her?", "What unusual phenomenon is associated with the ancient bell in the village?"]
batch_results, batch_exectime = store.search(queries, k=2)
print("Batch results:")
print("Query 1:", batch_results[0], "\n")
print("Query 2:", batch_results[1])
print(f"Search time: {batch_exectime} ms\n")
```

## Similarity Metrics

The vector store supports the following similarity metrics:

- `EUCLIDEAN`: Euclidean distance.
- `COSINE`: Cosine similarity.
- `MANHATTAN`: Manhattan distance.

You can set the similarity metric when initializing the vector store:

```python
store = Vectorstore.from_docs(docs, embedder, similarity_metric=SimMetric.COSINE)
```

## Saving and Loading the Vector Store

You can save and load the vector store to/from files using the `save_store` and `load_store` methods:

```python
# Save the vector store
store.save_store('vector_store.npz')

# Load the vector store
store.load_store('vector_store.npz')
```