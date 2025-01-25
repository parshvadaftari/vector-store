import time
from enum import Enum
from functools import wraps
from typing import *

import numpy as np
from sentence_transformers import SentenceTransformer

class SimMetric(Enum):
    """
    Enum for different similarity metrics.
    """
    EUCLIDEAN = 0
    COSINE = 1
    MANHATTAN = 2

def timeit(func):
    """
    Decorator to measure the execution time of a function.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: The wrapped function that returns the result and execution time.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        t_exec = time.time() - t_start
        return res, t_exec * 1000
    return inner

class Vectorstore:
    """
    A lightweight vector store for semantic search with support for multiple similarity metrics.
    """

    def __init__(
        self,
        docs: List[str],
        embedder: SentenceTransformer = None,
        similarity_metric: SimMetric = SimMetric.EUCLIDEAN
    ) -> None:
        """
        Initialize the Vectorstore.

        Args:
            docs (List[str]): List of documents to be stored.
            embedder (SentenceTransformer, optional): The sentence transformer model to use for embedding.
            similarity_metric (SimMetric, optional): The similarity metric to use. Defaults to SimMetric.EUCLIDEAN.
        """
        self.docs = np.array(docs)
        self.embedder = embedder
        self.similarity_metric = similarity_metric
        self._store: Optional[np.ndarray] = None
        self._set_sim_func()

    def set_metric(self, metric: SimMetric):
        """
        Set the similarity metric for the vector store.

        Args:
            metric (SimMetric): The similarity metric to set.
        """
        assert isinstance(metric, SimMetric), "Invalid similarity metric type."
        self.similarity_metric = metric
        self._set_sim_func()

    def _set_sim_func(self):
        """
        Initialize the similarity function based on the selected metric.
        """
        if self.similarity_metric == SimMetric.EUCLIDEAN:
            self._sim_func = self._dist_euclidean__
        elif self.similarity_metric == SimMetric.COSINE:
            self._sim_func = self._cosine__
        elif self.similarity_metric == SimMetric.MANHATTAN:
            self._sim_func = self._dist_manhattan__
        else:
            raise NotImplementedError(f"Similarity function for {self.similarity_metric} is not implemented.")

    @classmethod
    def from_docs(
        cls,
        docs: List[str],
        embedder: SentenceTransformer = None,
        similarity_metric=SimMetric.EUCLIDEAN
    ) -> "Vectorstore":
        """
        Create and build a vector store from a list of documents.

        Args:
            docs (List[str]): List of documents to be stored.
            embedder (SentenceTransformer, optional): The sentence transformer model to use for embedding.
            similarity_metric (SimMetric, optional): The similarity metric to use. Defaults to SimMetric.EUCLIDEAN.

        Returns:
            Vectorstore: The initialized and built vector store.
        """
        store = cls(docs, embedder=embedder, similarity_metric=similarity_metric)
        print(f"[ LOG ] Using similarity metric: {similarity_metric}\n")
        return store.build_store()

    def build_store(self):
        """
        Embed the documents and build the vector store.

        Returns:
            Vectorstore: The vector store with embedded documents.
        """
        if self.embedder is not None:
            self._store = self.embedder.encode(self.docs)
        return self

    @timeit
    def search(self, query: Union[str, List[str]], k: int = 5) -> tuple:
        """
        Get top K similar documents and their scores for a query or a list of queries.
        The lower score, the better.

        Args:
            query (Union[str, List[str]]): The query or list of queries.
            k (int, optional): The number of top similar documents to return. Defaults to 5.

        Returns:
            tuple: A tuple containing the results and execution time.
        """
        assert self.embedder is not None
        assert k >= 1

        # Handle batch queries
        if isinstance(query, list):
            q_emb = self.embedder.encode(query)
            assert q_emb.ndim == 2, "Query embeddings for batch queries must be 2-dimensional."

            results = []
            for i, single_query_emb in enumerate(q_emb):
                top_docs = self._get_topk_similar(single_query_emb, k=k)
                results.append((query[i], top_docs))
            return results
        else:
            # Handle single query
            q_emb = self.embedder.encode(query)
            assert q_emb.ndim == 1, "Query embedding must be 1-dimensional."
            return self._get_topk_similar(q_emb, k=k)

    def save_store(self, filepath: str):
        """
        Save the vector store to a file.

        Args:
            filepath (str): The file path to save the vector store.
        """
        assert self._store is not None, "Vector store is empty. Build it before saving."
        np.savez(filepath, docs=self.docs, store=self._store)

    def load_store(self, filepath: str):
        """
        Load the vector store from a file.

        Args:
            filepath (str): The file path to load the vector store from.

        Returns:
            Vectorstore: The vector store with loaded data.
        """
        data = np.load(filepath, allow_pickle=True)
        self.docs = data['docs']
        self._store = data['store']
        return self

    def _dist_euclidean__(self, query: np.ndarray):
        """
        Calculate Euclidean distance between all vectors in the store and the query.

        Args:
            query (np.ndarray): The query vector.

        Returns:
            np.ndarray: The Euclidean distances.
        """
        assert query.ndim == 1
        assert query.shape[0] == self._store.shape[1], "Shape mismatch between query and store."
        dist: np.ndarray = np.sqrt(np.sum((self._store - query) ** 2, axis=1))
        return dist

    def _cosine__(self, query: np.ndarray):
        """
        Calculate Cosine similarity between all vectors in the store and the query.

        Args:
            query (np.ndarray): The query vector.

        Returns:
            np.ndarray: The Cosine similarities.
        """
        assert query.ndim == 1
        assert query.shape[0] == self._store.shape[1], "Shape mismatch between query and store."
        norm_store = np.linalg.norm(self._store, axis=1)
        norm_query = np.linalg.norm(query)
        similarity = np.dot(self._store, query) / (norm_store * norm_query + 1e-10)
        return similarity

    def _dist_manhattan__(self, query: np.ndarray):
        """
        Calculate Manhattan distance between all vectors in the store and the query.

        Args:
            query (np.ndarray): The query vector.

        Returns:
            np.ndarray: The Manhattan distances.
        """
        assert query.ndim == 1
        assert query.shape[0] == self._store.shape[1], "Shape mismatch between query and store."
        dist: np.ndarray = np.sum(np.abs(self._store - query), axis=1)
        return dist

    def _get_topk_similar(self, query: np.ndarray, k: int = 5):
        """
        Get the top K similar documents and their scores based on the similarity metric.

        Args:
            query (np.ndarray): The query vector.
            k (int, optional): The number of top similar documents to return. Defaults to 5.

        Returns:
            List[str]: The top K similar documents.
        """
        reverse = self.similarity_metric == SimMetric.COSINE
        arr = self._sim_func(query)
        sorted_indices = np.argsort(arr)
        top_k_indices = sorted_indices[::-1][:k] if reverse else sorted_indices[:k]

        topk_docs = self.docs[top_k_indices]
        topk_dist = arr[top_k_indices]

        return list(topk_docs)

    def __repr__(self) -> str:
        """
        Return a string representation of the Vectorstore.

        Returns:
            str: The string representation of the Vectorstore.
        """
        return f"Vectorstore(embedder={self.embedder}, metric={self.similarity_metric})"