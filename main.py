from vectorstore.vectorstore import Vectorstore, SimMetric
from sentence_transformers import SentenceTransformer
from documents import document

docs = document.split('\n')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
store = Vectorstore.from_docs(docs, embedder, similarity_metric=SimMetric.COSINE)
# store.build_store()

# Single query
query = "What happens when someone steps into the circle of birch trees during the solstice?"
results, exectime = store.search(query, k=3)
print("Top results:", results[0])
print(f"Search time: {exectime} ms\n")

# Multiple queries
queries = ["What does Rachel discover in the library, and who presents it to her?", "What unusual phenomenon is associated with the ancient bell in the village?"]
batch_results, batch_exectime = store.search(queries, k=2)
print("Batch results:")
print("Query 1:", batch_results[0], "\n")
print("Query 2:", batch_results[1])
print(f"Search time: {batch_exectime} ms\n")