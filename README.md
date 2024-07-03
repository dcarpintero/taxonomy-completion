# An LLM Approach to Semantic Clustering and Topic Modeling of Academic Literature

[Clustering](https://en.wikipedia.org/wiki/Cluster_analysis) stands as a fundamental task in unsupervised learning, where the goal is to group unlabeled data into related categories; whereas [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model) focuses on identifying thematic structures within a collection of documents. These techniques find applications across various domains, enabling tasks such as information retrieval, anomaly detection, trend analysis, and biomedical research.

This article provides an end-to-end guide to building an LLM-based pipeline for automatic categorization of research articles into latent topics using open source. Our playground is a  [dataset of 25,000 research arXiv publications](https://huggingface.co/datasets/dcarpintero/arxiv.cs.CL.embedv3.clustering.medium) from Computational Linguistics (Natural Language Processing) published before May 2024.

At its core, the clustering problem relies on finding similar examples. This is a natural task for embeddings, as they capture the semantic relationships in a corpus, and can be provided as input features to a clustering algorithm to establish similarity links among the examples. We begin by transforming the `title:abstract` pairs of our dataset into an embeddings representation using  [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923), a BERT-ALiBi based attention model supporting 8192 sequence length, and subsequently applying [HDBSCAN](https://en.wikipedia.org/wiki/HDBSCAN) in a reduced dimensional space. Topic modeling is then performed at cluster level using a random subset of `titles` within each cluster. This latter process combines [LangChain](https://www.langchain.com/) and [Pydantic](https://docs.pydantic.dev/) with [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) to define a topic pipeline that generates structured `JSON` output.

To measure the clustering and topic modeling effectiveness, we visualize the outcomes after applying [UMAP](https://en.wikipedia.org/wiki/Uniform_Manifold_Approximation_and_Projection) dimensionality reduction.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/X-qkQEg1sgR1p3RecWb1L.png">
  <figcaption style="text-align: center;">Semantic Clustering and Topic Modeling of Academic Literature with LLMs</figcaption>
</figure>

## 1. Embeddings Transformation

Embeddings are numerical representations of real-world objects like text, images, and audio that encapsulate semantic information of the data they represent. They are used by AI models to understand complex knowledge domains in downstream applications such as clustering, as well as information retrieval and classification tasks.

We implement this step with [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923), an open-source text embedding model capable of accommodating up to 8192 tokens. This provides a sufficiently large sequence length for `title:abstract` pairs and other document sections that might be relevant.

```python
```

To overcome the conventional 512-token limit, Jina-Embeddings-v2 incorporates bidirectional [ALiBi](https://arxiv.org/abs/2108.12409) into the BERT framework. AliBi (Attention with Linear Biases) enables input length extrapolation (i.e. sequences exceeding 2048 tokens) by encoding positional information directly within the
self-attention layer, instead of introducing positional embeddings. In practice, it biases query-key attention scores with a penalty that is proportional to their distance, ensuring that proximate tokens demonstrate stronger mutual attention.

The semantic similarity between corpora can be trivially computed as the inner product of the embeddings. In the following heat map each entry [x, y] is colored based on said embeddings product for sentences [x] and [y].

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/4djmELIe2LkZ8_Tofc91Q.png">
  <figcaption style="text-align: center;">Semantic Similary in arXiv 'titles' w/ Jina-Embeddings-v2</figcaption>
</figure>

## 2. Projection for Dimensionality Reduction

We then project our (`title:abstract`) embeddings pairs from a high-dimensional space to a lower-dimensional one using 
[dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction). This process will reduce the computational complexity and memory usage during clustering. In this regard, [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection) is a popular technique known for its effectiveness in preserving both the local and global data structures. This makes it a preferred choice for handling complex datasets with high-dimensional embeddings.

```python
import umap

umap_reducer = umap.UMAP(n_neighbors=100,
                         n_components=5,
                         min_dist=0.1,
                         metric='cosine')
umap_embedding = umap_reducer.fit_transform(ds['embeddings'])
```

In our implementation, we configure UMAP with:
- `n_neighbors=100` to consider 100 nearest neighbors for each point (arXiv publication);
- `n_components=5` to reduce the embeddings to 5 dimensions;
- `min_dist=0.1` to maintain a balance between the local and global structure; and,
- `metric='cosine'` to measure the distance between points using the cosine similarity metric.

Note that when we apply HDBSCAN clustering in the next step, the clusters found will be influenced by how UMAP preserved the local structures. A smaller `n_neighbors` value means UMAP will focus more on local structures, whereas a larger value allows to capture more global representations, which might be beneficial for understanding overall patterns in the data.

## 3. Semantic Clustering

This section shows how to use the reduced (`title:abstract`) embeddings as input features of a clustering algorithm. This allows for the identification of related categories based on the distance between the provided embeddings.

We have opted for [HDBSCAN](https://en.wikipedia.org/wiki/HDBSCAN) (Hierarchical Density-Based Spatial Clustering of Applications with Noise), an advanced clustering algorithm that extends DBSCAN by adapting to varying density clusters. Unlike K-Means which requires pre-specifying the number of clusters, HDBSCAN has only one important hyperparameter, `n`, which establishes the minimum number of examples to include in a cluster. As a density-based method, it can also detect outliers in the data.

HDBSCAN works by first transforming the data space according to the density of the data points, making denser regions (areas where data points are close together in high numbers) more attractive for cluster formation. The algorithm then builds a hierarchy of clusters based on the minimum cluster size established by the hyperparameter `n`. This allows it to distinguish between noise (sparse areas) and dense regions (potential clusters). Finally, HDBSCAN condenses this hierarchy to derive the most persistent clusters, efficiently identifying clusters of different densities and shapes.

```python
import hdbscan

hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=100,
                                metric='euclidean',
                                cluster_selection_method='eom')
clusters = hdbscan_model.fit_predict(umap_embedding)
```

While we define a minimum cluster size similar to the number of neighbors in UMAP, in practice they do not need to be equal.

## 3. Topic Modeling

Having performed the clustering step, we now illustrate how to identify the topic of each cluster by combining an LLM such as [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) with [Pydantic](https://docs.pydantic.dev/) and [LangChain](https://www.langchain.com/) to create a topic modeling pipeline.

### 3.1 Pydantic Model

[Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/) are classes that derive from `pydantic.BaseModel`, defining fields as type-annotated attributes. They bear a strong resemblance to `Python` dataclasses. However, they have been designed with subtle but significant differences that optimize various operations such as validation, serialization, and `JSON` schema generation. Our `Topic` class defines a field named `category`. This will generate output in a structured format, rather than a free-form text block, facilitating easier processing and analysis of the topic modeling results.

```python
class Topic(BaseModel):
    """
    Pydantic Model to generate an structured Topic Model
    """
    category: str = Field(..., description="Identified topic")
```

### 3.2 LangChain Prompt Template

[LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/) are pre-defined recipes for generating prompts for language models.

```python
topic_prompt = """
    You are a helpful assistant. Your task is to analyze a set of research paper titles related to
    Natural Language Processing and determine the overarching topic. Based on the titles provided,
    identify the most relevant topic. The response should be a concise short sentence describing
    the identified topic in JSON format. No additional information or follow-up questions are needed.

    TITLES:
    {titles}

    EXPECTED OUTPUT:
    {{"category": "Topic Name"}}
    """
```

### 3.3 Inference of Topic Identification

This section illustrates how to compose a topic pipeline using the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/).

```python
```

To enable the model to infer the topic of each cluster, we provide a random subset of 15 paper titles from each cluster as input.

```python
```

## 4. Visualization

We prepare the dataset for visualization by reducing the number of dimensions, in this case to '2'.

```python
umap_reducer = umap.UMAP(n_neighbors=100,
                         n_components=2,
                         min_dist=0.1,
                         metric='cosine')
umap_embedding = umap_reducer.fit_transform(ds['embeddings'])
```