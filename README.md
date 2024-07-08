# An LLM-based Pipeline for Categorization and Taxonomy Completion of Academic Literature (with Embeddings Quantization)

[![GitHub license](https://img.shields.io/github/license/dcarpintero/semantic-clustering)](https://github.com/dcarpintero/semantic-clustering/blob/main/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/dcarpintero/semantic-clustering.svg)](https://GitHub.com/dcarpintero/semantic-clustering/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/dcarpintero/semantic-clustering.svg)](https://GitHub.com/dcarpintero/semantic-clustering/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/dcarpintero/semantic-clustering.svg)](https://GitHub.com/dcarpintero/semantic-clustering/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dcarpintero/semantic-clustering/blob/main/nb.semantic.clustering.topic.modeling.ipynb)

[![GitHub watchers](https://img.shields.io/github/watchers/dcarpintero/semantic-clustering.svg?style=social&label=Watch)](https://GitHub.com/dcarpintero/semantic-clustering/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/dcarpintero/semantic-clustering.svg?style=social&label=Fork)](https://GitHub.com/dcarpintero/semantic-clustering/network/)
[![GitHub stars](https://img.shields.io/github/stars/dcarpintero/semantic-clustering.svg?style=social&label=Star)](https://GitHub.com/dcarpintero/semantic-clustering/stargazers/)

The ever-growing volume of research publications necessitates efficient methods for categorizing academic literature and completing taxonomies. This article provides an end-to-end automated solution by building a Large Language Model (LLM)-based pipeline and using embedding quantization. Our playground is a dataset of [25,000 arXiv publications](https://huggingface.co/datasets/dcarpintero/arxiv.cs.CL.25k) from Computational Linguistics (Natural Language Processing) published before July 2024.

The LLM-based pipeline implements *semantic clustering* and *topic modeling*, utilizing open-source tools and LLMs. [Clustering](https://en.wikipedia.org/wiki/Cluster_analysis) stands as a foundational task in unsupervised learning, where the goal is to group unlabeled data into related categories; whereas [topic modeling](https://en.wikipedia.org/wiki/Topic_model) identifies thematic structures within such collections of documents.

At its core, the clustering problem relies on finding similar examples. This is a natural task for embeddings, as they capture the semantic relationships in a corpus, and can be provided as input features to a clustering algorithm to establish similarity links among the examples. We begin by transforming the (`title:abstract`) pairs of our [arXiv dataset](https://huggingface.co/datasets/dcarpintero/arxiv.cs.CL.25k) into a (quantized) embeddings representation using  [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923) - a BERT-ALiBi based attention model supporting 8192 sequence lengths - and subsequently applying HDBSCAN in a reduced dimensional space. Topic modeling is then performed at the cluster level. This latter process combines [LangChain](https://www.langchain.com/) and [Pydantic](https://docs.pydantic.dev/) with [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) to generate structured output.

The results hint at emerging research domains around Language Models (LLMs) in the field of Computational Linguistics (cs.CL in arXiv) such as: '*Vision-Language-Models*', '*Multilingual LLMs*', '*Bias-, Attacks-, and Hallucination in LLMs*', '*LLM-based Agents*', '*Model Alignment*', '*Model Compression and Acceleration*', '*Misinformation Detection*' and '*Mathematical Reasoning in LLMs*', among others. This approach might serve as a baseline for automatically identifying candidate (sub)topics within high-level [arXiv categories](https://arxiv.org/category_taxonomy) and efficiently completing taxonomies, addressing the challenge posed by the increasing volume of publications.

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/iE3e4VJSY84JyyTR9krmf.png">
  <!-- figcaption style="text-align: center;">LLM-based Pipeline for Categorization and Taxonomy Completion of Academic Literature</figcaption -->
</p>

<p align="center">LLM-based Pipeline for Categorization and Taxonomy Completion of Academic Literature</p>

## 1. Embedding Transformation with Quantization

Embeddings are numerical representations of real-world objects like text, images, and audio that encapsulate semantic information of the data they represent. They are used by AI models to understand complex knowledge domains in downstream applications such as clustering, as well as information retrieval and classification tasks, among others.

### Large Sequence Jina-Embeddings

We will map (`title:abstract`) pairs to a 768 dimensional space using [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923) [1], an open-source text embedding model capable of accommodating up to 8192 tokens. This provides a sufficiently large sequence length for titles, abstracts, and other document sections that might be relevant. To overcome the conventional 512-token limit present in other models, Jina-Embeddings incorporates bidirectional [ALiBi](https://arxiv.org/abs/2108.12409) [2] into the BERT framework. AliBi (Attention with Linear Biases) enables input length extrapolation (i.e. sequences exceeding 2048 tokens) by encoding positional information directly within the self-attention layer, instead of introducing positional embeddings. In practice, it biases query-key attention scores with a penalty that is proportional to their distance, ensuring that proximate tokens demonstrate stronger mutual attention.

### Encoding with Sentence Transformers

The first step to use the [Jina-Embeddings-v2](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) model is to load it through [Sentence Transformers](https://www.SBERT.net), a framework for accessing state-of-the-art models that is available at the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers&sort=downloads). There you can find over 500 hundred models such as [`all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) and [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) that are also suitable for text encoding.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
```

We now encode (`title:abstract`) pairs of our dataset using `batch_size = 64`. This allows for parallel computation on hardware accelerators like GPUs (at the cost of requiring more memory). 

```python
import pandas as pd

DS_PATH='./arxiv.cs.CL.25k.desc.json'
BATCH_SIZE = 64

df = pd.read_json(DS_PATH)
embeddings = model.encode(df['title'] + ':' + df['abstract'],
                          batch_size=BATCH_SIZE,
                          show_progress_bar=True)
df['embeddings'] = embeddings.tolist()
```

### Computing Semantic Similarity

The semantic similarity between corpora can now be trivially computed as the inner product of embeddings. In the following heat map each entry [x, y] is colored based on said embeddings product for `title` sentences [x] and [y].

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/4djmELIe2LkZ8_Tofc91Q.png">
  <!-- figcaption style="text-align: center;">Semantic Similary in arXiv 'titles' with Embeddings</figcaption -->
</p>

<p align="center">Semantic Similary in arXiv 'titles' with Embeddings</p>

### Reducing Memory Requirements with Quantization

Embeddings may be challenging to scale up. Currently, state-of-the-art models represent each embedding as `float32` which requires 4 bytes of memory. Given that Jina-Embeddings-v2 maps text to a 768 dimensional space, the memory requirements for our dataset would be around 73 MB without index and other metadata.

```python
25,000 embeddings * 768 dimensions/embedding * 4 bytes/dimension = 76,800,000 bytes
76,800,000 bytes / (1024^2) ≈ 73.24 MB
```

However, if you work with a larger dataset, the memory requirements and associated costs might increase significantly:

| Embedding<br>Dimension | Embedding<br>Model            | 2.5M<br>ArXiv Abstracts      | [60.9M<br>Wikipedia Pages](https://en.wikipedia.org/wiki/Wikipedia:Size_of_Wikipedia) | 100M<br>Embeddings |
|------------------------|-------------------------------|------------------------------|-----------------------|------------------------------|
| 384                    | all-MiniLM-L12-v2             | 3.57 GB                      | 85.26 GB              | 142.88 GB                    |
| 768                    | all-mpnet-base-v2             | 7.15 GB                      | 170.52 GB             | 285.76 GB                    |
| 768                    | jina-embeddings-v2            | 7.15 GB                      | 170.52 GB             | 285.76 GB                    |
| 1536                   | openai-text-embedding-3-small | 14.31 GB                     | 341.04 GB             | 571.53 GB                    |
| 3072                   | openai-text-embedding-3-large | 28.61 GB                     | 682.08 GB             | 1.143 TB                   |

A technique used to achieve memory saving is *Quantization*. The intuition behind this approach is that we can discretize  floating-point values by mapping their range [f_max, f_min] into a smaller range of fixed-point numbers [q_max, q_min], and linearly distributing all values in between. In practice, this typically reduces the precision of a 32-bit floating-point to lower bit widths like 8-bits (scalar-quantization) or 1-bit values (binary quantization).

We will compute and compare the [results](https://huggingface.co/blog/dcarpintero/llm-based-categorization-and-taxonomy-completion/#5-visualization-and-results) using both `float32` and `int8` embeddings. This results in 4x memory saving and faster arithmetic operations (matrix multiplication can be performed faster with integer arithmetic). 

```python
from sentence_transformers.quantization import quantize_embeddings
# quantization is applied in a post-processing step
int8_embeddings = quantize_embeddings(embeddings, precision="int8")
```

We can see the differences between `float32` and `int8` embeddings.

```python
>>> fp32_embeddings.dtype, fp32_embeddings.shape, fp32_embeddings.nbytes

>>> int8_embeddings.dtype, int8_embeddings.shape, int8_embeddings.nbytes
```

## 2. Projecting Embeddings for Dimensionality Reduction

In order to reduce the computational complexity and memory usage during clustering, we project the (`title:abstract`) embedding pairs from a high-dimensional space (768) to a lower-dimensional one (5 dimensions) using [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction). In this regard, [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection) [3] is a popular technique known for its effectiveness in preserving both the local and global data structures.

```python
import umap

umap_reducer = umap.UMAP(n_neighbors=100, # consider 100 nearest neighbors for each point
                         n_components=5,  # reduce the embeddings from 768 to 5 dimensions
                         min_dist=0.1,    # maintain balance between local and global structures
                         metric='cosine') # measure points distance using cosine similarity

umap_embedding = umap_reducer.fit_transform(ds['embeddings'])
```

Note that when we apply HDBSCAN clustering in the next step, the clusters found will be influenced by how UMAP preserved the local structures. A smaller `n_neighbors` value means UMAP will focus more on local structures, whereas a larger value allows to capture more global representations, which might be beneficial for understanding overall patterns in the data.

## 3. Semantic Clustering

This section shows how to use the reduced (`title:abstract`) embeddings as input features of a clustering algorithm. This allows for the identification of related categories based on the distance between said embeddings.

We have opted for [HDBSCAN](https://en.wikipedia.org/wiki/HDBSCAN) (Hierarchical Density-Based Spatial Clustering of Applications with Noise) [4], an advanced clustering algorithm that extends DBSCAN by adapting to varying density clusters. Unlike K-Means which requires pre-specifying the number of clusters, HDBSCAN has only one important hyperparameter, `n`, which establishes the minimum number of examples to include in a cluster. As a density-based method, it can also detect outliers in the data.

HDBSCAN works by first transforming the data space according to the density of the data points, making denser regions (areas where data points are close together in high numbers) more attractive for cluster formation. The algorithm then builds a hierarchy of clusters based on the minimum cluster size established by the hyperparameter `n`. This allows it to distinguish between noise (sparse areas) and dense regions (potential clusters). Finally, HDBSCAN condenses this hierarchy to derive the most persistent clusters, efficiently identifying clusters of different densities and shapes.

```python
import hdbscan

hdbs = hdbscan.HDBSCAN(min_cluster_size=100,            # minimum size of clusters to be formed
                       metric='euclidean',              # measure points distance using Euclidean
                       cluster_selection_method='eom')  # cluster selection w/ Excess of Mass (EOM)

clusters = hdbs.fit_predict(umap_embedding)             # apply HDBSCAN on reduced UMAP embeddings
```

Note that while we define a minimum cluster size similar to the number of neighbors in UMAP, in practice they do not need to be equal.

## 4. Topic Modeling

Having performed the clustering step, we now illustrate how to identify the latent topic of each cluster by combining an LLM such as [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) [5] with [Pydantic](https://docs.pydantic.dev/) and [LangChain](https://www.langchain.com/) to create a topic modeling pipeline.

### 4.1 Pydantic Model

[Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/) are classes that derive from `pydantic.BaseModel`, defining fields as type-annotated attributes. They bear a strong resemblance to `Python` dataclasses. However, they have been designed with subtle but significant differences that optimize various operations such as validation, serialization, and `JSON` schema generation. Our `Topic` class defines a field named `label`. This will generate output in a structured format, rather than a free-form text block, facilitating easier processing and analysis of the topic modeling results.

```python
from pydantic import BaseModel, Field

class Topic(BaseModel):
    """
    Pydantic Model to generate an structured Topic Model
    """
    label: str = Field(..., description="Identified topic")
```

### 4.2 LangChain Prompt Template

[LangChain Prompt Templates](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) are pre-defined recipes for translating user input and parameters into instructions for a language model.

```python
from langchain_core.prompts import PromptTemplate

topic_prompt = """
    You are a helpful assistant. Your task is to analyze a set of research paper titles related to
    Natural Language Processing and determine the overarching topic. Based on the titles provided,
    identify the most relevant topic. The response should be concise, clearly stating the single
    identified topic. Format your response in JSON as in the 'EXPECTED OUTPUT' section below.
    No additional information or follow-up questions are needed.

    EXPECTED OUTPUT:
    {{"label": "Topic Name"}}

    TITLES:
    {titles}
    """
```

### 4.3 Inference of Topic Identification

This section illustrates how to compose a topic pipeline using the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/).

```python
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser

from typing import List

def TopicModeling(titles: List[str]) -> str:
    """
    Infer the common topic of the given titles w/ LangChain, Pydantic, OpenAI
    """
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.2,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    prompt = PromptTemplate.from_template(topic_prompt)
    parser = PydanticOutputParser(pydantic_object=Topic)

    topic_chain = prompt | llm | parser
    return topic_chain.invoke({"titles": titles})
```

To enable the model to infer the topic of each cluster, we provide a random subset of '25' arXiv titles from each cluster as input.

```python
topics = []
for i, cluster in df.groupby('cluster'):
    titles = cluster['title'].head(25).tolist()
    topic = TopicModeling(titles)
    topics.append(topic.label)
```

In the next step, we map the latent topics identified by the LLM pipeline to the corresponding clusters.

```python
n_clusters = len(df['cluster'].unique())

topic_map = dict(zip(range(n_clusters), topics))
df['topic'] = df['cluster'].map(topic_map)
```

## 5. Visualization and Results

We prepare the dataset for visualization by further reducing the number of dimensions, in this case to '2' dimensions:

```python
x_y_embeddings = umap.UMAP(n_neighbors=100, # consider 100 nearest neighbors for each point
                           n_components=2,  # reduce embeddings space from 5 to 2 dimensions
                           min_dist=0.1,    # maintain balance between local and global structures
                           metric='cosine').fit_transform(ds['embeddings'])

df = pd.DataFrame(x_y_embeddings, columns=['x', 'y'])
df['cluster'] = clusters
df['title'] = ds['title']
df = df[df['cluster'] != -1] # remove outliers
```

And then we create an interactive scatter plot:

```python
chart = alt.Chart(df).mark_circle(size=5).encode(
    x='x',
    y='y',
    color='topic:N',
    tooltip=['title', 'topic']
).interactive().properties(
    title='Semantic Clustering and Topic Modeling of Academic Literature (25k arXiv publications)',
    width=600,
    height=400,
)
chart.display()
```

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/KUb54rlWen7Xf64RXzkUC.png">
  <!--figcaption style="text-align: center;">Semantic Clustering and Topic Modeling of Academic Literature (25k arXiv publications)</figcaption-->
</p>

<p align="center">
  Semantic Clustering and Topic Modeling of Academic Literature (25k arXiv publications)
</p>

## Resources

- [GitHub Repo](https://github.com/dcarpintero/llm-based-categorization-taxonomy-completion)
- [LangChain](https://python.langchain.com/v0.2/docs/introduction/) | [LangChain Prompt Templates](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) | [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [Sentence Transformers](https://www.sbert.net/index.html) | [Sentence Transformers Quantization](https://sbert.net/examples/applications/embedding-quantization/README.html)

## Citation

```
@article{carpintero2024categorization
  author = { Diego Carpintero},
  title = {An LLM-based Pipeline for Categorization and Taxonomy Completion of Academic Literature},
  journal = {Hugging Face Blog},
  year = {2024},
  note = {https://huggingface.co/blog/dcarpintero/llm-based-categorization-and-taxonomy-completion},
}
```

## References

- [1] Günther, et al. 2024. *Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents*. [arXiv:2310.19923](https://arxiv.org/abs/2310.19923).
- [2] Press, . et al. 2021. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409).
- [3] McInnes, et al. 2018. *Umap: Uniform manifold approximation and projection for dimension reduction*. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426).
- [4] Campello, et al. 2013. *Density-Based Clustering Based on Hierarchical Density Estimates. Advances in Knowledge Discovery and Data Mining*. Vol. 7819. Berlin, Heidelberg: Springer Berlin Heidelberg. pp. 160–172. [doi:10.1007/978-3-642-37456-2_14](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14).
- [5] Jiang, et al. 2023. *Mistral 7B*. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825).
- [6] Shakir, et al. 2024. *Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval*. [hf:shakir-embedding-quantization](https://huggingface.co/blog/embedding-quantization)