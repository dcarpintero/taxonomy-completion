# Taxonomy Completion with Embedding Quantization and an LLM-based Pipeline: A Case Study in Computational Linguistics

The ever-growing volume of research publications demands efficient methods for structuring academic knowledge. This task typically involves developing a supervised underlying scheme of classes and allocating publications to the most relevant class. In this article, we implement an end-to-end automated solution leveraging open-source embedding quantization and a Large Language Model (LLM) pipeline. Our playground is a dataset of [25,000 arXiv publications](https://huggingface.co/datasets/dcarpintero/arxiv.cs.CL.25k) from Computational Linguistics (cs.CL), published before July 2024, which we aim to organize under a novel candidate scheme of cs.CL sub-classes. 

## Methodology

Our approach centers on two key tasks: (i) unsupervised clustering of the arXiv dataset into related collections, and (ii) discovering the latent thematic structures within each cluster.

At its core, the clustering task requires identifying a sufficient number of similar examples within our *unlabeled* dataset.
This is a natural task for embeddings, as they capture semantic relationships in a corpus and can be provided as input features to a clustering algorithm to establish similarity links among examples. We begin by transforming the (*title*:*abstract*) pairs of the dataset into an embeddings representation using [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923), a BERT-ALiBi based attention model supporting 8192 sequence lengths. We then apply scalar quantization with [Sentence Transformers](https://www.sbert.net/). And run [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) in a reduced dimensional space to perform the clustering.

To discover the latent topics within each cluster of arXiv publications, we combine [LangChain](https://www.langchain.com/) and [Pydantic](https://docs.pydantic.dev/) with [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) into an LLM-pipeline that provides structured output.

The results hint at emerging research domains around Language Models (LLMs) in the field of Computational Linguistics (cs.CL), such as: '*Vision-Language-Models*', '*Multilingual LLMs*', '*Bias-, Attacks-, and Hallucination in LLMs*', '*LLM-based Agents*', '*Model Alignment*', '*Model Compression and Acceleration*', '*Misinformation Detection*' and '*Mathematical Reasoning in LLMs*', among others. This approach might serve as a baseline for automatically identifying candidate (sub)classes within high-level [arXiv categories](https://arxiv.org/category_taxonomy) and efficiently completing taxonomies, addressing the challenge posed by the increasing volume of academic literature.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/Ghc69tCVsY-RMXD50PeIC.png">
  <figcaption style="text-align: center;">Taxonomy Completion of Academic Literature with Embedding Quantization and an LLM-Pipeline</figcaption>
</figure>

## 1. Embedding Transformation

Embeddings are numerical representations of real-world objects like text, images, and audio that encapsulate semantic information of the data they represent. They are used by AI models to understand complex knowledge domains in downstream applications such as clustering, as well as information retrieval and classification tasks, among others.

#### Supporting Large Sequences

We will map (*title*:*abstract*) pairs from arXiv publications to a 768 dimensional space using [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923) [1], an open-source text embedding model capable of accommodating up to 8192 tokens. This provides a sufficiently large sequence length for titles, abstracts, and other document sections that might be relevant. To overcome the conventional 512-token limit  present in other models, Jina-Embeddings-v2 incorporates bidirectional [ALiBi](https://arxiv.org/abs/2108.12409) [2] into the BERT framework. ALiBi (Attention with Linear Biases) enables input length extrapolation (i.e. sequences exceeding 2048 tokens) by encoding positional information directly within the self-attention layer, instead of introducing positional embeddings. In practice, it biases query-key attention scores with a penalty that is proportional to their distance, favoring stronger mutual attention between proximate tokens.

#### Encoding with Sentence Transformers

The first step to use the [Jina-Embeddings-v2](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) model is to load it through [Sentence Transformers](https://www.SBERT.net), a framework for accessing state-of-the-art models that is available at the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers&sort=downloads). There you can find over 500 hundreds models such as [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) and [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) that are also suitable for text encoding.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
```

We now encode (*title*:*abstract*) pairs of our dataset using `batch_size = 64`. This allows for parallel computation on hardware accelerators like GPUs (albeit at the cost of requiring more memory). 

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

#### Computing Semantic Similarity

The semantic similarity between corpora can now be trivially computed as the inner product of embeddings. In the following heat map each entry [x, y] is colored based on said embeddings product for '*title*' sentences [x] and [y].

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/4djmELIe2LkZ8_Tofc91Q.png">
  <figcaption style="text-align: center;">Semantic Similary in <em>cs.CL arXiv-titles</em> using Embeddings</figcaption>
</figure>

## 2. Embedding Quantization for Memory Saving

Scaling up embeddings can be challenging. Currently, state-of-the-art models represent each embedding as `float32`, which requires 4 bytes of memory. Given that Jina-Embeddings-v2 maps text to a 768 dimensional space, the memory requirements for our dataset would be around 73 MB without index and other metadata.

```python
25,000 embeddings * 768 dimensions/embedding * 4 bytes/dimension = 76,800,000 bytes
76,800,000 bytes / (1024^2) ≈ 73.24 MB
```

However, if you work with a larger dataset, the memory requirements and associated costs might increase significantly:

| Embedding<br>Dimension | Embedding<br>Model            | 2.5M<br>ArXiv Abstracts      | [60.9M<br>Wikipedia Docs](https://en.wikipedia.org/wiki/Wikipedia:Size_of_Wikipedia) | 100M<br>Embeddings |
|------------------------|-------------------------------|------------------------------|-----------------------|------------------------------|
| 384                    | all-MiniLM-L12-v2             | 3.57 GB                      | 85.26 GB              | 142.88 GB                    |
| 768                    | all-mpnet-base-v2             | 7.15 GB                      | 170.52 GB             | 285.76 GB                    |
| 768                    | jina-embeddings-v2            | 7.15 GB                      | 170.52 GB             | 285.76 GB                    |
| 1536                   | openai-text-embedding-3-small | 14.31 GB                     | 341.04 GB             | 571.53 GB                    |
| 3072                   | openai-text-embedding-3-large | 28.61 GB                     | 682.08 GB             | 1.143 TB                   |

A technique used to achieve memory saving is *Quantization*. The intuition behind this approach is that we can discretize  floating-point values by mapping their range [`f_max`, `f_min`] into a smaller range of fixed-point numbers [`q_max`, `q_min`], and linearly mapping all values between these ranges. In practice, this typically reduces the precision of a 32-bit floating-point to lower bit widths like 8-bits (scalar-quantization) or 1-bit values (binary quantization).

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/9r3rD0Dk2H4FnvVOC3rro.png">
  <figcaption style="text-align: center;">Scalar Embedding Quantization - from <em>float32</em> to <em>(u)int8</em></figcaption>
</figure>

By plotting the frequency distribution of our embeddings, we observe that the values are indeed concentrated around a relatively narrow range [-2.0, +2.0]. This means we can effectively map the `float32` values to 256 `(u)int8` buckets without significant loss of information.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/Cx578eTvr8z3cj7yX7Nn5.png">
  <figcaption style="text-align: center;">Original <em>float32</em> jina-embeddings-v2 distribution</figcaption>
</figure>

We can easily calculate the `[f_min, f_max]` values.

```python
>>> np.max(embeddings), np.min(embeddings)
(2.074683, -2.0162134)
```

A conservative calibration set of 10k embeddings would cover 99.8% of the original `float32` values. This calibration practice intents to obtain representative `f_min` and `f_max` values without the computational overhead and potential issues caused by outliers that might appear in larger datasets.

```python
calibration_embeddings = embeddings[:10000]
f_min = np.min(calibration_embeddings)
f_max = np.max(calibration_embeddings)

# calculate percentage in range
f32_values = np.sum((embeddings >= f_min) & (embeddings <= f_max))
percentage = (f32_values / embeddings.size) * 100

>>> 
>>>
```

This scalar quantization approach can be easily applied with [Sentence Transformers](https://www.sbert.net/), resulting in a 4x memory saving compared to the original float32 representation. Moreover, we will also benefit from faster arithmetic operations since matrix multiplication can be performed faster with integer arithmetic. 

```python
from sentence_transformers.quantization import quantize_embeddings

# quantization is applied in a post-processing step
int8_embeddings = quantize_embeddings(
    np.array(embeddings),
    precision="int8",
    calibration_embeddings=np.array(embeddings[:10000]),
)
```

```python
>>> fp32_embeddings.dtype, fp32_embeddings.shape, fp32_embeddings.nbytes
(dtype('float32'), (25107, 768), 77128704) # 73.5 MB

>>> int8_embeddings.dtype, int8_embeddings.shape, int8_embeddings.nbytes
(dtype('int8'), (25107, 768), 19282176)    # 18.3 MB
```

## 3. Embedding Projection for Dimensionality Reduction

In order to reduce the computational complexity and memory usage during clustering, we project the (*title*:*abstract*) embedding pairs from a high-dimensional space (768) to a lower-dimensional one (5 dimensions) using [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction). In this regard, [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection) [3] is a popular technique known for its effectiveness in preserving both the local and global data structures.

```python
import umap

umap_embedding = umap.UMAP(n_neighbors=100, # consider 100 nearest neighbors for each point
                           n_components=5,  # reduce embedding space from 768 to 5 dimensions
                           min_dist=0.1,    # maintain local and global balance
                           metric='cosine').fit_transform(ds['embeddings'])
```

Note that when we apply HDBSCAN clustering in the next step, the clusters found will be influenced by how UMAP preserved the local structures. A smaller `n_neighbors` value means UMAP will focus more on local structures, whereas a larger value allows to capture more global representations, which might be beneficial for understanding overall patterns in the data.

## 4. Semantic Clustering

This section shows how to use the reduced (*title*:*abstract*) embeddings as input features of a clustering algorithm. This enables identification of related categories based on embedding distances.

We have opted for [HDBSCAN](https://en.wikipedia.org/wiki/HDBSCAN) (Hierarchical Density-Based Spatial Clustering of Applications with Noise) [4], an advanced clustering algorithm that extends DBSCAN by adapting to varying density clusters. Unlike K-Means which requires pre-specifying the number of clusters, HDBSCAN has only one important hyperparameter, `n`, which establishes the minimum number of examples to include in a cluster. 

HDBSCAN works by first transforming the data space according to the density of the data points, making denser regions (areas where data points are close together in high numbers) more attractive for cluster formation. The algorithm then builds a hierarchy of clusters based on the minimum cluster size established by the hyperparameter `n`. This allows it to distinguish between noise (sparse areas) and dense regions (potential clusters). Finally, HDBSCAN condenses this hierarchy to derive the most persistent clusters, efficiently identifying clusters of different densities and shapes. As a density-based method, it can also detect outliers in the data. 

```python
import hdbscan

hdbs = hdbscan.HDBSCAN(min_cluster_size=100,            # conservative clusters' size
                       metric='euclidean',              # points distance metric
                       cluster_selection_method='leaf') # favour more fine grained clustering
clusters = hdbs.fit_predict(umap_embedding)             # apply HDBSCAN on reduced UMAP
```

The `cluster_selection_method` determines how HDBSCAN selects flat clusters from the tree hierarchy. Using `eom` (Excess of Mass) in combination with embedding quantization appeared to favour some larger clusters that would have needed a reclustering step. Instead, by using the `leaf` selection method, the algorithm selects leaf nodes from the tree, and produced a more fine grained clustering than Excess of Mass.

The `cluster_selection_method` determines how HDBSCAN selects flat clusters from the tree hierarchy. Using `eom` (Excess of Mass) cluster selection method in combination with embedding quantization, tended to create a few larger, less specific clusters. These clusters would have required a further *reclustering process* to extract meaningful latent topics. Instead, by switching to the `leaf` selection method, the algorithm selects leaf nodes from the cluster hierarchy, and produced a more fine grained clustering compared to the Excess of Mass method.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/20_VarYLBZxlND0vtDlLy.png">
  <figcaption style="text-align: center;">HDBSCAN <em>eom</em> & <em>leaf</em> clustering method comparison using <em>int8-embedding-quantization</em></figcaption>
</figure>

## 5. Uncovering Latent Topics with an LLM-Pipeline

Having performed the clustering step, we now illustrate how to infer the latent topic of each cluster by combining an LLM such as [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) [5] with [Pydantic](https://docs.pydantic.dev/) and [LangChain](https://www.langchain.com/) to create an LLM pipeline that generates output in a structured format.

### 4.1 Pydantic Model

[Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/) are classes that derive from `pydantic.BaseModel`, defining fields as type-annotated attributes. They are similar to `Python` dataclasses. However, they have been designed with subtle but significant differences that optimize various operations such as validation, serialization, and `JSON` schema generation. Our `Topic` class defines a field named `label`. This will generate output in a structured format, rather than a free-form text block, facilitating easier processing and analysis of the inference results.

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

### 4.3 Inference Chain using LangChain Expression Language

This section illustrates how to compose a topic modeling pipeline using the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/).

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

To enable the model to infer the topic of each cluster, we provide a random subset of `25` arXiv titles from each cluster as input.

```python
topics = []
for i, cluster in df.groupby('cluster'):
    titles = cluster['title'].sample(25).tolist()
    topic = TopicModeling(titles)
    topics.append(topic.label)
```

In the next step, we map the latent topics identified by the LLM pipeline to the corresponding clusters.

```python
n_clusters = len(df['cluster'].unique())

topic_map = dict(zip(range(n_clusters), topics))
df['topic'] = df['cluster'].map(topic_map)
```

## 6. Results

### 6.1 Analyzing the effects of Quantization on Clustering

We prepare the dataset for visualization by further reducing the number of dimensions, in this case to `(x, y)` dimensions:

```python
x_y_embeddings = umap.UMAP(n_neighbors=100, # consider 100 nearest neighbors for each point
                           n_components=2,  # reduce embeddings space from 5 to 2 dimensions
                           min_dist=0.1,    # maintain local and global structure balance
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
    title='Clustering and Topic Modeling | 25k arXiv cs.CL publications)',
    width=600,
    height=400,
)
chart.display()
```

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/coTOJJMwDXoNoj5_dNgQe.png">
  <figcaption style="text-align: center;">HDBSCAN <em>float32</em> & <em>quantized-int8</em> leaf-clustering comparison</figcaption>
</figure>

The clustering results using `float32` and `int8` quantized embeddings show a similar general layout of well-defined clusters, indicating that (i) the HDBSCAN clustering algorithm was effective in both cases, and (ii) the core relationships in the data were maintained after quantization.

**Notably, it can be observed that using embedding quantization resulted in slightly more granular clustering (34 clusters versus 31) that appears to be semantically coherent. Our tentative hypothesis is that quantization might *paradoxically* guide the clustering algorithm to separate points that were previously grouped together.** This could be due to (i) noise (quantization might create variations in the data) and numerical precision (a reduced precision could act as a sort of regularization and lead to more decisive clustering decisions, as the algorithm has fewer in-between values to consider), or due to (ii) the alteration of distance calculations (this could amplify certain differences between points that were less pronounced in the `float32` representation). Further investigation with a larger dataset would be necessary to fully understand the implications of quantization on clustering.

### 6.2 Taxonomy Completion

The results suggest emerging research domains around Language Models (LLMs) in the field of Computational Linguistics (cs.CL) that might serve as a baseline for a candidate classification scheme within the [arXiv cs.CL category](https://arxiv.org/category_taxonomy). 

**Initial 'cs.CL' classification scheme**

**Candicate 'cs.CL' classification scheme**


## Resources

- [GitHub Repo](https://github.com/dcarpintero/llm-based-categorization-taxonomy-completion)

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

## Frameworks

- [Hugging Face Hub](https://huggingface.co/docs/hub/index)
- [LangChain](https://python.langchain.com/v0.2/docs/introduction/) | [LangChain Prompt Templates](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) | [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [Sentence Transformers](https://www.sbert.net/index.html) | [Sentence Transformers Quantization](https://sbert.net/examples/applications/embedding-quantization/README.html)