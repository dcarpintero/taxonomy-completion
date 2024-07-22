# Taxonomy Completion with Embedding Quantization and an LLM-based Pipeline: A Case Study in Computational Linguistics

[![GitHub license](https://img.shields.io/github/license/dcarpintero/taxonomy-completion)](https://github.com/dcarpintero/taxonomy-completion/blob/main/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/dcarpintero/taxonomy-completion.svg)](https://GitHub.com/dcarpintero/taxonomy-completion/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/dcarpintero/taxonomy-completion.svg)](https://GitHub.com/dcarpintero/taxonomy-completion/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/dcarpintero/taxonomy-completion.svg)](https://GitHub.com/dcarpintero/taxonomy-completion/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dcarpintero/taxonomy-completion/blob/main/nb.taxonomy-completion-with-embedding-quantization-and-llms.ipynb)

[![GitHub watchers](https://img.shields.io/github/watchers/dcarpintero/taxonomy-completion.svg?style=social&label=Watch)](https://GitHub.com/dcarpintero/taxonomy-completion/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/dcarpintero/taxonomy-completion.svg?style=social&label=Fork)](https://GitHub.com/dcarpintero/taxonomy-completion/network/)
[![GitHub stars](https://img.shields.io/github/stars/dcarpintero/taxonomy-completion.svg?style=social&label=Star)](https://GitHub.com/dcarpintero/taxonomy-completion/stargazers/)

## Introduction

The ever-growing volume of research publications necessitates efficient methods for structuring academic knowledge. This task typically involves developing a supervised underlying scheme of classes and allocating publications to the most relevant class. In this article, we implement an end-to-end automated solution using embedding quantization and a Large Language Model (LLM) pipeline.  Our case study starts with a dataset of [25,000 arXiv publications](https://huggingface.co/datasets/dcarpintero/arxiv.cs.CL.25k) from Computational Linguistics (cs.CL), published before July 2024, which we organize under a novel scheme of classes.

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/Kp5GNW9dUKYQZtAKSAm3q.png">
</p>

## Methodology

Our approach centers on three key tasks: (i) unsupervised clustering of the arXiv dataset into related collections, (ii) discovering the latent thematic structures within each cluster, and (iii) creating a candidate taxonomy scheme based on said thematic structures.

At its core, the clustering task requires identifying a sufficient number of similar examples within an *unlabeled* dataset.
This is a natural task for embeddings, as they capture semantic relationships in a corpus and can be provided as input features to a clustering algorithm for establishing similarity links among examples. We begin by transforming the (*title*:*abstract*) pairs of our dataset into an embeddings representation using [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923), a BERT-ALiBi based attention model. And applying scalar quantization using both [Sentence Transformers](https://www.sbert.net/) and a custom implementation.

For clustering, we run [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) in a reduced dimensional space, comparing the results using `eom` and `leaf` clustering methods. Additionally, we examine whether using `(u)int8` embeddings quantization instead of `float32` representations affects this process.

To uncover latent topics within each cluster of arXiv publications, we combine [LangChain](https://www.langchain.com/) and [Pydantic](https://docs.pydantic.dev/) with [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (and [GPT-4o](https://platform.openai.com/docs/models/gpt-4o), included for comparison) into an LLM-pipeline. The output is then incorporated into a refined prompt template that guides [Claude Sonnet 3.5](https://docs.anthropic.com/en/docs/welcome) in generating a hierarchical taxonomy.

The results hint at 35 emerging research topics, wherein each topic comprises at least `100` publications. These are organized within 7 parent classes in the field of Computational Linguistics (cs.CL). This approach may serve as a baseline for automatically generating hierarchical candidate schemes in high-level [arXiv categories](https://arxiv.org/category_taxonomy) and efficiently completing taxonomies, addressing the challenge posed by the increasing volume of academic literature.

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/fbCiM9DfjvDFThUQYngIO.png">
</p>
<p align="center">Taxonomy Completion of Academic Literature with Embedding Quantization and an LLM-Pipeline</p>

## 1. Embedding Transformation

Embeddings are numerical representations of real-world objects like text, images, and audio that encapsulate semantic information of the data they represent. They are used by AI models to understand complex knowledge domains in downstream applications such as clustering, information retrieval, and semantic understanding tasks, among others.

#### Supporting Large Sequences

We will map (*title*:*abstract*) pairs from arXiv publications to a 768-dimensional space using [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923) [1], an open-source text embedding model capable of accommodating up to 8192 tokens. This provides a sufficiently large sequence length for titles, abstracts, and other document sections that might be relevant. To overcome the conventional 512-token limit  present in other models, Jina-Embeddings-v2 incorporates bidirectional [ALiBi](https://arxiv.org/abs/2108.12409) [2] into the BERT framework. ALiBi (Attention with Linear Biases) enables input length extrapolation (i.e., sequences exceeding 2048 tokens) by encoding positional information directly within the self-attention layer, instead of introducing positional embeddings. In practice, it biases query-key attention scores with a penalty that is proportional to their distance, favoring stronger mutual attention between proximate tokens.

#### Encoding with Sentence Transformers

The first step to using the [Jina-Embeddings-v2](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) model is to load it through [Sentence Transformers](https://www.SBERT.net), a framework for accessing state-of-the-art models that is available at the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers&sort=downloads):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
```

We now encode (*title*:*abstract*) pairs of our [dataset]() using `batch_size = 64`. This allows for parallel computation on hardware accelerators like GPUs (albeit at the cost of requiring more memory):

```python
from datasets import load_dataset
ds = load_dataset("dcarpintero/arxiv.cs.CL.25k", split="train")

corpus = [title + ':' + abstract for title, abstract in zip(ds['title'], ds['abstract'])]
f32_embeddings = model.encode(corpus,
                              batch_size=64,
                              show_progress_bar=True)
```

#### Computing Semantic Similarity

The semantic similarity between corpora can now be trivially computed as the inner product of embeddings. In the following heat map, each entry [x, y] is colored based on said embeddings product for exemplary '*title*' sentences [x] and [y].

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/4djmELIe2LkZ8_Tofc91Q.png">
</p>
<p align="center">Semantic Similary in <em>cs.CL arXiv-titles</em> using Embeddings</p>

## 2. Embedding Quantization for Memory Saving

Scaling up embeddings can be challenging. Currently, state-of-the-art models represent each embedding as `float32`, which requires 4 bytes of memory. Given that [Jina-Embeddings-v2](https://arxiv.org/abs/2310.19923) maps text to a 768-dimensional space, the memory requirements for our dataset would be around 73 MB, without indexes and other metadata related to the publication records:

```python
25,000 embeddings * 768 dimensions/embedding * 4 bytes/dimension = 76,800,000 bytes
76,800,000 bytes / (1024^2) ≈ 73.24 MB
```

However, working with a larger dataset might increase significantly the memory requirements and associated costs:

| Embedding<br>Dimension | Embedding<br>Model            | 2.5M<br>ArXiv Abstracts      | 60.9M<br>Wikipedia Pages | 100M<br>Embeddings |
|------------------------|-------------------------------|------------------------------|-----------------------|------------------------------|
| 384                    | all-MiniLM-L12-v2             | 3.57 GB                      | 85.26 GB              | 142.88 GB                    |
| 768                    | all-mpnet-base-v2             | 7.15 GB                      | 170.52 GB             | 285.76 GB                    |
| 768                    | jina-embeddings-v2            | 7.15 GB                      | 170.52 GB             | 285.76 GB                    |
| 1536                   | openai-text-embedding-3-small | 14.31 GB                     | 341.04 GB             | 571.53 GB                    |
| 3072                   | openai-text-embedding-3-large | 28.61 GB                     | 682.08 GB             | 1.143 TB                   |

A technique used to achieve memory saving is *Quantization*. The intuition behind this approach is that we can discretize  floating-point values by mapping their range [`f_max`, `f_min`] into a smaller range of fixed-point numbers [`q_max`, `q_min`], and linearly distributing all values between these ranges. In practice, this typically reduces the precision of a 32-bit floating-point to lower bit widths like 8-bits (scalar quantization) or 1-bit values (binary quantization).

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/8PF8uD8wgk12Uuejddhnw.png">
</p>
<p align="center">Scalar Embedding Quantization - from <em>float32</em> to <em>(u)int8</em></p>

By plotting the frequency distribution of the *Jina-generated* embeddings, we observe that the values are indeed concentrated around a relatively narrow range [-2.0, +2.0]. This means we can effectively map `float32` values to 256 `(u)int8` buckets without significant loss of information:

```python
import matplotlib.pyplot as plt

plt.hist(f32_embeddings.flatten(), bins=250, edgecolor='C0')
plt.xlabel('float-32 jina-embeddings-v2')
plt.title('distribution')
plt.show()
```

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/Cx578eTvr8z3cj7yX7Nn5.png">
</p>
<p align="center">Original <em>float32 jina-embeddings-v2</em> distribution</p>

We can calculate the exact `[min, max]` values of the distribution:

```python
>>> np.min(f32_embeddings), np.max(f32_embeddings)
(-2.0162134, 2.074683)
```

The first step to implementing scalar quantization is to define a calibration set of embeddings. A typical starting point is a subset of 10k embeddings, which in our case would cover nearly 99.98% of the original `float32` embedding values. The use of calibration is intended to obtain representative `f_min` and `f_max` values along each dimension to reduce the computational overhead and potential issues caused by outliers that might appear in larger datasets.

```python
def calibration_accuracy(embeddings: np.ndarray, k: int = 10000) -> float:
  calibration_embeddings = embeddings[:k]
  f_min = np.min(calibration_embeddings, axis=0)
  f_max = np.max(calibration_embeddings, axis=0)

  # Calculate percentage in range for each dimension
  size = embeddings.shape[0]
  avg = []
  for i in range(embeddings.shape[1]):
      in_range = np.sum((embeddings[:, i] >= f_min[i]) & (embeddings[:, i] <= f_max[i]))
      dim_percentage = (in_range / size) * 100
      avg.append(dim_percentage)

  return np.mean(avg)

acc = calibration_accuracy(f32_embeddings, k=10000)
print(f"Average percentage of embeddings within [f_min, f_max] calibration: {acc:.5f}%")
>>> Average percentage of embeddings within [f_min, f_max] calibration: 99.98636%
```

The second and third steps of scalar quantization — *computing scales and zero point*, and *encoding* — can be easily applied with [Sentence Transformers](https://www.sbert.net/), resulting in a 4x memory saving compared to the original `float32` representation. Moreover, we will also benefit from faster arithmetic operations since matrix multiplication can be performed more quickly with integer arithmetic. 

```python
from sentence_transformers.quantization import quantize_embeddings

# quantization is applied in a post-processing step
int8_embeddings = quantize_embeddings(
    np.array(f32_embeddings),
    precision="int8",
    calibration_embeddings=np.array(f32_embeddings[:10000]),
)
```

```python
f32_embeddings.dtype, f32_embeddings.shape, f32_embeddings.nbytes
>>> (dtype('float32'), (25107, 768), 77128704) # 73.5 MB

int8_embeddings.dtype, int8_embeddings.shape, int8_embeddings.nbytes
>>> (dtype('int8'), (25107, 768), 19282176)    # 18.3 MB

# calculate compression
(f32_embeddings.nbytes - int8_embeddings.nbytes) / f32_embeddings.nbytes * 100
>>> 75.0
```

For completeness, we implement a scalar quantization method to illustrate those three steps:

```python
def scalar_quantize_embeddings(embeddings: np.ndarray,
                               calibration_embeddings: np.ndarray) -> np.ndarray:

    # Step 1: Calculate [f_min, f_max] per dimension from the calibration set 
    f_min = np.min(calibration_embeddings, axis=0)
    f_max = np.max(calibration_embeddings, axis=0)

    # Step 2: Map [f_min, f_max] to [q_min, q_max] => (scaling factors, zero point)
    q_min = 0
    q_max = 255
    scales = (f_max - f_min) / (q_max - q_min)
    zero_point = 0 # uint8 quantization maps inherently min_values to zero

    # Step 3: encode (scale, round)
    quantized_embeddings = ((embeddings - f_min) / scales).astype(np.uint8)

    return quantized_embeddings
```

```python
calibration_embeddings = f32_embeddings[:10000]
beta_uint8_embeddings = scalar_quantize_embeddings(f32_embeddings, calibration_embeddings)
```

```python
beta_uint8_embeddings[5000][64:128].reshape(8, 8)

array([[187, 111,  96, 128, 116, 129, 130, 122],
       [132, 153,  72, 136,  94, 120, 112,  93],
       [143, 121, 137, 143, 195, 159,  90,  93],
       [178, 189, 143,  99,  99, 151,  93, 102],
       [179, 104, 146, 150, 176,  94, 148, 118],
       [161, 138,  90, 122,  93, 146, 140, 129],
       [121, 115, 153, 118, 107,  45,  70, 171],
       [207,  53,  67, 115, 223, 105, 124, 158]], dtype=uint8)
```


We will continue with the version of the embeddings that have been quantized using Sentence Transformers (our custom implementation is also included in the results analysis):

```python
# `f32_embeddings` => if you prefer to not use quantization
# `beta_uint8_embeddings` => to check our custom implemention
embeddings = int8_embeddings 
```

## 3. Projecting Embeddings for Dimensionality Reduction

In this section, we perform a two-stage projection of (*title*:*abstract*) embedding pairs from their original high-dimensional space (768) to lower dimensions, namely:
- `5 dimensions` for reducing computational complexity during clustering, and 
- `2 dimensions` for enabling visual representation in `(x, y)` coordinates.

For both projections, we employ [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection) [3], a popular dimensionality reduction technique known for its effectiveness in preserving both the local and global data structures. In practice, this makes it a preferred choice for handling complex datasets with high-dimensional embeddings:

```python
import umap

embedding_5d = umap.UMAP(n_neighbors=100, # consider 100 nearest neighbors for each point
                         n_components=5,  # reduce embedding space from 768 to 5 dimensions
                         min_dist=0.1,    # maintain local and global balance
                         metric='cosine').fit_transform(embeddings)

embedding_2d = umap.UMAP(n_neighbors=100,
                         n_components=2,
                         min_dist=0.1,
                         metric='cosine').fit_transform(embeddings)
```

Note that when we apply HDBSCAN clustering in the next step, the clusters found will be influenced by how UMAP preserved the local structures. A smaller `n_neighbors` value means UMAP will focus more on local structures, whereas a larger value allows capturing more global representations, which might be beneficial for understanding overall patterns in the data.

## 4. Semantic Clustering

The reduced (*title*:*abstract*) embeddings can now be used as input features of a clustering algorithm, enabling the identification of related categories based on embedding distances.

We have opted for [HDBSCAN](https://en.wikipedia.org/wiki/HDBSCAN) (Hierarchical Density-Based Spatial Clustering of Applications with Noise) [4], an advanced clustering algorithm that extends DBSCAN by adapting to varying density clusters. Unlike K-Means which requires pre-specifying the number of clusters, HDBSCAN has only one important hyperparameter, `n`, which establishes the minimum number of examples to include in a cluster. 

HDBSCAN works by first transforming the data space according to the density of the data points, making denser regions (areas where data points are close together in high numbers) more attractive for cluster formation. The algorithm then builds a hierarchy of clusters based on the minimum cluster size established by the hyperparameter `n`. This allows it to distinguish between noise (sparse areas) and dense regions (potential clusters). Finally, HDBSCAN condenses this hierarchy to derive the most persistent clusters, identifying clusters of different densities and shapes. As a density-based method, it can also detect outliers.

```python
import hdbscan

hdbs = hdbscan.HDBSCAN(min_cluster_size=100,            # conservative clusters' size
                       metric='euclidean',              # points distance metric
                       cluster_selection_method='leaf') # favour fine grained clustering
clusters = hdbs.fit_predict(embedding_5d)               # apply HDBSCAN on reduced UMAP
```

The `cluster_selection_method` determines how HDBSCAN selects flat clusters from the tree hierarchy. In our case, using `eom` (Excess of Mass) cluster selection method in combination with embedding quantization tended to create a few larger, less specific clusters. These clusters would have required a further *reclustering process* to extract meaningful latent topics. Instead, by switching to the `leaf` selection method, we guided the algorithm to select leaf nodes from the cluster hierarchy, which produced a more fine-grained clustering compared to the Excess of Mass method:

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/20_VarYLBZxlND0vtDlLy.png">
</p>
<p align="center">HDBSCAN <em>eom</em> & <em>leaf</em> clustering method comparison using <em>int8-embedding-quantization</em></p>

## 5. Uncovering Latent Topics with an LLM-Pipeline

Having performed the clustering step, we now illustrate how to infer the latent topic of each cluster by combining an LLM such as [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) [5] with [Pydantic](https://docs.pydantic.dev/) and [LangChain](https://www.langchain.com/) to create an LLM pipeline that generates output in a composable structured format.

### 5.1 Pydantic Model

[Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/) are classes that derive from `pydantic.BaseModel`, defining fields as type-annotated attributes. They are similar to `Python` dataclasses. However, they have been designed with subtle but significant differences that optimize various operations such as validation, serialization, and `JSON` schema generation. Our `Topic` class defines a field named `label`. This will generate LLM output in a structured format, rather than a free-form text block, facilitating easier processing and analysis.

```python
from pydantic import BaseModel, Field

class Topic(BaseModel):
    """
    Pydantic Model to generate an structured Topic Model
    """
    label: str = Field(..., description="Identified topic")
```

### 5.2 Langchain Prompt Template

[LangChain Prompt Templates](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) are pre-defined recipes for translating user input and parameters into instructions for a language model. We define here the prompt for our intended task:

```python
from langchain_core.prompts import PromptTemplate

topic_prompt = """
  You are a helpful research assistant. Your task is to analyze a set of research paper
  titles related to Natural Language Processing, and determine the overarching topic. 
            
  INSTRUCTIONS:

  1. Based on the titles provided, identify the most relevant topic:
    - Ensure the topic is concise and clear.
            
  2. Format Respose:
    - Ensure the title response is in JSON as in the 'OUTPUT OUTPUT' section below.
    - No follow up questions are needed.

  OUTPUT FORMAT:

  {{"label": "Topic Name"}}

  TITLES:
  {titles}
  """
```

### 5.3 Inference Chain using LangChain Expression Language

Let's now compose a topic modeling pipeline using [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) to render our prompt template into LLM input, and parse the inference output as `JSON`:

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

To enable the model to infer the topic of each cluster, we include a subset of 25 paper titles from each cluster as part of the LLM input:

```python
topics = []
for i, cluster in df.groupby('cluster'):
    titles = cluster['title'].sample(25).tolist()
    topic = TopicModeling(titles)
    topics.append(topic.label)
```

Let's assign each arXiv publication to its corresponding cluster:

```python
n_clusters = len(df['cluster'].unique())

topic_map = dict(zip(range(n_clusters), topics))
df['topic'] = df['cluster'].map(topic_map)
```

## 6. Generating a Taxonomy

To create a hierarchical taxonomy, we craft a prompt to guide [Claude Sonnet 3.5](https://docs.anthropic.com/en/docs/welcome) in organizing the identified research topics corresponding to each cluster into a hierarchical scheme:

```python
from langchain_core.prompts import PromptTemplate

taxonomy_prompt = """
    Create a comprehensive and well-structured taxonomy
    for the ArXiv cs.CL (Computational Linguistics) category.
    This taxonomy should organize subtopics in a logical manner.

    INSTRUCTIONS:

    1. Review and Refine Subtopics:
      - Examine the provided list of subtopics in computational linguistics.
      - Ensure each subtopic is clearly defined and distinct from others.

    2. Create Definitions:
      - For each subtopic, provide a concise definition (1-2 sentences).

    3. Develop a Hierarchical Structure:
      - Group related subtopics into broader categories.
      - Create a multi-level hierarchy, with top-level categories and nested subcategories.
      - Ensure that the structure is logical and intuitive for researchers in the field.

    4. Validate and Refine:
      - Review the entire taxonomy for consistency, completeness, and clarity.

    OUTPUT FORMAT:

    - Present the final taxonomy in a clear, hierarchical format, with:

      . Main categories
        .. Subcategories
          ... Individual topics with their definitions

    SUBTOPICS:
    {taxonomy_subtopics}
    """
```

## 7. Results

### 7.1 Clustering Analysis

Let's create an interactive scatter plot:

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

And compare the clustering results using `float32` embedding representations and `int8` [Sentence Transformers](https://www.sbert.net/) quantization:

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/e8nQw98dKSmLaNAKfx7T4.png">
</p>
<p align="center">HDBSCAN leaf-clustering using <em>float32</em> & <em>quantized-int8</em> embeddings (sentence-transformers-quantization)</em></p>

We now perform the same comparison with our custom quantization implementation:

<p align="center">
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/smL046VV2i4N1ulIRmykw.png">
</p>
<p align="center">HDBSCAN leaf-clustering using <em>float32</em> & <em>quantized-uint8</em> embeddings (custom-quantization-implementation)</p>

The clustering results using `float32` and `(u)int8` quantized embeddings show a similar general layout of well-defined clusters, indicating that (i) the HDBSCAN clustering algorithm was effective in both cases, and (ii) the core relationships in the data were maintained after quantization (using sentence transformers and our custom implementation).

Notably, it can be observed that using embedding quantization resulted in both cases in slightly more granular clustering (35 clusters versus 31) that appears to be semantically coherent. Our tentative hypothesis for this difference is that scalar quantization might *paradoxically* guide the HDBSCAN clustering algorithm to separate points that were previously grouped together.

This could be due to (i) noise (quantization can create small *noisy* variations in the data, which might have a sort of *regularization* effect and lead to more sensitive clustering decisions), or due to (ii) the difference in numerical precision and alteration of distance calculations (this could amplify certain differences between points that were less pronounced in the `float32` representation). Further investigation would be necessary to fully understand the implications of quantization on clustering.

### 7.2 Taxonomy Scheme

The entire scheme is available at [cs.CL.taxonomy](https://github.com/dcarpintero/taxonomy-completion/blob/main/arxiv/cs.CL.scheme.md). This approach may serve as a baseline for automatically identifying candidate schemes of classes in high-level [arXiv categories](https://arxiv.org/category_taxonomy):
```
. Foundations of Language Models
  .. Model Architectures and Mechanisms 
    ... Transformer Models and Attention Mechanisms
    ... Large Language Models (LLMs)
  .. Model Optimization and Efficiency
    ... Compression and Quantization
    ... Parameter-Efficient Fine-Tuning
    ... Knowledge Distillation
  .. Learning Paradigms
    ... In-Context Learning
    ... Instruction Tuning

. AI Ethics, Safety, and Societal Impact
  .. Ethical Considerations
    ... Bias and Fairness in Models
    ... Alignment and Preference Optimization
  .. Safety and Security
    ... Hallucination in LLMs
    ... Adversarial Attacks and Robustness
    ... Detection of AI-Generated Text
  .. Social Impact
    ... Hate Speech and Offensive Language Detection
    ... Fake News Detection

[...]
```

## Citation

```
@article{carpintero2024
  author = { Diego Carpintero},
  title = {Taxonomy Completion with Embedding Quantization and an LLM-Pipeline: A Case Study in Computational Linguistics},
  journal = {Hugging Face Blog},
  year = {2024},
  note = {https://huggingface.co/blog/dcarpintero/taxonomy-completion},
}
```

## References

- [1] Günther, et al. 2024. *Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents*. [arXiv:2310.19923](https://arxiv.org/abs/2310.19923).
- [2] Press, . et al. 2021. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409).
- [3] McInnes, et al. 2018. *Umap: Uniform manifold approximation and projection for dimension reduction*. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426).
- [4] Campello, et al. 2013. *Density-Based Clustering Based on Hierarchical Density Estimates. Advances in Knowledge Discovery and Data Mining*. Vol. 7819. Berlin, Heidelberg: Springer Berlin Heidelberg. pp. 160–172. [doi:10.1007/978-3-642-37456-2_14](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14).
- [5] Jiang, et al. 2023. *Mistral 7B*. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825).
- [6] Shakir, et al. 2024. *Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval*. [hf:shakir-embedding-quantization](https://huggingface.co/blog/embedding-quantization)
- [7] Liu, Yue, et al. 2024. *Agent Design Pattern Catalogue: A Collection of Architectural Patterns for Foundation Model based Agents"*. [arXiv:2405.10467](https://arxiv.org/abs/2405.10467).