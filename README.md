# An Empirical Study on Training Paradigms for Deep Supervised Hashing

This project focuses on systematically evaluating the two main training paradigms in deep supervised hashing: pairwise hashing and pointwise hashing. Deep supervised hashing has become essential for large-scale image retrieval tasks, offering efficient storage and retrieval capabilities by transforming high-dimensional image data into compact binary hash codes. The study provides an extensive quantitative exploration, comparing the performance of these paradigms across multiple datasets.

## Dataset and Experimental Setup

The experiments are conducted on both single-label and multi-label datasets, utilizing a variety of hash code dimensions (e.g., 16-bit, 32-bit, and 64-bit). The evaluation protocol covers 1,833 experiments, involving 17 different methods across 9 single-label datasets (3 generic and 6 fine-grained) and 3 multi-label datasets, ensuring a comprehensive assessment of retrieval performance under various conditions, including seen and unseen class scenarios.

### Datasets:
- **Generic single-label datasets**: ImageNet-1K, CIFAR-10, and CIFAR-100.
- **Fine-grained single-label datasets**: CUB200-2011, Food101, Aircraft, NABirds, Stanford Dogs, and VegFru.
- **Multi-label datasets**: COCO, NUS-WIDE, and Flickr25K.

*Note*: Following previous DSH settings, models are pretrained on the ImageNet-1K dataset, which may lead to data leakage on ImageNet-1K and Stanford Dogs.

### Evaluation Protocols:
1. **seen@seen**: Both query images and database images are from seen classes.
2. **seen@all**: Based on the "seen@seen" protocol, unseen classes' images are added to the database.
3. **unseen@unseen**: Both query images and database images are from unseen classes.
4. **unseen@all**: Expanding on the "unseen@unseen" basis, the database is extended to include both seen and unseen images.

The evaluation metrics for these four tasks are mAP@k, calculated as:

$${\rm mAP}@k = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\min (m_q, k)} \sum_{t=1}^{\min (n_q,k)} P_q(t)rel_q(t)$$

where $Q$ is the number of query images; $m_q$ is the number of index images containing a landmark in common with the query image $q$ ($m_q > 0$); $n_q$ represents the number of predictions made by different methods for query $q$; $P_q(t)$ is the precision at rank $t$ for the $q$-th query. While $rel_q(t)$ denotes the relevance of prediction $t$ for the $q$-th query: it equals $1$ if the $t$-th prediction is correct, and $0$ otherwise.  

### Experimental Settings:

We standardized the experimental configurations across different datasets for equitable comparison by amalgamating existing pairwise and pointwise hashing methods. The training phase consists of iterations, each with a specific number of epochs. For instance, 50 iterations with 3 epochs per iteration results in a total of 150 epochs.

#### Settings for Generic Single-label Datasets:
- **CIFAR-10** and **CIFAR-100**: Randomly selected 2000 samples from the training set, with 50 iterations and 3 epochs. The evaluation metric is mAP@1000.
- **ImageNet-1K**: 130,000 images in 100 classes for training, and 5,000 for testing. Iteration: 50, Epoch: 1, Evaluation metric: mAP@1000.

We established 3 partition configurations for **CIFAR-100** and **ImageNet-1K**:
- The former 95%/85%/75% categories as seen classes.
- The latter 5%/15%/25% categories as unseen classes.
For **CIFAR-10**, we only set 1 configuration: the former 80% categories as seen classes, and the latter 20% as unseen classes.

#### Settings for Multi-label Datasets:
- **Flickr-25K**: Sampled 1000 images as queries and 20,000 as database points. Iteration: 50, Epoch: 3, Evaluation metric: mAP@5000.
- **NUS-WIDE**: 21 most frequent categories used for evaluation. Iteration: 50, Epoch: 3, Evaluation metric: mAP@5000.
- **MS COCO**: Pruned images with no category information. 82,081 images from the training set as database points, and 5,000 images from the validation set as queries. Iteration: 50, Epoch: 3, Evaluation metric: mAP@5000.

For multi-label datasets, two configurations were implemented:
1. **Seen classes (~95% of dataset)**: 32 categories (Flickr-25K), 10 categories (NUS-WIDE), 45 categories (MS COCO).
2. **Seen classes (~85% of dataset)**: 23 categories (Flickr-25K), 5 categories (NUS-WIDE), 27 categories (MS COCO).

#### Settings for Fine-grained Single-label Datasets:
- Datasets: **CUB200-2011**, **Food101**, **VegFru**, **Stanford Dogs**, **Aircraft**, **NABirds**.
- Iteration: 40, Epoch: 30.
- Randomly selected 2000 samples from the training set, but 4000 for **VegFru** and **NABirds**.

We established 3 partition configurations for these datasets:
- The former 95%/85%/75% categories as seen classes.
- The latter 5%/15%/25% categories as unseen classes.

For multi-label datasets, two images will be defined as a ground-truth neighbor (similar pair) if they share at least one common label.
