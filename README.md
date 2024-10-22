# An Empirical Study on Training Paradigms for Deep Supervised Hashing

This project focuses on systematically evaluating the two main training paradigms in deep supervised hashing: pairwise hashing and pointwise hashing. Deep supervised hashing has become essential for large-scale image retrieval tasks, offering efficient storage and retrieval capabilities by transforming high-dimensional image data into compact binary hash codes. The study provides an extensive quantitative exploration, comparing the performance of these paradigms across multiple datasets.

## Dataset and Experimental Setup

The experiments are conducted on both single-label and multi-label datasets, utilizing a variety of hash code dimensions (e.g., 16-bit, 32-bit, and 64-bit). The evaluation protocol covers 1,833 experiments, involving 17 different methods across 9 single-label datasets (3 generic and 6 fine-grained) and 3 multi-label datasets, ensuring a comprehensive assessment of retrieval performance under various conditions, including seen and unseen class scenarios.

### Datasets:
Generic single-label datasets: ImageNet-1K, CIFAR-10, and CIFAR-100.
Fine-grained single-label datasets: CUB200-2011, Food101, Aircraft, NABirds, Stanford Dogs and VegFru.
Multi-label datasets: COCO, NUS-WIDE, and Flickr25K.
(Following previous DSH settings, models are pretrain on the ImageNet-1K dataset, thus lead to data leakage on ImageNet-1K and Stanford Dogs.)

### Evaluation Protocols
1) seen@seen: Both query images and database images are from seen classes; 
2) seen@all: Building on the ``seen@seen" protocol, unseen classes' images are added to the database;
3) unseen@unseen: Both query images and database images are from unseen classes;
4) unseen@all: Expanding on the ``unseen@unseen" basis, the database is extended to include both seen and unseen images.

The evaluation metrics for these four tasks are mAP@$k$:
\begin{equation}
    {\rm mAP}@k = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\min (m_q, k)} \sum_{t=1}^{\min (n_q,k)} P_q(t)rel_q(t)\,,
\end{equation}
where $Q$ is the number of query images; $m_q$ is the number of index images containing a landmark in common with the query image $q$ ($m_q > 0$); $n_q$ represents the number of predictions made by different methods for query $q$; $P_q(t)$ is the precision at rank $t$ for the $q$-th query. While $rel_q(t)$ denotes the relevance of prediction $t$ for the $q$-th query: it equals $1$ if the $t$-th prediction is correct, and $0$ otherwise.  

### Experimental Settings:
By amalgamating the experimental setups of existing pairwise hashing and pointwise hashing methods, we standardized the experimental configurations across different datasets for equitable comparison. During the training phase, each iteration comprises a specific number of epochs. Taking an instance of 50 iterations with 3 epochs per iteration, the total training epochs amount to $50 \times 3 = 150$.

#### Settings for Generic Single-label Datasets
For the \emph{CIFAR-10} and \emph{CIFAR-100} datasets, we randomly selected 2000 samples from the training set. The iteration is set as 50 while the epoch is set as 3. The evaluation metric is mAP@1000. For the \emph{ImageNet-1K} dataset, we follow settings in CSQ and adopt 130,000 images in 100 classes for training while 5,000 for test. The iteration is set as 50 while the epoch is set as 1. The evaluation metric is mAP@1000. We established 3 distinct partitioning configurations for \emph{CIFAR-100} and \emph{ImageNet-1K}. Specifically, we select the former 95\%/85\%/75\% categories as seen classes, whereas the later 5\%/15\%/25\% categories as unseen classes. For the \emph{CIFAR-10} dataset, we only set 1 configuration, \emph{i.e.}, the former 80\% categories as seen classes, whereas the later 20\% as sunseen classes. 

#### Settings for Multi-label Datasets
For the \emph{Flickr-25K} dataset, we sample 1000 images as queries and 20000 images as database points. The iteration is set as 50 and we randomly sample 2000 points from the database per iteration for training. The epoch is set as 3 and the evaluation metric is mAP@5000. Following DPSH and ADSH, we choose images from the 21 most frequent categories for evaluation for the \emph{NUS-WIDE} dataset. The iteration is set as 50 while the epoch is set as 3. We randomly selected 2000 samples per iteration and the evaluation metric is mAP@5000. By combining settings from HashNet~\cite{cao2017hashnet} and ADSH~\cite{qingyuanAAAI18}, we prune images from the \emph{MS COCO} dataset with no category information. We obtain 82081 images from the train set and directly use them as database points. We randomly sample 5000 images from the validation set as queries. The iteration is set as 50 while the epoch is set as 3. We randomly selected 2000 samples from the database per iteration and the evaluation metric is mAP@5000. For each multi-label generic dataset, we implement 2 arrangements. For the first arrangement, around 95\% of the dataset's images are as seen classes images. Specifically, we designate 32 categories as seen classes for the \emph{Flickr-25K} dataset, 10 categories as seen classes for the \emph{NUS-WIDE} dataset and 45 categories as seen classes for the \emph{MS COCO} dataset. While for the second arrangement, around 85\% of the dataset's images are set as seen classes images. Specifically, we designate 23 categories as seen classes for the \emph{Flickr-25K} dataset, 5 categories as seen classes for the \emph{NUS-WIDE} dataset and 27 categories as seen classes for the \emph{MS COCO} dataset. The training and query sets of the multi-label datasets are reorganized according to the label divisions specified by each arrangement. Two images will be defined as a ground-truth neighbor (similar pair) if they share at least one common label.

#### Settings for Fine-grained Single-label Datasets
For the \emph{CUB200-2011}, \emph{Food101}, \emph{VegFru}, \emph{Stanford Dogs}, \emph{Aircraft} and \emph{NABirds} datasets, the iteration is set as 40 while the epoch is set as 30. We randomly selected 2000 samples from the training set for \emph{CUB200-2011}, \emph{Food101}, \emph{Stanford Dogs} and \emph{Aircraft} but 4000 samples for \emph{VegFru} and \emph{NABirds}. We established 3 distinct partitioning configurations for these datasets. Specifically, we select the former 95\%/85\%/75\% categories as seen classes, whereas the later 5\%/15\%/25\% categories as unseen classes.
