# Divide-and-Conquer-the-Embedding-Space-for-Metric-Learning
Computer Vision project

Link to paper: http://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf

- [x] Model script
- [x] Data Loader script
- [x] Loss script
- [x] Training and testing scripts
- [x] Query script for recommendation

1. Overview:

In an embedding space, we generally try to learn one distance metric and use that to train the given data. The issue with this is that it results in overfitting since the embedding space can be very large and based on a single distance metric, the model does not generalize well. ‘Divide and Conquer the Embedding Space for Metric Learning’ (CVPR 2019), aims to solve this problem by dividing the metric space into non-overlapping subsets of data and applying separate learning for each by an ensemble method. Due to the fact that it employs the embedding space in a better manner, the model outperforms state of the art approaches in metric learning and improves generalization. 

The applications of this model are plenty in image retrieval, and in building recommendation systems due to increased accuracy in clustering.

2. Method 
![Method doe divide nand conquer](https://camo.githubusercontent.com/d37e8f3401e53b9fc809e359e834300a6b6092a8/68747470733a2f2f6173616e616b6f792e6769746875622e696f2f696d616765732f7465617365725f6376707231395f646d6c2e6a7067)
- The embeddings for the training images are computed and then clustered into K-clusters (disjoint subsets).
- The d-dimensional embedding spaces are then split into K subspaces of d/K dimensions each.
- Next, for each of the subspaces, a separate loss is assigned and trained. Different weights are learned for different subspaces, and we get K different distance metrics for the disjoint learners.
- This process is repeated every T-epoch.

3. Contributors:
- Ajay Shrihari
- Aniket Mohanty
- Dharmala Amarthya Sasi Kiran


