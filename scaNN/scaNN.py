!pip install --quiet scann
import scann
import numpy as np
data = np.random.rand(500,6)
# print(data)
query = np.random.rand(1,6)
print(query)
# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
searcher = scann.scann_ops_pybind.builder(data, 8, "dot_product").tree(
    num_leaves=3, num_leaves_to_search=100, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()
neighbors, distances = searcher.search_batched(query)
print(neighbors)
print(distances)


query2 = np.random.rand(2,6)
neighbors2, distance2 = searcher.search_batched(query2)
print(neighbors2)
