# stack-overflow-code-retrieval

In this paper we aim at developing a robust code retrieval model that is capable of fetching relevant code snippets for the input query while ensuring a good measure of precision and recall. We implemented a model to convert the code into its control flow graph considering all the possible cases of flow change in python and have used similarity techniques based on eigen vectors, isomorphic structure of graphs and steady state of the adjacency matrices. We have also used different weighting factors to get the best possible combinations of the graph and text similarity.