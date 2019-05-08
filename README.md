# stack-overflow-code-retrieval

In this paper we aim at developing a robust code retrieval model that is capable of fetching relevant code snippets for the input query while ensuring a good measure of precision and recall. We implemented a model to convert the code into its control flow graph considering all the possible cases of flow change in python and have used similarity techniques based on eigen vectors, isomorphic structure of graphs and steady state of the adjacency matrices. We have also used different weighting factors to get the best possible combinations of the graph and text similarity.

Actual dataset link: https://www.kaggle.com/stackoverflow/pythonquestions

Contents of the python files:
1. model_final.py - complete running code
2. model_modular.py - individual functions of models and other implementations
3. sampling.py - code to sample data from original CSVs.
4. replaceVar.py - code to replace arbitrary variables in code for syntax matching
5. Generate-Control-Flow.py - code to generate control flow graphs out of code

To execute the above files:

a) Navigate to the folder where these files re present in your terminal
b) Execute the following command: python3 <filename.py> to execute the file