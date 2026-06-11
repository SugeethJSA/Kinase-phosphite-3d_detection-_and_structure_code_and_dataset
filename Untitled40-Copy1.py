import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = r"C:\Users\Tim\kinase_interactions.csv"
data = pd.read_csv(file_path)

# Step 2: Inspect the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Summary:")
print(data.info())
# Step 3: Filter for high-confidence interactions
confidence_threshold = 0.7  # Adjust this value as needed
filtered_data = data[data["Interaction Score"].astype(float) >= confidence_threshold]
print(f"\nNumber of interactions above {confidence_threshold}: {len(filtered_data)}")
# Step 4: Create a NetworkX graph
graph = nx.Graph()

# Add edges (kinase interactions)
for _, row in filtered_data.iterrows():
    source = row["Source Protein"]
    target = row["Target Protein"]
    weight = float(row["Interaction Score"])
    graph.add_edge(source, target, weight=weight)
# Step 5: Visualize the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(graph)  # Layout for graph visualization
nx.draw(
    graph, 
    pos, 
    with_labels=True, 
    node_color="lightblue", 
    node_size=700, 
    edge_color="gray", 
    font_size=8
)
plt.title(f"Kinase Interaction Network (Confidence ≥ {confidence_threshold})", fontsize=14)
plt.show()
# Step 1: Calculate Centrality Measures
degree_centrality = nx.degree_centrality(graph)
betweenness_centrality = nx.betweenness_centrality(graph)
closeness_centrality = nx.closeness_centrality(graph)
# Step 2: Combine Results into a DataFrame
centrality_df = pd.DataFrame({
    "Node": list(degree_centrality.keys()),
    "Degree Centrality": list(degree_centrality.values()),
    "Betweenness Centrality": list(betweenness_centrality.values()),
    "Closeness Centrality": list(closeness_centrality.values())
})
# Sort by Degree Centrality for top influencers
centrality_df = centrality_df.sort_values(by="Degree Centrality", ascending=False)
print("\nTop 10 Nodes by Degree Centrality:")
print(centrality_df.head(10))
output_file = 'centrality_results.csv'  # Save in the current working directory
centrality_df.to_csv(output_file, index=False)
print(f"Centrality results saved to {output_file}")
!pip install python-louvain
# Fix: Generate node sizes to match the number of nodes
node_sizes = [800 + 200 * graph.degree(n) for n in graph.nodes]  # Updated to graph.nodes
colors = [partition.get(node, 0) for node in graph.nodes]  # Default color if no partition found

# Draw the graph
plt.figure(figsize=(16, 14))
nx.draw_networkx_nodes(
    graph,
    pos,
    node_size=node_sizes,
    node_color=colors,
    cmap=cm.get_cmap("viridis"),
    alpha=0.8
)
nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.5)
nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black", font_family="sans-serif")

# Add title and display
plt.title("Kinase Interaction Network with Communities", fontsize=16)
plt.axis("off")
plt.show()

# Step 3: Save Community Results
community_df = pd.DataFrame(list(partition.items()), columns=["Node", "Community"])
output_file = 'community_results.csv'
community_df.to_csv(output_file, index=False)
print(f"Community results saved to {output_file}")
# Install gprofiler-official for enrichment analysis
!pip install gprofiler-official
print(protein_list[:10])  # Check the first 10 entries
# Install mygene library
!pip install mygene
from gprofiler import GProfiler

# Step 1: Extract all proteins from the graph
protein_list = [node for node in graph.nodes if not node.startswith("Kinase")]

# Step 2: Perform Enrichment Analysis
gp = GProfiler(return_dataframe=True)
enrichment_results = gp.profile(organism="hsapiens", query=protein_list)

# Step 3: Save Enrichment Results
output_file = 'enrichment_results.csv'
enrichment_results.to_csv(output_file, index=False)
print(f"Enrichment results saved to {output_file}")

from mygene import MyGeneInfo

mg = MyGeneInfo()

# Ensure protein_list is defined properly
if "protein_list" not in locals():
    raise ValueError("protein_list is not defined. Ensure it contains valid Ensembl protein IDs.")

# Map Ensembl protein IDs to HGNC gene symbols
query_ids = list(protein_list)  # Convert to list if not already
results = mg.querymany(query_ids, scopes="ensembl.protein", fields="symbol", species="human", returnall=True)

# Extract mapped symbols, handling missing cases
mapped_proteins = [res["symbol"] for res in results["out"] if "symbol" in res]

# Identify missing mappings
unmapped_proteins = [res["query"] for res in results["out"] if "symbol" not in res]

# Print first 10 mapped symbols
print("Mapped Proteins (first 10):", mapped_proteins[:10])

# Print summary of unmapped queries
if unmapped_proteins:
    print(f"{len(unmapped_proteins)} input query terms found no hit.")
    print("Example missing mappings:", unmapped_proteins[:10])

from gprofiler import GProfiler

# Ensure graph is defined
if "graph" not in locals():
    raise ValueError("graph is not defined. Ensure it contains nodes representing proteins.")

# Step 1: Extract all proteins from the graph (excluding Kinase nodes)
protein_list = [node for node in graph.nodes if not node.startswith("Kinase")]

# Check if the protein list is not empty
if not protein_list:
    raise ValueError("No valid proteins found in the graph.")

# Step 2: Perform Enrichment Analysis
gp = GProfiler(return_dataframe=True)
try:
    enrichment_results = gp.profile(organism="hsapiens", query=protein_list)
except Exception as e:
    raise RuntimeError(f"Enrichment analysis failed: {e}")

# Step 3: Save Enrichment Results
output_file = "enrichment_results.csv"
enrichment_results.to_csv(output_file, index=False)
print(f"Enrichment results saved to {output_file}")
import pandas as pd

# Load the dataset
file_path = r"C:\Users\Tim\kinase_interactions.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
df.info(), df.head()
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.from_pandas_edgelist(df, source="Source Protein", target="Target Protein", edge_attr="Interaction Score", create_using=nx.DiGraph())

# Draw the network
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=False, node_size=50, edge_color="gray", alpha=0.7)
plt.title("Kinase Interaction Network")
plt.show()

# Basic network statistics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

num_nodes, num_edges
# Compute betweenness centrality
betweenness = nx.betweenness_centrality(G)

# Sort kinases by importance
top_kinases = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

# Display the top 10 kinases with highest betweenness centrality
top_kinases
from mygene import MyGeneInfo

mg = MyGeneInfo()
query_ids = ["9606.ENSP00000269305", "9606.ENSP00000264657", "9606.ENSP00000275493"]  # Add your IDs
results = mg.querymany(query_ids, scopes="ensembl.protein", fields="symbol", species="human")

# Extract symbols
mapped_kinases = {res["query"]: res.get("symbol", "Not Found") for res in results}
print(mapped_kinases)
import networkx as nx
import community as community_louvain

# Compute Louvain partition
partition = community_louvain.best_partition(G.to_undirected())

# Add community labels to nodes
nx.set_node_attributes(G, partition, "community")

# Count the number of detected clusters
num_clusters = len(set(partition.values()))
print(f"Number of detected functional clusters: {num_clusters}")
# Compute degree centrality (importance based on connections)
degree_centrality = nx.degree_centrality(G)

# Sort proteins by degree centrality
top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

# Display the top 10 most connected proteins
print("Top 10 Hub Proteins (Highly Connected Nodes):")
for protein, centrality in top_hubs:
    print(f"{protein}: {centrality:.4f}")
import matplotlib.pyplot as plt

# Get the top 10 hub proteins
top_hub_nodes = [node for node, _ in top_hubs]

# Draw network with highlighted hubs
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # Layout for visualization
nx.draw(G, pos, node_size=20, edge_color="gray", alpha=0.5, with_labels=False)

# Highlight top hub nodes
nx.draw_networkx_nodes(G, pos, nodelist=top_hub_nodes, node_color="red", node_size=100)

plt.title("Kinase Network with Highlighted Hub Proteins")
plt.show()
import pandas as pd

# Load kinase interactions
kinase_df = pd.read_csv(r"C:\Users\Tim\kinase_interactions.csv")

# Display first few rows
print(kinase_df.head())
# Load Excel file
top_20_df = pd.read_excel(r"C:\Users\Tim\OneDrive\Documents\Copy of TOP_20_FOR_STRUCTURE(1).xlsx")

# Display first few rows
print(top_20_df.head())

# Merge the two dataframes based on the 'Kinase' column
merged_data = pd.merge(interaction_data, fasta_data, on='Kinase', how='inner')

# Display the merged dataframe
print(merged_data.head())
