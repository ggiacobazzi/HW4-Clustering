# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import collections
import random
from itertools import islice


class node:
    
    def __init__(self, node_name):
        self.node_name = node_name
        self.adjacency_nodes = []
        self.community = self
        self.index = 0
        self.picked = False
       
    def change_community(self, node):
        self.community = node
        
    def get_adjacency_nodes(self):
        return self.adjacency_nodes
    
    def print_attrs(self):
        print("Node name: " + self.node_name)
        print("Len adj nodes: " +str(len(self.adjacency_nodes)))
        print("Node community: " + self.community.node_name)
    

# Create nodes list with their adj-nodes
def create_lists(graph):
    # Nodes list
    nodes_list = []

    count = 0
    for n in graph.nodes:
        new_node = node(n)
        new_node.index = count
        count +=1
        new_node.change_community(new_node)
        nodes_list.append(new_node)

    # Adj-nodes list
    for n in nodes_list:
        for adj_node in graph.adj[n.node_name]:
            n.adjacency_nodes.append(adj_node)
            
    return nodes_list
    

# Async community update method
def async_community_update(node_list):
    #print(len(node_list))
    for i in range(3):
        new_nodes_list = []
        changed = False
        changedComm = 0
        for n in node_list:
            # Get max community label from neighbors
            if len(n.get_adjacency_nodes()) > 0:
                # print("old comm: " + n.community.node_name)
                community = get_max_community(n.get_adjacency_nodes(), node_list)
                # print("new comm: " + community.node_name)
    
                # Check changed community status
                if community.node_name != n.community.node_name:
                    changed = True
                    changedComm +=1
                    
                # Create new node 
                new_node = node(n.node_name)
                new_node.adjacency_nodes = n.get_adjacency_nodes()
                new_node.change_community(community)
                new_node.index = n.index
                
                # print("----------------------")
                new_nodes_list.append(new_node)
        
        print("ChangedComm: " + str(changedComm))     
        print("----------------------")

        # Set new list to use for async update
        for n in new_nodes_list:
            node_list[n.index] = n
            nodes_list[n.index] = n
            
        if not changed:
            print_list(nodes_list)
            print("TERMINATED BEFORE -> index: " + str(i))
            return nodes_list
    return nodes_list


# Sync community update method
def sync_community_update(nodes_list):
    for i in range(3):
        changed = False
        changedComm = 0
        for n in nodes_list:
            if len(n.get_adjacency_nodes()) > 0:
                # Get max community label from neighbors
                
                # print("old comm: " + n.community.node_name)
                community = get_max_community(n.get_adjacency_nodes(), nodes_list)
                # print("new comm: " + community.node_name)
    
                # Check changed community status
                if community.node_name != n.community.node_name:
                    changed = True
                    changedComm += 1
                    
                # Create new node 
                new_node = node(n.node_name)
                new_node.adjacency_nodes = n.get_adjacency_nodes()
                new_node.change_community(community)
                new_node.index = n.index
                # print("----------------------")
                
                nodes_list[new_node.index] = new_node
        
        print("ChangedComm: " + str(changedComm))     
        print("----------------------")
        
        if not changed:
            # print_list(nodes_list)
            print("TERMINATED BEFORE -> index: " + str(i))
            return nodes_list
    # print_list(nodes_list)
    return nodes_list

    
# Hierarchical clustering method (big_graph attribute is used for chunking purposes)
def hierarchical_clustering(nodes_list, graph, big_graph = False):
    clusters_num = len(count_labels_dict(nodes_list, True).keys())
    print("*****")
    print("Clusters num before hierarchical clustering: " + str(clusters_num))
    
    if clusters_num == 1:
        print("No clusters to merge")
        return
    clusters = {}
    
    # Pick centroids
    print("Picking centroids")
    clusters_centroids = random.sample(nodes_list, clusters_num)
    for c in clusters_centroids:
        nodes_list[c.index].picked = True
        clusters[c.node_name] = []
    
    # Populate clusters
    print("Populating clusters")
    for n in nodes_list:
        if not n.picked:
            best_cluster = compute_dist(n, graph, clusters_centroids)
            n.picked = True
            clusters[best_cluster.node_name].append(n)
    
    # Check clusters (debug)
    # check_clusters(clusters)
            
    # If length of clusters > 1 start merging until we have only one left
    print("Starting merging phase")
    
    if big_graph:
        processes = 100
        chunk_size = int(len(clusters) / processes)
        cluster_chunks = chunks(clusters, chunk_size)
        for clus in cluster_chunks:
            clusters = clus
            break
    
    while len(clusters) > 1:
        distances = compute_dist_matrix(clusters, graph)
        clusters = merge_clusters(clusters, distances, nodes_list)
            
    print("Finished hierarchical clustering")
    print("*****")
            
    # Print final cluster
    clus_key = next(iter(clusters))
    print("Final cluster")
    print("Cluster key: " + clus_key)
    print("Nodes: ")
    print("*****")
    for n in clusters[clus_key]:
        print(n.node_name)
    print("*****")
    
    # clusters[key] is nodes_list minus 1 due to the cluster key missing
    print("Total nodes: " + str(len(clusters[clus_key])))
    print("Total clusters: " + str(len(clusters)))
    
    return
    

# Calculate shortest path of a node to a certain centroid to add it to its cluster
def compute_dist(node, graph, clusters_centroids):
    
    distances = []
    for centroid in clusters_centroids:        
        try:
            distance = int(nx.algorithms.shortest_paths.dijkstra_path_length
            (graph, node.node_name, 
              centroid.node_name))
        except nx.NetworkXNoPath:
            distance = 99999999999999999
 
        distances.append(distance)

    index = distances.index(min(distances))  
    return clusters_centroids[index]
    
    
# Compute distance matrix of the clusters
def compute_dist_matrix(clusters, graph):
    distances = {}
    count = 1
    
    
    for cluster in clusters:
        cluster_dist_matrix = {}
        print("Computing matr dist of " + cluster + " " + str(count) + "/" + str(len(clusters)))
        for clus in clusters:
            if cluster is not clus:
                try:
                    source_node, target_node = get_nodes(graph, cluster, clus)
                    distance = int(nx.algorithms.shortest_paths.dijkstra_path_length
                    (graph, source_node, 
                      target_node))
                except nx.NetworkXNoPath:
                    distance = 99999999999999999
                    
                cluster_dist_matrix[clus] = distance 
        # print(cluster_dist_matrix)
        distances[cluster] = cluster_dist_matrix
        print("Finished computing")
        count +=1
    
    return distances
                

# Get source and target node to compute distance
def get_nodes(graph, source_node, target_node):
    for n in graph.nodes:
        if n is source_node:
            source = n
        elif n is target_node:
            target = n
            
    return source, target
    

# Merge clusters based on their distance
def merge_clusters(clusters, distance_matrix, nodes_list):
    minimum_dist = 99999999999999999
    clusters_node_mins = []
    
    source = 0
    target = 0
    
    # Find best clusters to merge
    for cluster in clusters:
        # print(cluster)
        for clus in distance_matrix[cluster]:
            # print(clus)
            if cluster is not clus:
                if distance_matrix[cluster][clus] < minimum_dist:
                    minimum_dist = distance_matrix[cluster][clus]
                    clusters_node_mins = []
                    clusters_node_mins.append(clus)
                    source = cluster
                elif distance_matrix[cluster][clus] == minimum_dist:
                    source = cluster
                    minimum_dist = distance_matrix[cluster][clus]
                    clusters_node_mins.append(clus)
                    
    # Select target cluster to merge
    if len(clusters_node_mins) > 1:
        target = random.choice(clusters_node_mins)
    else:
        target = clusters_node_mins[0]
    
    # Don't merge
    if source is target:
        return clusters
    
    print("Merging " + source + " " + target)
    print("Source nodes: " + str(len(clusters[source])))
    print("Target nodes: " + str(len(clusters[target])))
    
    # Merge lists
    clusters[source] += clusters[target]
    clusters[source].append(get_elem_from_list(nodes_list, target))
    
    print("Merged cluster len: " + str(len(clusters[source])))
    
    # Remove key from dict
    del clusters[target]
    
    return clusters
    

def create_clusters_list(nodes_list):
    clusters = {}
    
    for n in nodes_list:
        if n.community.node_name in clusters:
            clusters[n.community.node_name].append(n)
        else:
            clusters[n.community.node_name] = []
            clusters[n.community.node_name].append(n)
            
    return clusters
    

# Get max community of labels
def get_max_community(adj_nodes_list, node_list):
    adj_nodes_list = [x for y in adj_nodes_list for x in node_list
                      if x.node_name == y]

    dic = count_labels_dict(adj_nodes_list)
    max_a = max(dic.values())
    maximums = [key for key, value in dic.items() if value == max_a]

    if len(maximums) > 1:
        rand_choice = random.choice(maximums)
        #print(rand_choice)
        return get_elem_from_list(node_list, rand_choice)
    #print(maximums[0])
    
    return get_elem_from_list(node_list, maximums[0])


# Count labels 
def count_labels_dict(node_list, clusters=False):
    label_dict = {}
    
    for n in node_list:
        if n.community.node_name in label_dict:
            label_dict[n.community.node_name] += 1
        else:
            label_dict[n.community.node_name] = 1
    if not clusters:
        return collections.Counter(label_dict)
    
    return label_dict
    
    
# Print complete list if it has more than x elements (e.g: 5)
def print_list(nodes_list):
    for node in nodes_list:
        if len(node.adjacency_nodes) > 5:
            node.print_attrs()
            print("------------------------------------")
        
        
# Return a node element from a node list
def get_elem_from_list(nodes_list, node_name):
    for x in nodes_list:
        if x.node_name == node_name:
            return x


# Separate data into chunks of equal size
def chunks(data, SIZE=1000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}
        

# Debug: check clusters info
def check_clusters(clusters):
    count = 0
    print(len(clusters.keys()))
    for key in clusters.keys():
        if len(clusters[key]) > 0:
            print("Key: " + key)

            count +=1
            for obj in clusters[key]:
                print(obj.node_name)
            print("-------------------------------")
    print("Clusters populated: " + str(count))   
    

# Setup graph choosing method
def graph_setup(graph_name, algorithm_type=True):
    if graph_name == "ants":
        graph = nx.read_graphml('ant_mersch_col1_day04_attribute.graphml')
        nx.draw(graph)  
    elif graph_name == "erdos":
        graph = nx.read_pajek('ERDOS972.NET')
        nx.draw_circular(graph)  
        
    nodes_list = create_lists(graph)
    return nodes_list, graph


# Community update choosing method
def community_update(nodes_list, algorithm_type=True):
    if algorithm_type:
        nodes_list = async_community_update(nodes_list)
    else:
        nodes_list = sync_community_update(nodes_list)
    

# Hierarchical clustering choosing method
def hierarchical_clus(graph_name, nodes_list, graph):
    if hierarchical_clus:
        if graph_name == "erdos":
            # The boolean attribute is used for chunking purposes
            hierarchical_clustering(nodes_list, graph, True)
        else:
            hierarchical_clustering(nodes_list, graph)
    
    
    
if __name__ ==  '__main__':
    graph_name = "erdos"
    algo_type = True
    nodes_list, graph = graph_setup(graph_name)
    plt.show()
    
    # First assignment
    community_update(nodes_list, algo_type)
    
    # Second assignment
    hierarchical_clus(graph_name, nodes_list, graph)

 

#Inputs for ant colony social interactions are in graphml format
#You can find the entire dataset at https://github.com/bansallab/asnr/tree/master/Networks/Insecta/ants_proximity_weighted
#Everyone get a different ant dataset from colony 1, days 1 to 5
#Example for reading the dataset
#Treat the network as undirected

########################################

#The second input graph is the truncated collaboration network for Paul Erdős in the year 1997
#In this case, everyone should work on the same dataset
#Truncated means that it does not inlcude Erdős himself
#Because of the above fact, there will be an extremeley small number of edges, so try to determine your communities based on this fact
#Full datastet at: http://vlado.fmf.uni-lj.si/pub/networks/pajek/data/gphs.htm
#If you are interested in a version where Erdős is included, check Erdos02.net at: http://vlado.fmf.uni-lj.si/pub/networks/data/default.htm
#Even the truncated graph is quite large, so processing times can be long as well. Try tuning your algorithms to be efficient for this size.
#For testing on a smaller graph, you can check ERDOS971.NET

   

