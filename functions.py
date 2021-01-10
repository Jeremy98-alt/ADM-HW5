import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import networkx as nx

####### Load the datasets

def read_edges():
    edges = pd.read_csv("wikigraph_reduced.csv", sep="\t", usecols = ["0", "1"])
    edges.columns = ["source", "dest"] # rename the columns with source and dest
    return edges

def read_pages_name():
    pages_names = pd.read_csv("wiki-topcats-page-names.txt", names=["node name"])

    pages_names['node'] = [s.split()[0] for s in list(pages_names['node name']) ]
    pages_names['name'] = [' '.join(s.split()[1:]) for s in list(pages_names['node name'])]

    pages_names = pages_names.set_index("node").drop('node name', axis=1)
    return pages_names

def read_categories():
    categories = pd.read_csv("wiki-topcats-categories.txt", sep=";", names = ["Category", "Pages List"])
    categories["Category"] = categories.Category.apply(lambda x: x[9:])
    categories["Pages List"] = categories["Pages List"].apply(lambda x: x.split())

    # Remove categories with length not in the range 5000 and 30000
    categories['Length'] = [len(x) for x in categories['Pages List']]

    # Removes the categories whose number of articles in less than 5000 and more than 30000.
    categories = categories.loc[(categories['Length']>=5000) & (categories['Length']<=30000)]
    del categories['Length'] # remove the useless column

    # reset index
    categories = categories.reset_index() # reset the indexes
    del categories['index'] # remove the useless column
    
    return categories

# Supports loading datasets

def create_vocabulary(categories):
    pages_list = list(categories["Pages List"]) # save in a list the nested list with all pages per category

    nodes = {}
    for i, list_pages in enumerate(pages_list): # read each list and check if the page is in the vocabulary
        for node in list_pages: # given a list read each node
            if node not in nodes: # check if the node is in the vocabulary
                nodes[node] = []
            nodes[node].append(i)
    
    return nodes

def red_nodez(edges):
    sources = set(edges["source"])
    dests = set(edges["dest"])
    red_nodes = sources.union(dests)

    return red_nodes

def create_nodes2(nodes, edges):
    red_nodes = red_nodez(edges)

    nodes2 = defaultdict(list)
    for node in nodes.keys():
        cat = random.sample(nodes[node], 1)[0] # select randomly one category over the possibily category where the page is assigned
        if int(node) in red_nodes:
            nodes2[cat].append(node)  
    return nodes2

def set_category(x, categories):
    return categories.loc[x, "Category"] # return the name of the category in oder to change the name

def fix_categories(categories, edges, nodes):
    nodes2 = create_nodes2(nodes, edges)

    new_categories = pd.DataFrame() # create the new dataset well pre-processed
    new_categories['category'] = nodes2.keys()
    new_categories['pages list'] = nodes2.values()
        
    new_categories["category"] = new_categories.category.apply(lambda x: set_category(x, categories)) # set the category not the integer value
    return new_categories

####### RQ1

def isDirected(edges):
    if len(set(list(edges['source'])).intersection(set(list(edges['dest'])))) > 0: # check if the source node is also in the dest nodes, in order to check if the graph is directed or not
        return "directed"
    else:
        return "undirected"

def check_creation_graph(edges):
    if isDirected(edges) == "directed":
        G = nx.DiGraph(directed=True) # the graph is directed
    else:
        G = nx.Graph() # the graph is undirecteddd
    return G

def create_graph(edges, pages_names, G):
    #extract the edges list from the source and dest column from edges dataset
    sources = list(edges['source'])
    dests = list(edges['dest'])

    for i in range(len(edges)): # add for each line of edges dataset the corresponding edge
        G.add_node(sources[i], name = pages_names.loc[str(sources[i]), 'name']) # source node
        G.add_node(dests[i], name = pages_names.loc[str(dests[i]), 'name']) # dest node
        
        G.add_edge(sources[i], dests[i], weight=1) # add edges with weight equal to one according to the professors requests
    return G

def degree_histogram_directed(G, in_degree=False, out_degree=False):
    # save the nodes and then collect the corresponding frequencies in order to create the loglog histogram
    nodes = G.nodes() 
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]

    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq

def plot_degree_distro(in_degree_freq, out_degree_freq):
    plt.figure(figsize=(12, 8)) 
    plt.grid(True)
    plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'ro-', label='in-degree') 
    plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bv-', label='out-degree')
    plt.legend(['In-degree','Out-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Visualize the nodes degree distribution')    

####### RQ2

def initialization(G):
    nodes = list(G.nodes()) # return the list of nodes
    dist = [float('inf') for x in range(len(nodes))] # initialize the distance to inf in order to avoid the distance = 0, that means other thing
    visited = [False for x in range(len(nodes))] # initialize the visited = False
    return nodes, dist, visited

def BFS(G, v, d):
    nodes, dist, visited = initialization(G) # initialize the main parameters useful to do this question
    
    queue = [] # create queue for BFS
    
    # start BFS and insert the source node into the queue in order to start the algorithm
    queue.append(v)
    visited[nodes.index(v)] = True # check if the nodes is visited or not
    dist[nodes.index(v)] = 0 # set the distance equal to 0.. is the first node explored!
    
    i = 0
    while queue and i <= d: # check if the queue is empty and if we achieve the d clicks!
        s = queue.pop(0)
        
        for neigh in nx.neighbors(G, s): # find any neighbors!
            if visited[nodes.index(neigh)] == False:
                visited[nodes.index(neigh)] = True
                dist[nodes.index(neigh)] = dist[nodes.index(s)] + 1
                queue.append(neigh)
        i+=1

    if i == d+1:
        dist = list(filter(lambda a: a != float('inf'), dist))
    else:
        dist = float("inf") # I can't arrive with d clicks into any pages...
        
    return dist, nodes

####### RQ3

####### RQ4

####### RQ5

####### RQ6