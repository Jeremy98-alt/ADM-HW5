import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import networkx as nx


####### Load the datasets

def read_edges():
    edges = pd.read_csv("wikigraph_reduced.csv", sep="\t", usecols=["0", "1"])
    edges.columns = ["source", "dest"]  # rename the columns with source and dest
    return edges


def read_pages_name():
    pages_names = pd.read_csv("wiki-topcats-page-names.txt", names=["node name"])

    pages_names['node'] = [s.split()[0] for s in list(pages_names['node name'])]
    pages_names['name'] = [' '.join(s.split()[1:]) for s in list(pages_names['node name'])]

    pages_names = pages_names.set_index("node").drop('node name', axis=1)
    return pages_names


def read_categories():
    categories = pd.read_csv("wiki-topcats-categories.txt", sep=";", names=["Category", "Pages List"])
    categories["Category"] = categories.Category.apply(lambda x: x[9:])
    categories["Pages List"] = categories["Pages List"].apply(lambda x: x.split())

    # Remove categories with length not in the range 5000 and 30000
    categories['Length'] = [len(x) for x in categories['Pages List']]

    # Removes the categories whose number of articles in less than 5000 and more than 30000.
    categories = categories.loc[(categories['Length'] >= 5000) & (categories['Length'] <= 30000)]
    del categories['Length']  # remove the useless column

    # reset index
    categories = categories.reset_index()  # reset the indexes
    del categories['index']  # remove the useless column

    return categories


# Supports loading datasets

def create_vocabulary(categories):
    pages_list = list(categories["Pages List"])  # save in a list the nested list with all pages per category

    nodes = {}
    for i, list_pages in enumerate(pages_list):  # read each list and check if the page is in the vocabulary
        for node in list_pages:  # given a list read each node
            if node not in nodes:  # check if the node is in the vocabulary
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
        cat = random.sample(nodes[node], 1)[
            0]  # select randomly one category over the possibily category where the page is assigned
        if int(node) in red_nodes:
            nodes2[cat].append(node)
    return nodes2


def set_category(x, categories):
    return categories.loc[x, "Category"]  # return the name of the category in oder to change the name


def fix_categories(categories, edges, nodes):
    nodes2 = create_nodes2(nodes, edges)

    new_categories = pd.DataFrame()  # create the new dataset well pre-processed
    new_categories['category'] = nodes2.keys()
    new_categories['pages list'] = nodes2.values()

    new_categories["category"] = new_categories.category.apply(
        lambda x: set_category(x, categories))  # set the category not the integer value
    return new_categories


####### RQ1

def isDirected(edges):
    if len(set(list(edges['source'])).intersection(set(list(edges[
                                                                'dest'])))) > 0:  # check if the source node is also in the dest nodes, in order to check if the graph is directed or not
        return "directed"
    else:
        return "undirected"


def check_creation_graph(edges):
    if isDirected(edges) == "directed":
        G = nx.DiGraph(directed=True)  # the graph is directed
    else:
        G = nx.Graph()  # the graph is undirecteddd
    return G


def create_graph(edges, pages_names, G):
    # extract the edges list from the source and dest column from edges dataset
    sources = list(edges['source'])
    dests = list(edges['dest'])

    for i in range(len(edges)):  # add for each line of edges dataset the corresponding edge
        G.add_node(sources[i], name=pages_names.loc[str(sources[i]), 'name'])  # source node
        G.add_node(dests[i], name=pages_names.loc[str(dests[i]), 'name'])  # dest node

        G.add_edge(sources[i], dests[i],
                   weight=1)  # add edges with weight equal to one according to the professors requests
    return G


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    # save the nodes and then collect the corresponding frequencies in order to create the loglog histogram
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq = [in_degree.get(k, 0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq = [out_degree.get(k, 0) for k in nodes]
    else:
        degseq = [v for k, v in G.degree()]

    dmax = max(degseq) + 1
    freq = [0 for d in range(dmax)]
    for d in degseq:
        freq[d] += 1
    return freq


def plot_degree_distro(in_degree_freq, out_degree_freq):
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'ro-', label='in-degree')
    plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bv-', label='out-degree')
    plt.legend(['In-degree', 'Out-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Visualize the nodes degree distribution')


####### RQ2

def initialization(G):
    nodes = list(G.nodes())  # return the list of nodes
    dist = [float('inf') for x in range(
        len(nodes))]  # initialize the distance to inf in order to avoid the distance = 0, that means other thing
    visited = [False for x in range(len(nodes))]  # initialize the visited = False
    return nodes, dist, visited


def BFS(G, v, d):
    nodes, dist, visited = initialization(G)  # initialize the main parameters useful to do this question

    queue = []  # create queue for BFS

    # start BFS and insert the source node into the queue in order to start the algorithm
    queue.append(v)
    visited[nodes.index(v)] = True  # check if the nodes is visited or not
    dist[nodes.index(v)] = 0  # set the distance equal to 0.. is the first node explored!

    i = 0
    while queue and i <= d:  # check if the queue is empty and if we achieve the d clicks!
        s = queue.pop(0)

        for neigh in nx.neighbors(G, s):  # find any neighbors!
            if visited[nodes.index(neigh)] == False:
                visited[nodes.index(neigh)] = True
                dist[nodes.index(neigh)] = dist[nodes.index(s)] + 1
                queue.append(neigh)
        i += 1

    if i == d + 1:
        dist = list(filter(lambda a: a != float('inf'), dist))
    else:
        dist = float("inf")  # I can't arrive with d clicks into any pages...

    return dist, nodes

####### RQ3

def get_dict(new_categories):
    ctg = list(new_categories["category"])
    pg_lst = list(new_categories["pages list"])

    union_lst = [ [x, list(map(int,pg_lst[i]))] for i, x in enumerate(ctg) ]

    my_dict = {x[0]: x[1] for x in union_lst}
    return my_dict


def random_cat(my_dict):
    cat = random.sample(my_dict.keys(), 1)[0]
    return cat


def cat_subgraph_inf(my_dict,cat,G):
    new_cat = G.subgraph(my_dict[cat])
    print(f"the number of pages are {len(new_cat)} and the number of edges are {len(new_cat.edges)}")


def indegree_node(cat,G,my_dict):
    new_cat = G.subgraph(my_dict[cat])
    indegree_node = { node: new_cat.in_degree(node) for node in new_cat.nodes if new_cat.in_degree(node) != 0 } # is only to check the max value linked, i don't consider the node without edges setted with inf
    indegree_node = dict(sorted(indegree_node.items(), key=lambda item: item[1], reverse=True))
    return indegree_node


def most_centered(indegree_node):
    min_val = max(indegree_node.items(), key=lambda x: x[1])
    source = min_val[0]
    print(f"the node {min_val[0]} is the most centered with {min_val[1]} edges")
    return source


def dict_all_bfs(my_dict,cat):
    dict_allbfs = {k: {x: float("inf") for x in my_dict[cat]} for k in my_dict[cat]}
    return dict_allbfs


def different_bfs(G, v):
    queue = []  # create queue for BFS

    # start BFS
    queue.append(v)
    distances = {}
    distances[v] = 0

    while queue:
        s = queue.pop(0)

        for neigh in nx.neighbors(G, s):
            if neigh not in distances:
                distances[neigh] = distances[s] + 1
                queue.append(neigh)

    return distances


def calculate_bfs(dict_allbfs,G,cat,my_dict):
    new_cat = G.subgraph(my_dict[cat])
    for node in tqdm(dict_allbfs):
        distances = different_bfs(new_cat, node)
        for val in distances:
            dict_allbfs[node][val] = distances[val]
    return dict_allbfs


def create_directed_weighted(dict_allbfs,pages_names):
    G_cat_W = nx.DiGraph(directed=True)
    for node in dict_allbfs:
        G_cat_W.add_node(node, name = pages_names.loc[str(node), 'name'])
        for edge in dict_allbfs[node]:
            if (dict_allbfs[node][edge] != float("inf")) and (dict_allbfs[node][edge] != 0):
                G_cat_W.add_edge(node, edge, weight=dict_allbfs[node][edge])
    return G_cat_W


def nearest_neigh(G, v):
    queue = []
    queue.append(v)

    explored = {}
    explored[v] = 0
    while queue:
        s = queue.pop(0)

        near_dists = []
        for neigh in nx.neighbors(G, s):
            near_dists.append([neigh, G[s][neigh]['weight']])

        if near_dists:
            near_neigh = sorted(near_dists, key=lambda x: x[1])[0][0]
            if near_neigh not in explored:
                explored[neigh] = G[s][neigh]['weight']
                queue.append(near_neigh)

    if len(explored) != len(G.nodes):
        return "we can't achieve all pages, so.. NOT POSSIBLE!"
    else:
        return f"we need a minimum of {sum(explored.values())} steps to achieve all pages"

####### RQ4

class Graph:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.org_graph = [i[:] for i in graph]
        self.ROW = len(graph)
        self.COL = len(graph[0])

    def BFS(self, s, t,
            parent):  # Returns true if there is a path from source 's' to sink 't' in residual graph. Also fills parent[] to store the path
        visited = [False] * (self.ROW)
        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            u = queue.pop(0)
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
        return True if visited[t] else False

    # Function for Depth first search Traversal of the graph
    def dfs(self, graph, s, visited):
        visited[s] = True
        for i in range(len(graph)):
            if graph[s][i] > 0 and not visited[i]:
                self.dfs(graph, i, visited)

                # Returns the min-cut of the given graph

    def minCut(self, source, sink):
        parent = [-1] * (self.ROW)
        max_flow = 0
        while self.BFS(source, sink, parent):
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            max_flow += path_flow
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
        s = 0
        visited = len(self.graph) * [False]
        self.dfs(self.graph, s, visited)
        counter = 0
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.graph[i][j] == 0 and self.org_graph[i][j] > 0 and visited[i]:
                    counter += 1
        return counter
def choose_two_categories(new_categories,G):
    c1 = str(input('Choose the first category: ')).strip()
    c1_flag = False
    while c1_flag == False:
        c1_nodes = list(new_categories[new_categories.category == c1]["pages list"])[0]
        c1_nodes = [int(node) for node in c1_nodes]
        c1_subgraph = G.subgraph(c1_nodes)
        if list(c1_subgraph.edges) == []:
            print("The chosen category does not exist in the graph")
            input("Choose a new category: ").strip()
        else:
            c1_flag = True
    c2 = str(input('Choose the second category: ')).strip()
    c2_flag = False
    while c2_flag == False:
        c2_nodes = list(new_categories[new_categories.category == c2]["pages list"])[0]
        c2_nodes = [int(node) for node in c2_nodes]
        c2_subgraph = G.subgraph(c2_nodes)
        if list(c2_subgraph.edges) == []:
            print("The chosen category does not exist in the graph")
            input("Choose a new category: ").strip()
        else:
            c2_flag = True
    return c1,c2
def combine_subg(G,c1,c2,new_categories):
    c1_nodes = list(new_categories[new_categories.category == c1]["pages list"])[0]
    c1_nodes = [int(node) for node in c1_nodes]
    c2_nodes = list(new_categories[new_categories.category == c2]["pages list"])[0]
    c2_nodes = [int(node) for node in c2_nodes]
    subgraph_nodes = list(set(c1_nodes) | set(c2_nodes))
    subg = G.subgraph(subgraph_nodes)
    return subg
def random_nodes_implement(subg):
    subg_adj = nx.adjacency_matrix(subg)
    subg_graph = Graph(list(subg_adj.toarray()))
    u, v = random.sample(range(np.shape(nx.adjacency_matrix(subg))[0]), 2)
    return subg_graph.minCut(u, v)

    ####### RQ5

####### RQ6