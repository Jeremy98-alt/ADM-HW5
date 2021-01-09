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