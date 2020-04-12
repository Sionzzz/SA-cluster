from __future__ import division
import networkx as nx

def load_graph(path):
    G = nx.Graph()
    with open(path) as text:
        for line in text:
            vertices = line.strip().split(" ")
            # print(vertices[0],vertices[1])
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    n=G.number_of_nodes()
    Degree=0
    Max_func = 0
    for index in G.nodes:
        Degree += G.degree(index)
    average_degree = Degree/n
    print("The average degree of the network:" +str(average_degree))
    g = 1 / (average_degree*average_degree)
    for index in set(G.edges):
        jaccard = len(set(G.neighbors(index[0])) & set(G.neighbors(index[1]))) / (len(set(G.neighbors(index[0])) | set(G.neighbors(index[1])))+1)
        jaccard_distance = 1 - jaccard
        func = g*G.degree(index[0]) * G.degree(index[1]) / jaccard_distance
        if func > Max_func:
            Max_func = func
        G.remove_edge(index[0],index[1])
        G.add_edge(index[0], index[1], weight=func)
    for index in set(G.edges):
            G[index[0]][index[1]]['weight'] /= Max_func
    return G

if __name__ == "__main__":
    G = load_graph("F:/CommunityDetection/Code/Python/CommunityDetection/network/LFR data/network1k_0.1.txt")
    print("The number of nodes:",len(G.nodes(False)))
    print("The number of edges:",len(G.edges(None, False)))
