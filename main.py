# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:06:00 2020
@author: Ahmad Al Musawi
We want to compare the best model 

Undirected!
"""

import datetime
import networkx as nx
import time
import random

def jaccard(A, B): 
    """ Return similarity between A, B in undirected network"""
    bast = len(set(A).intersection(set(B)))
    makam = len(set(A).union(set(B)))
    if makam !=0 :
        return bast/makam
    else:
        return 0
    
def jaccardd(a,b, network):
    Ain = list(network.predecessors(a))
    Aout = list(network.successors(a))
    Bin = list(network.predecessors(b))
    Bout = list(network.successors(b))
    """ Return similarity between A, B in directed network"""
    if a in Bin: Bin.remove(a)
    if a in Bout: Bout.remove(a)
    if b in Ain: Ain.remove(b)
    if b in Aout: Aout.remove(b)    
    lin1 = len(set(Ain).intersection(set(Bin)))
    lin2 = len(set(Ain).union(set(Bin)))
    lout1 = len(set(Aout).intersection(set(Bout)))
    lout2 = len(set(Aout).union(set(Bout)))
    in_result=0
    out_result=0
    total=0
    if lin2!=0: 
        in_result = lin1/lin2*1.0
    if lout2!=0:
        out_result = lout1/lout2*1.0        
    if (lin2 + lout2) !=0: 
        total = (lin1 + lout1)/(lin2 + lout2)*1.0
    return total

 


def compact(network):
    print("Intrinsic Compacting ....")
    # print("Compacting Network....")
    #print(info(network))
    Nodes = []
    deleted = []
    newNetwork = network
    for i in network.nodes:
        Nodes.append(i)
    #print("Network: ", Nodes)
    for i in range(len(Nodes)-1):
        if Nodes[i] not in deleted:
            for j in range(i+1, len(Nodes),1):
                if Nodes[j] not in deleted:
                    Ni = network.neighbors(Nodes[i])
                    Nj = network.neighbors(Nodes[j])
                    if jaccard(Ni, Nj)==1:
                        deleted.append(Nodes[j])
    newNetwork.remove_nodes_from(deleted)
    return newNetwork

def compact_d(network):
    print("Intrinsic Compacting ....")
    # print("Compacting Network....")
    #print(info(network))
    Nodes = []
    deleted = []
    newNetwork = network
    for i in network.nodes:
        Nodes.append(i)
    #print("Network: ", Nodes)
    for i in range(len(Nodes)-1):
        if Nodes[i] not in deleted:
            for j in range(i+1, len(Nodes),1):
                if Nodes[j] not in deleted:
                    if jaccardd(Nodes[i], Nodes[j], network)==1:
                        deleted.append(Nodes[j])
    newNetwork.remove_nodes_from(deleted)
    return newNetwork



def av(D):
    s = sum(D.values())
    if len(D)!=0 :
        return s/len(D)
    else:
        return 0

def convert_time(ss):
    return datetime.datetime.fromtimestamp(ss).isoformat()
def diff_month(d1, d2):
    y1 = int(d1.split('-')[0])
    m1 = int(d1.split('-')[1])
    y2 = int(d2.split('-')[0])
    m2 = int(d2.split('-')[1])
    return (y1 - y2) * 12 + m1 - m2

def read_file(i):
    """Returning Temporal Network list of edges and timestamps Array"""
    path = 'D:\Documents\Research Projects\Complex Networks Researches\Compaction Model\python codes\Dataset'
    datasets = ["\email-dnc\email-dnc.txt",
                "\CollegeMsg\CollegeMsg.csv", 
                "\soc-sign-bitcoinotc\soc-sign-bitcoinotc.txt"]
    separater = [',', ',','\t']
    print(datasets[i])
    newPath = path + datasets[i]
    lines = open(newPath, "r", encoding=('utf-8')).readlines()
    edges = []
    for line in lines: 
        line = line.strip('\ufeff')
        line = line.strip('\n')
        nline = line.split(separater[i]) # could be ("\t")
        newDate = convert_time(int(nline[2])).split('T')[0]
        #print(newDate)
        nline2 = [int(nline[0]), int(nline[1]), newDate]
        edges.append(nline2)
    Edges = sorted(edges, key=lambda x: x[2])
    return Edges

def info(G):
    return len(G.nodes()), len(G.edges())

def main(bb):
    #part One, Read from text file and convert to array of networks
    edges = read_file(bb)
    first_date = edges[0][2]
    last_date = edges[len(edges)-1][2]
    print(first_date, last_date)
    print(diff_month(last_date, first_date))
    y1 = int(first_date.split('-')[0])
    m1 = int(first_date.split('-')[1])
    d1 = int(first_date.split('-')[2])
    y2 = int(last_date.split('-')[0])
    d2 = int(last_date.split('-')[2])
    
    networks = []
    print("-------------Months------------")
    for i in range(diff_month(last_date, first_date)+1):
        print(i, y1, m1)
        m1 = m1 + 1
        if m1==13:
            m1 = 1
            y1 = y1 + 1
        G = nx.DiGraph()
        for e in edges:
            if (int(e[2].split('-')[0]) == y1) & (int(e[2].split('-')[1]) == m1):
                G.add_node(e[0])
                G.add_node(e[1])
                G.add_edge(e[0], e[1])
        networks.append([G, y1, m1])

    print("-------------------------")
    Networks = []
    for N in networks:
        print(info(N[0]))
        Networks.append(N[0])
    print("------------Dynamic-------------")
    dynamic(Networks)
    print("------------end Dynamic-------------")
    
    
def dynamic(D):
    A = []
    A.append(D[0])
    results = ""
    inc = compact_d(A[0].copy())
    for i in range(1, len(D)):
        (x1,y1) = info(A[i-1])
        (x2,y2) = info(D[i])
        B = nx.DiGraph()
        B.add_edges_from(A[i-1].edges())
        B.add_edges_from(D[i].edges())
        A.append(B)
        (x3,y3) = info(A[i])
        print(x1,y1,':', x2,y2,':',x3,y3)
        b, xx, inc2 = directed_compare(A[i-1].copy(), D[i].copy(), inc)
        inc = inc2
        print(i-1,'\t', b,'\t', xx,'\n')
        results = results + '\n {} \t {} \t {}'.format(i-1, b, xx)
    print(results)
    
def undirected_compare(G1, d2, inc1, c):
    newG2 = nx.Graph()
    newG2.add_edges_from(G1.edges())
    newG2.add_edges_from(d2.edges())
    
    oldInc = nx.Graph()
    start2 = time.time()
    centralities(newG2, c)
    end2 = time.time()
    b0 = end2 - start2
    
    oldInc = compact(newG2)
    
    start1 = time.time()
    newInc2, xx = Compact2(G1, inc1, d2)
    centralities(newInc2, c)
    end1 = time.time()
    b2 = end1 - start1

    print("{}/{}={}".format(b0, b2, speedup(b0, b2)))
    b2 = speedup(b0, b2)
    N2, E2 = info(newG2)
    n2, e2 = info(newInc2)
    print("---------------------------")
    print(N2, E2)
    print(n2, e2)
    print("---------------------------")
    return b2, xx, oldInc


def directed_compare(G1, d2, inc1, c):
    newG2 = nx.DiGraph()
    newG2.add_edges_from(G1.edges())
    newG2.add_edges_from(d2.edges())
    
    oldInc = nx.DiGraph()
    start2 = time.time()
    centralities(newG2, c)
    end2 = time.time()
    b0 = end2 - start2

    oldInc = compact_d(newG2)
    
    start1 = time.time()
    newInc2, xx = Compact2_d(G1, inc1, d2)
    centralities(newInc2, c)
    end1 = time.time()
    b2 = end1 - start1

    b2 = speedup(b0, b2)
    N2, E2 = info(newG2)
    n2, e2 = info(newInc2)
    print("---------------------------")
    print(N2, E2)
    print(n2, e2)
    print("---------------------------")
    return b2, xx, oldInc

    
    

def centralities(network, c):
    if c==1:
        nx.betweenness_centrality(network)
    if c==2:
        nx.density(network)
    if c==3:
        nx.average_clustering(network)
    if c==4:
        nx.closeness_centrality(network)
    if c==5:
        nx.eigenvector_centrality(network.to_undirected())
    if c==6:
        nx.katz_centrality(network)
    if c==7:
        nx.pagerank(network)
    if c==8:            
        nx.harmonic_centrality(network)
    if c==9:
        nx.betweenness_centrality(network)
        nx.density(network)
        nx.average_clustering(network)
        nx.closeness_centrality(network)
        nx.eigenvector_centrality(network.to_undirected())
        nx.katz_centrality(network)
        nx.pagerank(network)
        nx.harmonic_centrality(network)
        

def newMain():
    path = 'D:\Documents\Research Projects\Complex Networks Researches\Compaction Model\Intrinsic Model with approximating\Python Codes\\new Dataset\static\my example network'
    sample = path + '\sample.txt'
    g1 = nx.read_adjlist(sample, create_using = nx.Graph(), nodetype = int)

    c1 = nx.Graph()
    temp = nx.Graph()
    temp.add_edges_from(g1.edges())
    temp1 = compact_d(temp)
    c1.add_edges_from(temp1.edges())

    g2 = nx.Graph()
    g2.add_edges_from(g1.edges())
    g2.add_edges_from([(23, 10), (23, 15), (23, 16), (4,1), (4,2),
                       (20, 25), (20, 26), (20, 27)])
    result1 = nx.Graph()
    result1, xx = Compact1(g1, c1, g2)

    result2 = nx.Graph()
    result2 = Compact2(g1, c1, g2)

    result3 = nx.Graph()
    result3 = Compact3(g1, c1, g2)
    
    r2 = nx.Graph()
    r2 = compact_d(g2)
    print("-------------------------------")
    print(info(r2))
    print(info(result1))
    print(info(result2))
    print(info(result3))


def newMain2():
    # U = input('u: undirected; *: directed')
    path = 'D:\Documents\Research Projects\Complex Networks Researches\Compaction Model\Intrinsic Model with approximating\Python Codes\Dataset\Static'
    sample = path + '\my example network\sample.txt'
    
    spath = "D:\Documents\Research Projects\Complex Networks Researches\Compaction Model\Intrinsic Model with approximating\Python Codes\\new Dataset"
    datasets = ["\\ba_1k_2k\\ba_1k_2k.txt",
                "\\bio-DM-LC\\bio-DM-LC.txt",
                "\\bio-diseasome\\bio-diseasome.txt",
                "\sociopatterns-hypertext\sociopatterns-hypertext.txt",
                "\\ba_1k_40k\\ba_1k_40k.txt",
                "\er_graph_1k_4k\er_graph_1k_4k.txt",
                "\er_graph_1k_6k\er_graph_1k_6k.txt",                   
                '\Ecoli\Ecoli.txt',
                '\\rt_assad\\rt_assad.txt',
                "\\bio-CE-LC\\bio-CE-LC.txt",
                "\\bio-yeast\\bio-yeast.txt",
                "\ca-CSphd\ca-CSphd.mtx",
                "\ca-GrQc\ca-GrQc.mtx",
                "\mammalia-voles-kcs-trapping\mammalia-voles-kcs-trapping.txt",
                "\socfb-Reed98\socfb-Reed98.mtx",
                "\socfb-Simmons81\socfb-Simmons81.mtx",
                "\\bio-CE-PG\\bio-CE-PG.txt",
                "\socfb-Haverford76\socfb-Haverford76.mtx",
                "\\bio-CE-CX\\bio-CE-CX.txt"]
    
    types = ['u', 'd']
    centrality = {"Betweenness":1,
                  "Density":2,
                  "Clustering":3,
                  "Closeness": 4,
                  "pageRank": 7,
                  "Harmonic": 8}
    selectedG = {"pa-1":0, 
                 'bio-DM-LC':1,
                 "er-1":5, 
                 'sociopatterns-hypertext': 3, 
                 'mammalia-voles-kcs-trapping':13,
                 'Ca-CSPhd': 11
                     }
    print("Select network index")
    for i in selectedG:
        print(i, selectedG[i])
    g = int(input("Enter networks:  "))
    print ("you selected : ", datasets[g])
    ppp = input("Press Enter to start")
    
    total = "{} \t {} \t {} \t {} \t {} \t {} \n".format('G','C','Speedup','Affected','Speedup','Affected')
    for c in centrality:
        res = "{} \t {} \t ".format(g, c)
        for U in types:
            print('g: ',g,' U: ',U,' c: ',c)
            if U=='u':
                G = nx.read_adjlist(spath + datasets[int(g)], create_using = nx.Graph(), nodetype = int)
            else:
                G = nx.read_adjlist(spath + datasets[int(g)], create_using = nx.DiGraph(), nodetype = int)
            if c==0:
                for q in range(1, 50):
                    performance, affected, r = incremental_100(G.copy(), U, q, centrality[c])
                    print("{}- P = {} \t Aff = {}".format(q, sum(performance)/r, sum(affected)/r))
                    result = "{} \t {} ".format(sum(performance)/r, sum(affected)/r)
            else:
                q=1
                performance, affected, r = incremental_100(G.copy(), U, q, centrality[c])
                print("{}- P = {} \t Aff = {}".format(q, sum(performance)/r, sum(affected)/r))
                result = "{} \t {} ".format(sum(performance)/r, sum(affected)/r)
            res = res +"\t"+ result
        total = total + res +"\n"
    print(total)
        


def incremental_100(G, U, z, c):
    """
    G: network
    U = 0 undirected, U = 1 directed!
    z : increment size
    c: centrality
    """
    sampling = 50
    performance = []
    affected = []
    if U=='u':
        print("directed 100 random edges simulation")
        print(info(G))
        random_edges = random.sample(G.edges(), sampling)
        for e in random_edges:
            G.remove_edge(*e)
        G1= nx.Graph()
        G1.add_edges_from(G.edges())
        r = 0
        inc = nx.Graph()
        inc = compact(G1)
        for i in range(0, sampling, z):
            RE = retrieve(random_edges, i, z)
            d2 = nx.Graph()
            d2.add_edges_from(RE)
            b2, xx, oldInc = undirected_compare(G1, d2, inc, c)
            print("{} \t {} \n".format(r, b2))
            performance.append(b2)
            affected.append(xx)
            r = r + 1
            G1.add_edges_from(RE)
            inc = nx.Graph()
            inc = oldInc  
        return performance, affected, r

    else:
        print("directed 100 random edges simulation")
        print(info(G))
        random_edges = random.sample(G.edges(), sampling)
        for e in random_edges:
            G.remove_edge(*e)
        G1= nx.DiGraph()
        G1.add_edges_from(G.edges())
        r = 0
        inc = nx.DiGraph()
        inc = compact_d(G1)
        for i in range(0, sampling, z):
            RE = retrieve(random_edges, i, z)
            d2 = nx.DiGraph()
            d2.add_edges_from(RE)
            b2, xx, oldInc = directed_compare(G1, d2, inc, c)
            print("{} \t {} \n".format(r, b2))
            performance.append(b2)
            affected.append(xx)
            r = r + 1
            G1.add_edges_from(RE)
            inc = nx.DiGraph()
            inc = oldInc        
        return performance, affected, r


def retrieve(E, start_point, length):
    """
        E: List of edges
        n: number of edges reqyured
        s: start point
    """
    return (E[start_point: start_point+length])    
        
    

def Compact1(g1, c1, g2):
    # we must find diff in edges bet T = g2-g1
    # diff = list(set(g2.edges())-set(g1.edges()))
    diff = list(g2.edges())
    print("diff is !", len(diff))
    g2.add_edges_from(g1.edges())

    T = nx.Graph()
    T.add_edges_from(diff)
    
    CC2 = nx.Graph()
    CC2.add_edges_from(c1.edges())
    CC2.add_edges_from(diff)

    Diff = nx.Graph()
    for v in T.nodes():
        for u in g2.neighbors(v):
            if v not in Diff.nodes():
                Diff.add_node(u)
    # print("--------------------------------------")
    c2 = nx.Graph()
    c2, xx = NewIntrinsic(g2, Diff, CC2)
    return c2, xx

def Compact2(g1, c1, g2):
    # we must find diff in edges bet T = g2-g1
    diff = list(g2.edges())
    print("diff is !", len(diff))

    g2.add_edges_from(g1.edges())

    T = nx.Graph()
    T.add_edges_from(diff)
    # print(diff)
    
    CC2 = nx.Graph()
    CC2.add_edges_from(c1.edges())
    CC2.add_edges_from(diff)

    Diff = nx.Graph()
    for v in T.nodes():
        for u in g2.neighbors(v):
            if v not in Diff.nodes():
                Diff.add_node(u)
    # print("--------------------------------------")
    c2 = nx.Graph()
    c2,xx = NewIntrinsic2(g2, Diff, CC2)
    return c2, xx

def Compact2_d(g1, c1, d2):
    # we must find diff in edges bet T = g2-g1
    diff = list(d2.edges())
    print("diff is !", len(diff))

    T = nx.DiGraph()
    T.add_edges_from(diff)

    g2 = nx.DiGraph()
    g2.add_edges_from(g1.edges())
    g2.add_edges_from(d2.edges())

    CC2 = nx.DiGraph()
    CC2.add_edges_from(c1.edges())
    CC2.add_edges_from(diff)

    Diff = nx.DiGraph()
    for v in T.nodes():
        for u in list(g2.predecessors(v)) + list(g2.successors(v)):
            if v not in Diff.nodes():
                Diff.add_node(u)
    # print("--------------------------------------")
    c2 = nx.DiGraph()
    c2, xx = NewIntrinsic2_d(g2, Diff, CC2)
    return c2, xx

def Compact3(g1, c1, g2):
    # print("compacting 3....")
    # print("g1: ", info(g1))
    # print("g2: ", info(g2))
    
    # we must find diff in edges bet T = g2-g1
    # diff = list(set(g2.edges())-set(g1.edges()))
    diff = list(g2.edges())
    print("diff is !", len(diff))
    g2.add_edges_from(g1.edges())

    T = nx.Graph()
    T.add_edges_from(diff)
    # print(diff)
    
    CC2 = nx.Graph()
    CC2.add_edges_from(c1.edges())
    CC2.add_edges_from(diff)

    Diff = nx.Graph()
    for v in T.nodes():
        for u in g2.neighbors(v):
            if v not in Diff.nodes():
                Diff.add_node(u)
    # print("--------------------------------------")
    c2 = nx.Graph()
    c2 = NewIntrinsic3(g2, Diff, CC2)
    return c2
    
    

def NewIntrinsic(G2, D, CC2):
    """compacting Neighboors list only in G"""
    if len(G2.nodes())>0:
        XX = len(D.nodes())*100/len(G2.nodes())
    else:
        XX = 0

    Nodes = []
    deleted = []
    added = set()
    for nn in D.nodes():
        for v in G2.neighbors(nn):
            if v not in added:
                added.add(v)
    # comparisons = []
    avoided = []
    newNetwork = CC2
    x = 0
    y = 0
    comp = nx.Graph()
    for k in D.nodes():
        Nodes = list(G2.neighbors(k))
        for i in range(0, len(Nodes)-1):
            v = Nodes[i]
            if v not in deleted:
                for j in range(i+1, len(Nodes)):
                    u = Nodes[j]
                    if u not in deleted:
                        compared = False
                        if not comp.has_edge(u, v):                        
                            compared = True
                            comp.add_edge(u, v)
                            Nu = G2.neighbors(u)
                            Nv = G2.neighbors(v)
                            if jaccard(Nu, Nv)>= 1:
                                deleted.append(u)
                        if compared:
                            x = x + 1
                        else:
                            y = y + 1
                            avoided.append((u,v))
    print("Comparisons done: ", x)
    print("Comparisons passed: ", y)
    # print("new Network", newNetwork.nodes())
    # print("To be added", added)
    # print("To be deleted", set(deleted))
    newNetwork.remove_nodes_from(deleted)
    for i in deleted:
        if i in added:
            added.remove(i)
    for v in added:
        E=[]
        for e in G2.edges(v):
            if e[1] in added:
                E.append(e)
        newNetwork.add_edges_from(E)
    #print(info(newNetwork))
    return newNetwork, XX

def NewIntrinsic2(G, D, CC2):
    if len(G.nodes())>0:
        xx = len(D.nodes())*100/len(G.nodes())

    Nodes = list(D.nodes())
    deleted = []
    newNetwork = CC2
    
    for i in range(0, len(Nodes)-1):
        v = Nodes[i]
        if v not in deleted:
            for j in range(i+1, len(Nodes),1):
                u = Nodes[j]
                if u not in deleted:
                    Nu = G.neighbors(u)
                    Nv = G.neighbors(v)
                    if jaccard(Nu, Nv)>= 1:
                        deleted.append(u)
 
    newNetwork.remove_nodes_from(deleted)
    D.remove_nodes_from(deleted)

    for v in D.nodes():
        E=[]
        for e in G.edges(v):
            if e[1] in D.nodes():
                E.append(e)
        newNetwork.add_edges_from(E)

    return newNetwork, xx

def NewIntrinsic2_d(G, D, CC2):
    if len(G.nodes())>0:
        XX = len(D.nodes())*100/len(G.nodes())
    else:
        XX = 0
    Nodes = list(D.nodes())
    deleted = []
    newNetwork = CC2
    
    for i in range(0, len(Nodes)-1):
        v = Nodes[i]
        if v not in deleted:
            for j in range(i+1, len(Nodes),1):
                u = Nodes[j]
                if u not in deleted:
                    if jaccardd(u, v, G)>= 1:
                        deleted.append(u)
 
    newNetwork.remove_nodes_from(deleted)
    D.remove_nodes_from(deleted)

    for v in D.nodes():
        E=[]
        for e in G.edges(v):
            if e[1] in D.nodes():
                E.append(e)
        newNetwork.add_edges_from(E)

    return newNetwork, XX

def NewIntrinsic3(G, D, CC2):
    if len(G.nodes())>0:
        print("Compacting 3....", len(D.nodes())*100/len(G.nodes()))

    Nodes = list(D.nodes())
    deleted = []
    newNetwork = CC2
    
    for v in Nodes:
        NN = list(G.neighbors(v))
        for i in range(0, len(NN)-1):
            if NN[i] not in deleted:
                for j in range(i+1, len(NN)):
                    if NN[j] not in deleted:
                        v, u = NN[i], NN[j]
                        Nu = G.neighbors(u)
                        Nv = G.neighbors(v)
                        if jaccard(Nv, Nu)>= 1:
                            deleted.append(u)

    newNetwork.remove_nodes_from(deleted)
    D.remove_nodes_from(deleted)

    for v in D.nodes():
        E=[]
        for e in G.edges(v):
            if e[1] in D.nodes():
                E.append(e)
        newNetwork.add_edges_from(E)

    return newNetwork


def speedup(a, b):
    if b !=0 :
        return a/b
    else:
        return 0
    
    


