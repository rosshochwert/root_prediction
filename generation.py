import sys
import string
import os
import subprocess
import csv
from itertools import combinations_with_replacement, combinations
import networkx as nx
import numpy as np
import pandas as pd
import random
import os

random.seed(1234)

############################################
##### 0. Helper Command To Ensure Connected Graph
############################################

def convertEdgeList(edges):
	return ' '.join(str(x) for x in edges).replace('(', '').replace(')'
			, '').replace(',', '')


def createGraph(nodes, edges):
	G = nx.MultiGraph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	return G


def createWeightedGraph(g):
	G = nx.Graph()
	for u,v in g.edges():
		if G.has_edge(u,v):
			G[u][v]['weight'] += 1
		else:
			G.add_edge(u, v, weight=1)
		 
	return G

def checkValid(nodes, edges):
	g = createGraph(nodes, edges)
	if len(nodes) - 1 > len(edges) or len(nodes) <= 1:
		return False
	elif nx.is_tree(g):
		return False
	else:
		return nx.is_connected(g)


def pruneSolos(nodes, edges):
	g = createGraph(nodes, edges)
	solo_nodes = [node for (node, degree) in dict(g.degree).items() if degree == 1]
	[g.remove_node(node) for node in solo_nodes]
	return (g.nodes, [(a,b) for a,b,c in g.edges]) #<--just take the edge list (first two params). third param is tracker for how many times this edge has been seen in graph.

def checkTree(nodes,edges):
	g = createGraph(nodes, edges)
	solo_nodes = [node for (node, degree) in dict(g.degree).items() if degree == 1]
	return len(solo_nodes) > 0


def getGraphStats(nodes, edges):
	graph = createGraph(nodes, edges)
	determinant = checkDeterminant(nodes,edges)
	graph_degree = dict(graph.degree)
	nodes = len(graph.nodes)
	edges = len(convertEdgeList(edges).split()) / 2
	max_degree = np.max(list(graph_degree.values()))
	min_degree = np.min(list(graph_degree.values()))
	avg_degree = np.mean(list(graph_degree.values()))
	std_degree = np.std(list(graph_degree.values()))

	avg_distance = nx.average_shortest_path_length(graph)
	diameter = nx.algorithms.distance_measures.diameter(graph)
	betti = edges-nodes+1

	weighted = createWeightedGraph(graph)
	transitivity = nx.transitivity(weighted)
	cluscoef = nx.average_clustering(weighted)
	stats = [
		nodes,
		edges,
		max_degree,
		min_degree,
		avg_degree,
		avg_distance,
		std_degree,
		diameter,
		determinant,
		transitivity,
		cluscoef,
		betti
		]
	return stats

def checkDeterminant(nodes,edges):
	graph = createGraph(nodes, edges)
	adjacency = nx.to_scipy_sparse_array(graph).todense()
	return np.absolute(np.round(np.linalg.det(adjacency),1))

############################################
##### 0.1 Set up CSV file structure and params
############################################

total = []
headers = [
	'index',
	'edge list',
	'power',
	'root',
	'all_roots_found?',
	'nodes',
	'edges',
	'max_degree',
	'min_degree',
	'avg_degree',
	'avg_distance',
	'std_degree',
	'diameter',
	'determinant',
	'transitivity',
	'cluscoef',
	'betti',
	'count',
	'margin',
	'count_percent',
	'margin_percent',
	'i',
	]
total.append(headers)

file_name = '../rootsPowersKC'

import time
import glob
import re

start = time.time()

############################################
##### 0.2 Load Existing CSV
############################################

try:
	files = glob.glob("train*.csv")
	if len(files)>0:
		#check to see if train.csv is in set (which is final output)
		if "train.csv" in files:
			current_df = pd.read_csv("train.csv")
			print("Full database detected, loading that in")
		else:
			ints = [int(re.sub('[^A-Za-z0-9]+','',x.strip(string.ascii_letters))) for x in files]
			index, element = max(enumerate(ints), key=lambda x: x[1])
			current_df = pd.read_csv(files[index])
			print("Partial database detected, loading in " + str(element) + " graphs")
	else:
		print("No existing database, no problem")
		current_df = None
except Exception as e:
	print(e)
	print("Some sort of error loading database")



############################################
##### 1. Create Fake Polytopes Up to 5 Nodes
############################################

print('Generating Fake Polytopes')

###Explanation###
##
## edges creates a list of all possible edges, except not to self
## we then loop and create a list of polytopes with edges <= max edges (where max = # of nodes + 2 ).
## a polytope can of course have more edges than # of nodes, but we are capping to start
## we are looking for unique graphs, so we check aspects of each new graph and ensure it is new
##
#################

polytopes = []
stats_tracker = {}
graph_tracker = set()

for node in range(2, 6):
	nodes = [x for x in range(node)]
	edges = sorted(list(combinations(nodes, 2)))
	# edges = list(combinations_with_replacement(nodes, 2)) <--removes self loops

	for size in range(1, len(nodes) * 2):
		polys = sorted(list(combinations_with_replacement(edges, size)))  # <--with replacement allows multiple edges between same nodes
		for edge in polys:
			if checkValid(nodes, edge) & checkTree(nodes,edge):
				graph_stats = getGraphStats(nodes, edge)
				t_g_s = tuple(graph_stats)
				if t_g_s not in graph_tracker:
					key = tuple(edge)
					stats_tracker[key] = graph_stats
					polytopes.append(edge)
					graph_tracker.add(t_g_s)


polytopes.sort(key=len)


polytopes_convert = [str(convertEdgeList(polytopes[index])) for index in range(0,len(polytopes))]

print("Writing Results to CSV File")
with open('graphs.csv',"w",newline="") as f:
	writer = csv.writer(f)
	writer.writerows(polytopes_convert)


print("We have generated " + str(len(polytopes)) + " polytopes")
print("Finished generating graphs in " + str(round(time.time()-start,2)) + " seconds")

############################################
##### 2. Calculate Canoncial Results for Each Root
############################################
print("Calculating Canonical Results for Each Root")


for index in range(0,len(polytopes)):
	edge_list = convertEdgeList(polytopes[index])
	if current_df is None or len(current_df[current_df['edge list'] == edge_list])==0:
		##dealing with new entry, let's compute
		for power in range(1,7):
			key = tuple(polytopes[index])
			stats = stats_tracker[key]
			betti = stats[11]
			degree_bundle = int(power*(2*betti-2))
			for root in range(2,degree_bundle+1):
				if(degree_bundle%root==0):
					parameters = edge_list + " " + str(root) + " " + str(power)
					print(parameters)
					result = subprocess.run([file_name, parameters],stdout=subprocess.PIPE).stdout.decode('utf-8').split()
					if len(result)!=5:
						if(len(result)==0):
							#negative genus?
							print("Invalid argument. Skipping this entry in CSV file")
							pass;
						else:
							print(result)
							print("Unexpected output found, trying to write into CSV")
							#infinite solutions?
							row = [index, edge_list, power, root, False]
							row = row + stats + result[-3:]
							total.append(row)
					else:
						row = [index, edge_list, power, root, True]
						row = row + stats + result
						total.append(row)
	else:
		##old entry from before, no need to waste computer space on it
		rows = current_df[current_df['edge list'] == edge_list]
		for i, row in rows.iterrows():
			total.append(row.values.flatten().tolist())

	if (index%5==0):
		print("Writing results to CSV file")
		with open('train' + str(index) + '.csv', "w",newline="") as f:
			writer = csv.writer(f)
			writer.writerows(total)
		print("Passing through polytopes of index: " + str(index) + " out of " + str(len(polytopes)))
		print("Total minutes elapsed: " + str(round((time.time()-start)/60,1)))


######################################
##### 3. Write Results to Our File
######################################

print("Writing Results to CSV File")
with open('train.csv',"w",newline="") as f:
	writer = csv.writer(f)
	writer.writerows(total)


print("Total minutes elapsed (finished): " + str(round((time.time()-start)/60,1)))