import sys
import time
import math
import torch
import copy
import itertools
import random
import struct
import imghdr
import torch.nn.functional as F
from  IPython import embed
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from PIL import ImageFile
import json
import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage import measure

CONFIG_center_K = 2

def get_center_list(cand_list) :
    global CONFIG_center_K
    center_list = []
    for i in range(CONFIG_center_K) :
        x = random.randint(0, len(cand_list) - 1)
        center_list.append(cand_list[x])
    return center_list

def rgb_distance(node_1, node_2, cvImg) :
    c_1 = cvImg[node_1[0], node_1[1], :]
    c_2 = cvImg[node_2[0], node_2[1], :]
    #print(np.sum(((c_1 - c_2) ** 2) ** 0.5))
    return np.sum(((c_1 - c_2) ** 2) ** 0.5)

def bfs_tree(adj_list, center_list, cvImg = None) :

    #print("bfsing tree...\n")
    vis = np.zeros(1005)
    node_dist = {}
    seq = []
    for node in center_list :
        node_dist[node['num']] = 0
        seq.append(node['num'])
        vis[node['num']] = 1
        
    while (len(seq) > 0) :
        node = seq[0]
        for neighbour in adj_list[node] :
            if vis[neighbour] == 0 and neighbour in adj_list.keys():
                #node_dist[neighbour] = node_dist[node] + rgb_distance(node, neighbour, cvImg)
                node_dist[neighbour] = node_dist[node] + 1
                seq.append(neighbour)
                vis[neighbour] = 1
        seq.pop(0)
    return node_dist

def Euclidean_distance(node_1, node_2) :
    dx2 = (node_1[0] - node_2[0]) ** 2
    dy2 = (node_1[1] - node_2[1]) ** 2
    return (dx2 + dy2) ** 0.5

def Euclidean_distance_weight(cand_list, center_list) :
    node_dist = {}
    for node in cand_list :
        node_dist[node['num']] = 10000000
        for center in center_list :
            node_dist[node['num']] = min(node_dist[node['num']], Euclidean_distance(node['pos'], center['pos']))
    return node_dist

def update_weight(cand_list, adj_list = None, mode = "Euclidean", cvImg = None) :
    global CONFIG_center_K

    center_list = get_center_list(cand_list)

    #print(center_list)

    if mode == "tree" :
        node_dist_tree = bfs_tree(adj_list, center_list, cvImg = cvImg)
    #elif mode == "Euclidean" :
        node_dist_Euclidean = Euclidean_distance_weight(cand_list, center_list)
    for node in cand_list :
        if node['num'] in node_dist_tree:
            node['weight'] = node_dist_tree[node['num']] + node_dist_Euclidean[node['num']]
            #node['weight'] = node_dist_tree[node['num']]
            #node['weight'] = node_dist_Euclidean[node['num']]
        else :
            node['weight'] = 10000000

    return cand_list

def Find(x, father) :
    if father[x] == x :
        return x
    father[x] = Find(father[x], father)
    return father[x]

def build_edge(adj_list, x, y) :
    if x in adj_list.keys() :
        adj_list[x].append(y)
    else :
        adj_list[x] = []
        adj_list[x].append(y)
    return adj_list

def build_random_tree(cand_list, slic_adj_list, number_of_trees = 1) :
    father = {}    
    edgeList = []

    #print("\nbuilding random tree...")
    adj_list = {}
    flag = np.zeros(1005)
    for item in cand_list :
        flag[item['num']] = 1
    for item in cand_list:
        node = item['num']
        adj_list[node] = []
        father[node] = node
        for nb in slic_adj_list[node] :
            if flag[nb] :
                edgeList.append((node, nb))

    random.shuffle(edgeList)

    number_of_tree_edges = 0
    number_of_tree_nodes = len(cand_list)
    for edge in edgeList :
        x = edge[0]
        y = edge[1]
        if Find(x, father) != Find(y, father) :
            father[Find(x, father)] = Find(y, father)
            adj_list = build_edge(adj_list, x, y)
            adj_list = build_edge(adj_list, y, x)
            number_of_tree_edges += 1
            if number_of_tree_edges >= number_of_tree_nodes - number_of_trees :
                break

    return adj_list 
    

