import numpy as np
import graphtools as gt
from graphtools import stdout
from graphtools import h5py
import pandas as pd
import sys, getopt, os
import time
from tqdm import tqdm, trange

mpicomm = gt.comm
mpirank = gt.rank
mpisize = gt.size

arglen = len(sys.argv[1:])
arglist = sys.argv[1:]

runname = ""
runpath = "/scratch/walterms/traffic/graphnn/veldata/"
outpath = "/scratch/walterms/traffic/graphnn/nn_inputs/"

node_mindist = 0.05 # nodes closer than this will be eliminated for redundancy
node_maxdist = 2.0
maxnbr = 8
nvel = None
disable_tqdm = False

try:
    opts, args = getopt.getopt(arglist,"",["node_mindist=","node_maxdist=","maxnbr=",\
        "nvel=","runname=", "runpath=", "outpath=", "disable_tqdm="])
except:
    stdout("Error in opt retrieval...", mpirank)
    sys.exit(2)

for opt, arg in opts:
    if opt == "--node_mindist":
        node_mindist = float(arg)
    elif opt == "--node_maxdist":
        node_maxdist = float(arg)
    elif opt == "--maxnbr":
        maxnbr = int(arg)
    elif opt == "--nvel":
        nvel = int(arg)
    elif opt == "--runname":
        runname = arg
    elif opt == "--runpath":
        runpath = arg
    elif opt == "--outpath":
        outpath = arg
    elif opt == "--disable_tqdm":
        disable_tqdm = bool(arg)

info = {}
infofname = runpath+runname+".info"
stdout("Creating info dict", mpirank)
info = gt.get_info_dict(infofname)

# Gather nodes
stdout("Generating nodes and edges...", mpirank)
tnode = time.time()
xmin,xmax = info["xmin"]/gt.long2km, info["xmax"]/gt.long2km
ymin,ymax = info["ymin"]/gt.lat2km, info["ymax"]/gt.lat2km
region_gsi = [xmin,xmax,ymin,ymax]
nTG = info["nTG"]

nodes, _ = gt.generate_nodes(region=region_gsi, mindist=node_mindist, 
    maxdist=node_maxdist, maxnbr=maxnbr, disable_tqdm=disable_tqdm)
nnode = len(nodes.index)

h5f = h5py.File(outpath+runname+".hdf5", 'a', driver="mpio", comm=mpicomm)
h5f.atomic=True

nodes.sort_index(inplace=True)
nodeseries = nodes["coords_km"]
nodearr = np.zeros((nnode,2),dtype=np.float)
for i,row in enumerate(nodeseries):
    nodearr[i]=row

h5_nodes = h5f.create_dataset("node_coords", shape=(nnode,2),dtype=np.float)
if mpirank==0:
    h5_nodes[:] = nodearr
h5f.close()
