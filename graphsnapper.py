import getopt
import sys
import time

import numpy as np
from tqdm import trange

import graphtools as gt
from graphtools import h5py, stdout

# Fifth ring
# xmin = 116.1904 * long2km
# xmax = 116.583642 * long2km
# ymin = 39.758029 * lat2km
# ymax = 40.04453 * lat2km

global_start = time.time()

# Grab MPI components from graphtools
mpicomm = gt.comm
mpirank = gt.rank
mpisize = gt.size

arglen = len(sys.argv[1:])
arglist = sys.argv[1:]

runname = ""
runpath = "/scratch/walterms/traffic/graphnn/veldata/"
outpath = "/scratch/walterms/traffic/graphnn/nn_inputs/"

node_mindist = 0.05  # nodes closer than this will be eliminated for redundancy
node_maxdist = 2.0
maxnbr = 8
nvel = None
disable_tqdm = False

try:
    opts, args = getopt.getopt(
        arglist, "", ["node_mindist=", "node_maxdist=", "maxnbr=", \
            "nvel=", "runname=", "runpath=", "outpath=", "disable_tqdm="]
    )
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

if not runname:
    stdout("Please specify velocity filename", mpirank)

stdout("Processing velocity file " + runpath + runname + " for nn inputs", mpirank)
stdout("Saving input files to " + outpath, mpirank)

# Write params to output info
if mpirank == 0:
    info_outname = outpath + runname + ".info"
    info_out = open(info_outname, 'w')
    info_out.write("run " + runpath + runname + "\n")
    info_out.write("outpath " + outpath + "\n")
    info_out.write("node_mindist " + str(node_mindist) + "\n")
    info_out.write("node_maxdist " + str(node_maxdist) + "\n")
    info_out.write("maxnbr " + str(maxnbr) + "\n")

info = {}
infofname = runpath + runname + ".info"
stdout("Creating info dict", mpirank)
info = gt.get_info_dict(infofname)
if mpirank == 0:
    for key, val in sorted(info.items()):
        stdout(str(key) + "\t" + str(val), mpirank)
        info_out.write(str(key) + " " + str(val) + "\n")

# Gather nodes
stdout("Generating nodes and edges...", mpirank)
tnode = time.time()
xmin, xmax = info["xmin"] / gt.long2km, info["xmax"] / gt.long2km
ymin, ymax = info["ymin"] / gt.lat2km, info["ymax"] / gt.lat2km
region_gsi = [xmin, xmax, ymin, ymax]
nTG = info["nTG"]

nodes, edges = gt.generate_nodes(
    region=region_gsi, mindist=node_mindist,
    maxdist=node_maxdist, maxnbr=maxnbr, disable_tqdm=disable_tqdm
)

tnode_ = time.time()
stdout("Done nodes and edges: " + str(tnode_ - tnode) + " seconds", mpirank)

n_nodes, n_edges = len(nodes.index), len(edges.index)
if mpirank == 0:
    stdout("Number of nodes " + str(n_nodes), mpirank)
    info_out.write("n_nodes " + str(n_nodes) + "\n")
    stdout("Number of edges " + str(n_edges), mpirank)
    info_out.write("n_edges " + str(n_edges) + "\n")

stdout("Generating velocity dataframe...", mpirank)
stdout("vel file contains " + str(info["n_points"]) + " points", mpirank)

tvdf = time.time()
# Grab the hdf5 file and divide up
f5vel = h5py.File(runpath + runname + ".hdf5", 'r', driver="mpio", comm=mpicomm)
# Check if they have the right amount of vel points
if f5vel.attrs["nvel"] != info["n_points"]:
    stdout("nvel from hdf5 (" + str(nvel) + ") != nvel from info (" + str(info["n_points"]) + ")", mpirank)
    sys.exit(2)

# Use f5 nvel is not specified
if not nvel:
    nvel = f5vel.attrs["nvel"]
if mpirank == 0:
    info_out.write("nvel " + str(nvel) + "\n")

nvel_per = nvel // mpisize
vstart, vend = nvel_per * mpirank, nvel_per * (mpirank + 1)
remainder = nvel % mpisize
if mpirank == (mpisize - 1):
    vend += remainder
    nvel_per += remainder

# Divide up the dataset
vdset = f5vel["veldat"]
vdata_np = vdset[vstart:vend]

# I think we can close f5vel now
f5vel.close()
del vdset

# We can append all the vdfs after using comm.gather which creates a list
vdf = gt.build_vdf(vdata_np, nodes, nTG=nTG, nvel=nvel_per, disable_tqdm=disable_tqdm)

# Sometimes a stationary car will take up a lot of points
vdflist = mpicomm.gather(vdf, root=0)
if mpirank == 0:
    vdf = vdf.append(vdflist[1:], ignore_index=True)
    tvdf_ = time.time()
del vdflist, vdata_np
# Perhaps I should broadcast vdf?...what if it's very big...
vdf = mpicomm.bcast(vdf, root=0)

if mpirank == 0:
    stdout(str(tvdf_ - tvdf) + " seconds", mpirank)
    stdout("Number of vel points " + str(len(vdf.index)), mpirank)
    info_out.write("Number of vel points" + str(len(vdf.index)) + "\n")


def zero_velstats(node_df, edge_df):
    node_df["ncar"] = 0.
    node_df["v_avg"] = 0.
    node_df["v_std"] = 0.
    edge_df["ncar_out"] = 0.
    edge_df["ncar_in"] = 0.
    edge_df["v_avg_out"] = 0.
    edge_df["v_avg_in"] = 0.
    edge_df["v_std_out"] = 0.
    edge_df["v_std_in"] = 0.


zero_velstats(nodes, edges)

if mpirank == 0:
    info_out.write("node feature header: [ncar, v_avg, v_std]" + "\n")
    info_out.write("edge feature header: [ncar_out, v_avg_out, v_std_out, ncar_in, v_avg_in, v_std_in]" + "\n")
    info_out.write("global feature header: [day, tg]" + "\n")

nsnap = 7 * nTG

# Make output hdf5 file
h5out = h5py.File(outpath + runname + ".hdf5", 'w', driver="mpio", comm=mpicomm)
h5out.atomic = True

# Add info attributes
for key, val in sorted(info.items()):
    h5out.attrs[key] = val
h5out.attrs['source'] = runpath + runname + ".hdf5"
h5out.attrs['node_mindist'] = node_mindist
h5out.attrs['node_maxdist'] = node_maxdist
h5out.attrs['maxnbr'] = maxnbr
h5out.attrs['n_nodes'] = n_nodes
h5out.attrs['n_edges'] = n_edges
h5out.attrs['n_unique_vel'] = nvel
h5out.attrs['nvel'] = len(vdf.index)
h5out.attrs['node_feat_header'] = ['ncar', 'v_avg', 'v_std']
h5out.attrs['edge_feat_header'] = ['ncar_out', 'v_avg_out', 'v_std_out', 'ncar_in', 'v_avg_in', 'v_std_in']
h5out.attrs['glbl_feat_header'] = ['day', 'tg']
# Rename these
h5out.attrs['n_source_drivers_total'] = int(h5out.attrs['n_drivers'])
h5out.attrs['n_source_vels'] = int(h5out.attrs['n_points'])
mpicomm.Barrier()
h5out.attrs.__delitem__('n_drivers')
h5out.attrs.__delitem__('n_points')

h5out.create_group("glbl_features")
h5out.create_group("node_features")
h5out.create_group("edge_features")

np_send = edges['sender'].values.copy().astype(np.int)
np_rece = edges['receiver'].values.copy().astype(np.int)
h5_send = h5out.create_dataset("senders", np_send.shape, dtype=np.int)
h5_rece = h5out.create_dataset("receivers", np_rece.shape, dtype=np.int)
if mpirank == 0:
    h5_send[:] = np_send
    h5_rece[:] = np_rece

# Create datasets for each day.tg
# Use ABC for shapes
A = nodes[["ncar", "v_avg", "v_std"]].values.copy()
B = edges[["ncar_out", "v_avg_out", "v_std_out", "ncar_in", "v_avg_in", "v_std_in"]].values.copy()
for day in range(7):
    for tg in range(nTG):
        h5out.create_dataset(
            "node_features/day" + str(day) + "tg" + str(tg),
            A.shape, dtype=np.float
        )
        h5out.create_dataset(
            "edge_features/day" + str(day) + "tg" + str(tg),
            B.shape, dtype=np.float
        )
        h5out.create_dataset(
            "glbl_features/day" + str(day) + "tg" + str(tg), \
            (1, 2), dtype=np.int
        )
del A, B

# Need to figure out how to divy up the processors
# accross the days and tgs
mydays, mytgs, subrank = None, None, None
if mpisize < 7:
    day_per_proc = 7 // mpisize
    rem_day = 7 % mpisize
    mydays = np.arange(mpirank * day_per_proc, (mpirank + 1) * day_per_proc, dtype=np.int)
    if mpirank == (mpisize - 1):
        mydays = np.append(mydays, np.arange(7 - rem_day, 7))
    mytgs = np.arange(info["nTG"])
    subrank = mpirank
else:
    np_per_daygroup = mpisize // 7
    np_remainder = mpisize % 7
    # Should distribute the remainder processes as thin as possible
    # So add one for each daygroup < remainder
    daygroup = mpirank % 7
    if daygroup < np_remainder:
        np_per_daygroup += 1
    mydays = [mpirank % 7]
    # Now divide up ntg for each group
    ntg_per = nTG // np_per_daygroup
    tg_remainder = nTG % np_per_daygroup
    subrank = mpirank // 7
    tgstart, tgend = subrank * ntg_per, (subrank + 1) * ntg_per
    if subrank == 6:
        tgend += tg_remainder
    mytgs = np.arange(tgstart, tgend)

stdout("Performing main feature construction loop", mpirank)
tloop_start = time.time()
bdiag = False
tg_diag = len(mytgs) // 10

for day in trange(mydays[0], mydays[-1] + 1, desc='Days     ', disable=disable_tqdm):
    dayloop_start = time.time()
    for tg in trange(mytgs[0], mytgs[-1] + 1, desc='TimeGroup', disable=disable_tqdm):
        # Zero the velstats
        zero_velstats(nodes, edges)

        # Get velstats for this day, tg
        vdf_ = vdf[(vdf['day'] == day) & (vdf['tg'] == tg)]
        for idx, node in nodes.iterrows():
            # Given this subset of vels, calc stats for each node
            vels = vdf_[vdf_['nodeID'] == idx]
            ncar = len(vels.index)
            nodes.at[idx, 'ncar'] = ncar
            if ncar == 0:
                continue
            nodes.at[idx, 'v_avg'] = vels.mean(axis=0)['v']
            if ncar > 1:
                nodes.at[idx, 'v_std'] = vels.std(axis=0)['v']

            # Iterate over this nodes edges, adding vel stats as necessary
            # Not that many edges for a given idx, so don't need to parallelize
            edges_ = edges[edges["sender"] == idx]
            for eidx, e in edges_.iterrows():
                v_out, v_in = [], []
                # Iterate over vels which belong to this node
                for iv, v in vels.iterrows():
                    dtheta = v['angle'] - e['angle']
                    if (abs(dtheta) < 0.25 * np.pi) | (abs(dtheta) > 1.75 * np.pi):
                        v_out.append(v['v'])
                    if (abs(dtheta) > np.pi * 0.75) & (abs(dtheta) < np.pi * 1.25):
                        v_in.append(v['v'])

                if len(v_out) > 0:
                    edges.at[eidx, "ncar_out"] = len(v_out)
                    v_avg_out, v_std_out = np.mean(v_out), np.std(v_out)
                    edges.at[eidx, "v_avg_out"] = v_avg_out
                    edges.at[eidx, "v_std_out"] = v_std_out
                if len(v_in) > 0:
                    edges.at[eidx, "ncar_in"] = len(v_in)
                    v_avg_in, v_std_in = np.mean(v_in), np.std(v_in)
                    edges.at[eidx, "v_avg_in"] = v_avg_in
                    edges.at[eidx, "v_std_in"] = v_std_in

        # Add to arrays
        h5_A = h5out["node_features/day" + str(day) + "tg" + str(tg)]
        h5_B = h5out["edge_features/day" + str(day) + "tg" + str(tg)]
        h5_C = h5out["glbl_features/day" + str(day) + "tg" + str(tg)]
        h5_A[:] = nodes[["ncar", "v_avg", "v_std"]].values
        h5_B[:] = edges[["ncar_out", "v_avg_out", "v_std_out", "ncar_in", "v_avg_in", "v_std_in"]].values
        h5_C[:] = np.array([[day, tg]], dtype=np.int)

        # Print a diagnostic
        if (mpirank == 0) and not bdiag:
            if tg == mytgs[tg_diag]:
                stdout("Progress diagnostic:", mpirank)
                stdout(
                    "Time to complete " + str(tg_diag) + " tgs (~10%) of " + str(len(mytgs)) \
                    + ": " + str((time.time() - dayloop_start) / 60.) + " min", mpirank
                )
                bdiag = True

mpicomm.Barrier()

global_end = time.time()
if mpirank == 0:
    stdout("\nTotal time for main loop: " + str(global_end - tloop_start) + " s", mpirank)
    info_out.write("Total time for main loop: " + str(global_end - tloop_start) + " s" + "\n")
    stdout("Total time for graphsnapper: " + str(global_end - global_start) + " s", mpirank)
    info_out.write("Total time for graphsnapper: " + str(global_end - global_start) + " s" + "\n")
    info_out.close()

h5out.close()

mpicomm.Barrier()
gt.finalize_mpi()
