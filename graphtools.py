import numpy as np
import csv
import sys
import time
import pandas as pd
from tqdm import tqdm


import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
import h5py

MPI.Init()
assert MPI.Is_initialized()


# Initialize MPI actors
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

long2km = 1/0.011741652782473
lat2km = 1/0.008994627867046

def finalize_mpi():
    MPI.Finalize()
    return

def stdout(s, rank=None):
    if not rank:
        sys.stdout.write(str(s)+'\n')
    elif rank==0:
        sys.stdout.write(str(s)+'\n')
    else: return

def dist_from(ref,rs):
    # ref is 1x2 ndarray
    # rs is list of vectors to compare, nx2
    return np.linalg.norm((ref-rs),axis=1)

def coord2km(coord):
    return [coord[0]*long2km, coord[1]*lat2km]

def df_empty(columns, dtypes, index=None):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

def nodes4vel(p, nodes_df, within):
    nodes_df['z'] = dist_from(p, np.asarray(nodes_df['coords_km'].tolist()))
    nodes_df.sort_values(by=['z'],inplace=True)
    nodes_return, d2n_return = [],[]
    z = nodes_df.iloc[0]['z']
    i = 0
    while z < within:
        nodes_return.append(nodes_df.index[i])
        d2n_return.append(z)
        i+=1
        z = nodes_df.iloc[i]['z']
    return nodes_return, d2n_return

def node_coords_np(nodedict):
    return np.array([node["coords"] for node in nodedict.values()],dtype=np.float)

def get_edge_angle(nodedf,i,j):
    dr = np.array(nodedf.at[j,"coords_km"]) - np.array(nodedf.at[i,"coords_km"])
    z = np.complex(dr[0], dr[1])
    return np.angle(z)   
    
    
def build_vdf(data_np, nodedf, days=[], tgs=[], nTG=None, nvel=None, node4vel_range=1.0, **kwargs):
    #
    # Let's have graphsnapper call this function from each processor
    # Pass build_vdf the appropriate numpy data slice to be processed each call
    #
    if len(days) == 0:
        # Grab all days
        days = np.arange(7)
    if len(tgs) == 0:
        if not nTG:
            print("If no tgs provided, specify nTG. Exiting")
            return
        tgs = np.arange(nTG)

    vdf = pd.DataFrame(columns=["day","tg","x_km","y_km","vx","vy","v","nodeID","dist2node","angle"])

    vi = 0
    #for v in tqdm(vfile.readlines(), disable=kwargs["disable_tqdm"]):
    for row in tqdm(data_np, disable=kwargs["disable_tqdm"]):
        day = int(row[0])
        tg = int(row[1])
        if day not in days: continue
        if tg not in tgs: continue
        # Apparently appending a list of dict
        # preserves the datatype
        # See: https://stackoverflow.com/questions/21281463/appending-to-a-dataframe-converts-dtypes
        x_km, y_km = float(row[2]), float(row[3])
        nbrnodes, d2ns = nodes4vel([x_km, y_km], nodedf, within=node4vel_range)
        if len(nbrnodes) == 0:
            # point is too far from any nodes for 
            # this to be accurate data
            continue
        
        # Get vel angle in [-pi,pi] format
        vx, vy = float(row[4]), float(row[5])
        z = np.complex(vx,vy)
        angle = np.angle(z)

        # Iterate over neighbours and add to vdf
        for inbr in range(len(nbrnodes)):
            vdf = vdf.append([{
                "day": day,
                "tg": tg,
                "x_km": x_km,
                "y_km": y_km,
                "vx": vx,
                "vy": vy,
                "v": float(row[6]),
                "nodeID": nbrnodes[inbr],
                "dist2node": d2ns[inbr],
                "angle": angle
            }], ignore_index=True)

        vi+=1
        if nvel and vi >= nvel: break

    vdf.drop_duplicates(inplace=True)
    return vdf


def generate_nodes(fname="./hwy_pts.csv",
                   mindist=0.05, 
                   region=None, 
                   **kwargs):
    # region is scope for domain, [xmin,xmax,ymin,ymax] in GSI coordinates
    # use kwargs for passing kws to generate_edges
    # mindist to reduce redundant nodes
    # We can optimize node removal during the edge finding process
    nodedict = {}
    nodeidx = 0
    with open(fname, newline='') as hwyfile:
        reader = csv.reader(hwyfile)
        next(reader, None) # skip header
        stdout("Creating nodes...", rank)
        for l in tqdm(reader, disable=kwargs["disable_tqdm"]):
            x,y = float(l[0]), float(l[1])
            if region:
                if x<region[0] or x>region[1] or y<region[2] or y>region[3]:
                    continue
            nodedict.update({nodeidx:{"coords":[float(l[0]),float(l[1])]}})
            nodedict[nodeidx].update({"nbrs": []})
            nodeidx+=1

    df = pd.DataFrame(nodedict).T
    df["coords_km"] = df.apply(lambda x: coord2km(x["coords"]), axis=1)
    
    # Link neighbours and drop redundant nodes
    stdout("Linking neighbours...", rank)
    tn1 = time.time()
    link_neighbours(df,mindist=mindist, **kwargs)
    # Reorder by index
    df.sort_index(inplace=True)
    tn2 = time.time()
    stdout("Linked neighbours in "+str(tn2-tn1)+" seconds",rank)

    # Let's re-index the nodes
    rekey = {}
    newkey = 0
    for key,node in df.iterrows():
        rekey.update({key: newkey})
        newkey+=1
    df.reset_index(drop=True,inplace=True)

    # Update nbr lists with new keys
    for key,node in df.iterrows():
        nbrs = node['nbrs']
        newnbrs = []
        for nbr in nbrs:
            newnbrs.append(rekey[nbr])
        df.at[key,'nbrs'] = list(newnbrs)

    # Edges
    edges = pd.DataFrame(columns=["sender","receiver","angle"])
    stdout("Generating edges...", rank)
    nnode = len(df.index)
    nnode_per = nnode//size
    remainder = nnode%size
    nstart, nend = rank*nnode_per, (rank+1)*nnode_per
    if rank == (size-1): nend += remainder
        
    # First we do out initial edge making with the maxnbr restriction
    for key,node in tqdm(df[nstart:nend].iterrows(), disable=kwargs["disable_tqdm"]):
        for nbr in node['nbrs']:
            # Get theta in [-pi,pi] radians
            theta = get_edge_angle(df,key,nbr)
            edges = edges.append([{
                "sender": key, 
                "receiver": nbr, 
                "angle": theta
            }], ignore_index=True)

    stdout("Finished the initial edgemaking limited by maxnbr "+str(kwargs["maxnbr"]),rank)
    # Combine edge dfs
    edges_dflist = comm.gather(edges,root=0)
    if rank==0:
        edges = edges.append(edges_dflist[1:], ignore_index=True)
    del edges_dflist
    edges.drop_duplicates(inplace=True)
    edges = comm.bcast(edges,root=0)

    stdout("Patching up the one-way edges to create two-ways...",rank)
    # Now we patch up the one-way connections that were missed from the maxnbr restriction
    # The edges df is unique to each process, so iterate the whole lot
    for idx,edge in tqdm(edges.iterrows(), disable=kwargs["disable_tqdm"]):
        # Grab a sender and see if it appears as a receiver to each of its
        # own receivers. If not, add it
        s,r,th = edge['sender'], edge['receiver'], edge['angle']
        r_receivers = edges.loc[edges['sender']==r, 'receiver'].values
        if s not in r_receivers:
            th_ = (th+np.pi) if th<0 else (th-np.pi)
            edges = edges.append({'sender': r, 'receiver': s, 'angle': th_}, ignore_index=True)
        
    return df, edges


def link_neighbours(df, mindist=0.05, maxdist=1.0, maxnbr=8, **kwargs):
    '''
    This function has two main stages
    First it finds which nodes are too close together (the nearnbrs loop)
    It finds these and compiles the list using gather and then removes nodes
    Then it does the same loop again but for the real nbrs up to maxnbr
    '''
    # Create km coords
    if "coords_km" not in df.columns:
        df["coords_km"] = df.apply(lambda x: coord2km(x['coords']), axis=1)

    # Create edgeref column
    if "nbrs" not in df.columns:
        df["nbrs"] = [[] for _ in df.index]

    # Create nearnbrs column for redundancy removal
    # Also, by adding how many neighbours we can
    # sort by num nearnbrs to optimize which nodes
    # get deleted. A node with many neighbours should
    # kill the neighbours because this node is a good approx
    # for all of them
    df["nearnbrs"] = [[] for _ in df.index]
    df["n_nearnbrs"] = 0
    df['z'] = 0.

    nnode = len(df.index)
    nnode_per = nnode//size
    remainder = nnode%size
    nstart, nend = rank*nnode_per, (rank+1)*nnode_per
    if rank == (size-1):
        nend += remainder
    for idx,node in df[nstart:nend].iterrows():
        df['z'] = dist_from(node['coords_km'], np.asarray(df['coords_km'].tolist()))
        df.sort_values(by=['z'], inplace=True)

        nearnbrs = []
        for j in df.index[1:]:
            if df.loc[j]['z'] < mindist:
                nearnbrs.append(j)
            else:
                break
        n=len(nearnbrs)
        if n > 0:
            df.at[idx,"nearnbrs"] = list(nearnbrs)
            df.at[idx,"n_nearnbrs"] = n


    # Each node has a nearnbr list
    # Sort by n_nearnbr and remove nodes
    # You need to combine all the nearnbr lists
    # and then broadcast the compiled list
    # then each processor delete the rows in their (to-be-)identical dfs
    df.sort_index(inplace=True)
    g_nbrlist = comm.gather(df[["nearnbrs","n_nearnbrs"]],root=0)
    df_ = None
    if rank==0:
        df_ = g_nbrlist[0].copy()
        for d in g_nbrlist[1:]:
            # Assert there is no overlap
            assert df_[d['n_nearnbrs']>0].sum(axis=0)['n_nearnbrs'] == 0
            df_.loc[d["n_nearnbrs"]>0,['nearnbrs','n_nearnbrs']] = d[['nearnbrs','n_nearnbrs']]
        # Assert we have no nans
        assert df_.isna().sum().sum() == 0
    df_ = comm.bcast(df_,root=0)
    df.loc[df_["n_nearnbrs"]>0,['nearnbrs','n_nearnbrs']] = df_[['nearnbrs','n_nearnbrs']]

    # Want to find nodes with the most neighbours and elimate said nbrs
    # (If only it were that easy)
    df.sort_values(by=["n_nearnbrs"],inplace=True, ascending=False)

    # Remove redundant neighbours
    stdout("Deleting nearnbrs...",rank)
    pre_nnode = nnode
    for lbl,row in df.iterrows():
        if row["n_nearnbrs"] < 1: break
        if row["n_nearnbrs"] == 1:
            nb = int(row["nearnbrs"][0])
            if nb in df.index:
                removes = nb
            else:
                removes = []
        else:
            removes = list(set(list(row["nearnbrs"])) & set(df.index))
        df.drop(removes, inplace=True)
    post_nnode = len(df.index)
    stdout("Removed "+str(pre_nnode-post_nnode)+" redundant nbrs",rank)

    df.drop(columns=['n_nearnbrs','nearnbrs'],inplace=True)
    stdout("Connecting final nbrs...", rank)
    nnode = len(df.index)
    nnode_per = nnode//size
    remainder = nnode%size
    nstart, nend = rank*nnode_per, (rank+1)*nnode_per
    if rank == (size-1):
        nend += remainder
    # Fill the 'nbrs' col of df 
    df['n_nbrs'] = 0
    for idx,node in tqdm(df[nstart:nend].iterrows(), disable=kwargs["disable_tqdm"]):
        # Populate 'z' column
        df['z'] = dist_from(node['coords_km'], np.asarray(df['coords_km'].tolist()))
        df.sort_values(by=['z'],inplace=True)
        nbrs = []
        n_nbr = 0
        for j in df.index[1:]:
            if df.loc[j,'z'] < maxdist:
                nbrs.append(j)
                n_nbr+=1
            else: break
            if n_nbr == maxnbr: break
        df.at[idx,"nbrs"] = list(nbrs)
        df.at[idx,"n_nbrs"] = n_nbr

    # Update eachother's nbr lists
    g_nbrlist = comm.gather(df[["n_nbrs","nbrs"]],root=0)
    df_ = None
    if rank==0:
        df_ = g_nbrlist[0].copy()
        for d in g_nbrlist[1:]:
            assert df_.loc[d['n_nbrs']>0].sum(axis=0)['n_nbrs'] == 0
            df_.loc[d['n_nbrs']>0,['n_nbrs','nbrs']] = d[['n_nbrs','nbrs']]
        # Assert we have no nans
        assert df_.isna().sum().sum() == 0
    df_ = comm.bcast(df_,root=0)
    df.loc[df_["n_nbrs"]>0,['nbrs','n_nbrs']] = df_[['nbrs','n_nbrs']]

    # Remove z and n_nbrs
    df.drop(columns=['z','n_nbrs'],inplace=True)
    return
    
        
class graphplot:
    '''
    At some point you should create a function to zoom in on certain regions
    
    Possibly a very useful example if wanting to implement a live inset viewer:
    https://matplotlib.org/3.1.1/users/event_handling.html
    '''
    def __init__(self,nodes_obj,idxlist=[],figsize=None,figure=None,axis=None,usekm=False):
        self.coordunits = "coords"
        if usekm: self.coordunits = "coords_km"
            
        # First see if nodes is a dataframe
        if "DataFrame" in str(type(nodes_obj)):
            nodedict = nodes_obj.to_dict("index")
        else:
            nodedict = nodes_obj
            
        # Create index list of all nodes if empty
        if len(idxlist) == 0:
            for k, v in nodedict.items():
                idxlist.append(k)
        self.idxlist = idxlist
        self.nodes = {}
        self.edges = {}
        for idx in idxlist:
            # If idxlist is a subset of all nodes
            # we need to create a temporary edge list
            # to only plot the edges involved
            self.nodes.update({idx:nodedict[idx]})
            eidxs = nodedict[idx]["nbrs"]
            new_eidxs = []
            for ei in eidxs:
                if ei in idxlist: new_eidxs.append(ei)
            self.edges.update({idx:list(new_eidxs)})
        self.npnodes = np.array([node[self.coordunits] for node in self.nodes.values()],dtype=np.float)
        self.nnodes = len(self.npnodes)
        self.xlims = [np.min(self.npnodes[:,0]),np.max(self.npnodes[:,0])]
        self.ylims = [np.min(self.npnodes[:,1]),np.max(self.npnodes[:,1])]
        self.graphwidth = self.xlims[1] - self.xlims[0]
        self.graphheight = self.ylims[1] - self.ylims[0]
        xmargin = self.graphwidth * 0.1
        ymargin = self.graphheight * 0.1
        self.xlims = [self.xlims[0]-xmargin,self.xlims[1]+xmargin]
        self.ylims = [self.ylims[0]-ymargin,self.ylims[1]+ymargin]
        self.graphwidth *= 1.2
        self.graphheight *= 1.2
        self.nodecolor = "darkblue"
        self.noderadius = self.graphwidth / self.nnodes
        if usekm:
            if self.noderadius < 0.1: self.noderadius = 0.1
            if self.noderadius > 0.5: self.noderadius = 0.5
        else:
            if self.noderadius < 0.001: self.noderadius = 0.001
            if self.noderadius > 0.01: self.noderadius = 0.01
        
        if not axis or not figure:
            self.fig, self.ax = plt.subplots(1,1,figsize=(9,9) if not figsize else figsize)
        else:
            self.fig, self.ax = figure, axis
        self.ax.set_ylim(self.ylims[0], self.ylims[1])
        self.ax.set_xlim(self.xlims[0], self.xlims[1])
        self.ax.set_aspect('equal', adjustable='box',anchor="NW")
        
    def drawgraph(self):
        for i in self.idxlist:
            node, edges = self.nodes[i][self.coordunits], self.edges[i]
            for edge in edges:
                nbr = self.nodes[edge][self.coordunits]
                self.plotLine(self.ax,node[0],node[1],nbr[0],nbr[1])
            self.plotnode(self.ax,node[0], node[1])            
        
    def plotnode(self,ax,x,y):
        circ = Circle((x,y),self.noderadius,color="b",alpha=0.8)
        ax.add_patch(circ)        
        
    def plotLine(self,ax,x1,y1,x2,y2,c='slategrey',lw=1.0,alpha=1.0):
        ax.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw, alpha=alpha);
        
    def savefig(self,path):
        self.fig.savefig(path)
        
    def get_rect_coords(self,rect):
        # returns [xmin,ymin,xmax,ymax]
        return rect.get_bbox().get_points().flatten()

        
class viewer(graphplot):
    '''
    Do:
    viewer = viewer(window)
    viewer.connect()
    To instantiate and use
    '''
    def __init__(self, nodes_obj, window=None, figsize=None, idxlist=[], usekm=False):
        # window are the [xmin,xmax,ymin,ymax] dimensions of the viewer
        
        self.coordunits = "coords"
        if usekm: self.coordunits = "coords_km"
        
        # Create nodedict if nodes is a dataframe
        if "DataFrame" in str(type(nodes_obj)):
            nodedict = nodes_obj.to_dict("index")
        else:
            nodedict = nodes_obj
            
        self.figsize = (9,9) if not figsize else figsize
        self.fig_main = plt.figure(figsize=self.figsize)
        self.grid = plt.GridSpec(2,2,hspace=0.1, wspace=0.1,width_ratios=[2.2,1],height_ratios=[0.8,1])
        self.axs = [self.fig_main.add_subplot(self.grid[:,0])]
        self.axs.append(self.fig_main.add_subplot(self.grid[0,1]))

        # Initialize parent graph object
        super().__init__(nodedict,idxlist,(self.figsize[0]*0.75,self.figsize[1]*0.75), 
                         self.fig_main, self.axs[0],usekm=usekm)
        self.axs[0].set_aspect('equal', adjustable='box',anchor="NW")

        # Draw main graph
        self.drawgraph()
        
        # Setup viewer
        if not window:
        #    window = [self.xlims[0]+0.4*self.graphwidth,
        #              self.xlims[0]+0.6*self.graphwidth,
        #              self.ylims[0]+0.4*self.graphheight,
        #              self.ylims[0]+0.6*self.graphheight]
            dw = 1. / long2km
            if usekm: dw = 1.
            window = [self.xlims[0] + 0.5*self.graphwidth - dw,
                      self.xlims[0] + 0.5*self.graphwidth + dw,
                      self.ylims[0] + 0.5*self.graphheight - dw,
                      self.ylims[0] + 0.5*self.graphheight + dw]
        xmin, xmax, ymin, ymax = window[0], window[1], window[2], window[3]
        self.window = Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,linewidth=1.6,edgecolor="orangered")
        self.axs[0].add_patch(self.window)
        
        # Create inset axis
        xmin,ymin,xmax,ymax = self.get_rect_coords(self.window)
        self.axs[1].set_ylim(ymin,ymax)
        self.axs[1].set_xlim(xmin,xmax)
        self.axs[1].set_aspect('equal', adjustable='box',anchor="NE")
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])
        self.press = None
        
        self.updateinset()
        
    def updateinset(self):
        # Remove any artists if present
        self.axs[1].clear()
        
        # Collect which nodes are included
        insetnodes = {}
        xmin,ymin,xmax,ymax = self.get_rect_coords(self.window)        
        for key,node in self.nodes.items():
            x,y = node[self.coordunits]
            if x>xmin and x<xmax and y>ymin and y<ymax:
                insetnodes.update({key: node})
                self.plotnode(self.axs[1],x,y)
                edges = node["nbrs"]
                for e in edges:
                    if e in self.idxlist:
                        nbr = self.nodes[e][self.coordunits]
                        self.plotLine(self.axs[1],x,y,nbr[0],nbr[1])

        self.axs[1].set_ylim(ymin,ymax)
        self.axs[1].set_xlim(xmin,xmax)
        #self.axs[1].set_aspect('equal', adjustable='box',anchor="NE")
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])

#         pickle.dump(self., file('myplot.pickle', 'w'))        
        
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.window.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.window.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.window.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        
    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.window.axes: return

        contains, attrd = self.window.contains(event)
        self.window.set_x(event.xdata - self.window.get_width()*0.5)
        self.window.set_y(event.ydata - self.window.get_height()*0.5)
        if True:
            # Update attributes for motion
            print('event contains', self.window.xy)
            x0, y0 = self.window.xy
            self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the window if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.window.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))

        self.window.set_x(x0+dx)
        self.window.set_y(y0+dy)
        self.window.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.updateinset()
        self.window.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.window.figure.canvas.mpl_disconnect(self.cidpress)
        self.window.figure.canvas.mpl_disconnect(self.cidrelease)
        self.window.figure.canvas.mpl_disconnect(self.cidmotion)

        
def get_info_dict(fname, type="vel"):
    info = {}
    finfo = open(fname,'r')
    for l in finfo.readlines():
        words = l.split()
        if "tglen" in words:
            info.update({"tglen": int(words[1])})
        elif "nTG" in words:
            info.update({"nTG": int(words[1])})
        elif "source" in words:
            info.update({"source": words[1]})
        elif "drivers" in words:
            info.update({"n_drivers": int(words[0])})
        elif "points" in words:
            info.update({"n_points": int(words[0])})
        else:
            info.update({str(words[0]): float(words[1])})
      
    finfo.close()
    return info
