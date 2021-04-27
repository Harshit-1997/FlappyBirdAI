from collections import defaultdict as ddict
from torch import nn
import torch as th
import random

import networkx as nx
import matplotlib.pyplot as plt

class Brain(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.DNA = ddict(lambda : [])
        self.depth = dict()
        self.Memory = ddict(lambda : th.zeros((1,1)))
        self.order = []
        self.remaining_inputs = []
        self.remaining_outputs = []

        if type(in_size) == int:
            self.input_size = in_size
            self.output_size = out_size

            for inp in range(self.input_size):
                self.order.append("inp"+str(inp+1))
                self.remaining_inputs.append("inp"+str(inp+1))
                self.depth["inp"+str(inp+1)]=0

            for out in range(self.output_size):
                self.remaining_outputs.append("out"+str(out+1))
                self.DNA["out"+str(out+1)]=[]

        else:
            # Merge Parents
            pass

########################################################################################################################

    def topologicalSortUtil(self,node,visited):
        visited[node] = True

        parents = [ v[0] for v in self.DNA[node] if v[-1]=='f']
        for pnode in parents:
            if visited[pnode] == False:
                self.topologicalSortUtil(pnode,visited)

        self.order.append(node)

########################################################################################################################

    def topologicalSort(self):
        visited=ddict(lambda : False)
        self.order = []

        nodes =list(self.DNA.keys())
        for node in nodes:
            if visited[node] == False and len(self.DNA[node])>0:
                self.topologicalSortUtil(node,visited)


########################################################################################################################
    def makeConnection(self):
        while True:
            a, b = random.choices(list(enumerate(self.remaining_inputs+self.order+self.remaining_outputs)),k=2)
            ifrm, cfrm = a
            ito, cto = b
            exists = False
            if ifrm != ito and not cto.startswith('inp') and (cfrm in self.order or cfrm in self.remaining_inputs):
                exists = False
                for v in self.DNA[cto]:
                    if v[0]==cfrm:
                        exists = True
                if not exists:
                    direction = 'f'
                    if cfrm[:3] != cto[:3] and cfrm.startswith("out"):
                        direction = 'b'
                    if cfrm[:3] == cto[:3] and ifrm > ito:
                        direction = 'b'
                    break
        
        weight = th.randint(-97,97,(1,1))/97
        self.DNA[cto].append([cfrm,weight,direction])
        if cto in self.remaining_outputs:
            self.remaining_outputs.remove(cto)
        if cfrm in self.remaining_inputs:
            self.remaining_inputs.remove(cfrm)
        self.topologicalSort()

########################################################################################################################

    def forward(self,inp):
        pass

########################################################################################################################    

    def drawNetwork(self):
        G = nx.DiGraph()
        pos = dict()
        ys = ddict(lambda : [])

        for node in self.order:
            if not node.startswith("inp"):
                self.depth[node] = max([ self.depth[src[0]] for src in self.DNA[node] if src[-1]=='f' ])+1
                ys[self.depth[node]].append(node)
            else:
                ys[0].append(node)

        for depth in ys:
            for y,node in enumerate(ys[depth]):
                y = y - len(ys[depth])/2 + random.randint(-97,97)/(97*1.5)
                pos[node]=(depth, y)


        for node in self.DNA.keys():
            for v in self.DNA[node]:
                G.add_edge(v[0], node)
        nx.draw(G,pos,node_size=200,with_labels=True)
        plt.show()

########################################################################################################################
if __name__ == "__main__":
    x = Brain(6,4)
    for _ in range(15):
        x.makeConnection()
x.drawNetwork()
