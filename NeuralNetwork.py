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
        self.Memory = ddict(lambda : th.zeros((1,1)))
        self.order = []
        self.inputs = []
        self.remaining_outputs = []

        if type(in_size) == int:
            self.input_size = in_size
            self.output_size = out_size
            for inp in range(self.input_size):
                self.order.append("inp"+str(inp+1))
                self.inputs.append("inp"+str(inp+1))
            for out in range(self.output_size):
                self.remaining_outputs.append("out"+str(out+1))
                self.DNA["out"+str(out+1)]=[]
        else:
            # Merge Parents
            pass

########################################################################################################################

    def topologicalSortUtil(self,node,visited):
        print(node)
        visited[node] = True

        parentz = [ v for v in self.DNA[node] if v[-1]=='f']
        print(parentz)
        parents = [ v[0] for v in self.DNA[node] if v[-1]=='f']
        for pnode in parents:
            if visited[pnode] == True:
                self.topologicalSortUtil(pnode,visited)

        self.order.append(node)

########################################################################################################################

    def topologicalSort(self):
        visited=ddict(lambda : False)
        self.order = []

        for node in self.DNA:
            if visited[node] == False:
                self.topologicalSortUtil(node,visited)

########################################################################################################################
    def makeConnection(self):
        while True:
            a, b = random.choices(list(enumerate(self.order+self.remaining_outputs)),k=2)
            ifrm, cfrm = a
            ito, cto = b
            exists = False
            if ifrm != ito and not cto.startswith('inp') and cfrm in self.order:
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
            self.order.append(cto)
        #self.topologicalSort()

########################################################################################################################

    def forward(self,inp):
        pass

########################################################################################################################    

    def drawNetwork(self):
        G = nx.DiGraph()
        pos =dict()
        for node in self.DNA.keys():
            for v in self.DNA[node]:
                G.add_edge(v[0], node)
        for i in range(1,self.input_size+1):
            pos["inp"+str(i)]=(1,i)
        for i in range(1,self.output_size+1):
            pos["out"+str(i)]=(2,i)
        nx.draw(G,pos,with_labels=True)
        plt.show()

########################################################################################################################
if __name__ == "__main__":
    x = Brain(6,4)
    for _ in range(24):
        x.makeConnection()
    for node , data in x.DNA.items():
        print(node,":")
        for v in data:
            print(v)
    x.drawNetwork()
