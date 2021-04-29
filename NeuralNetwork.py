from collections import defaultdict as ddict
from torch import nn
import torch as th
import random
import networkx as nx
import matplotlib.pyplot as plt

class Brain(nn.Module):

    HIDDEN_UNIT_ID=1
    HIDDEN_UNIT_RECORD=ddict(lambda : -1)

    def __init__(self, in_size, out_size=2, initial_connections=5):
        super().__init__()
        self.DNA = ddict(lambda : [])
        self.node_data = ddict(lambda : [])
        self.depth = dict()
        self.Memory = ddict(lambda : th.Tensor([0]))
        self.order = []
        self.remaining_inputs = []
        self.remaining_outputs = []
        self.layer = nn.Linear(1,1)

        if type(in_size) == int:
            self.input_size = in_size
            self.output_size = out_size

            for inp in range(self.input_size):
                self.remaining_inputs.append("inp"+str(inp+1))
                self.depth["inp"+str(inp+1)] = 0

            for out in range(self.output_size):
                self.remaining_outputs.append("out"+str(out+1))
                self.DNA["out"+str(out+1)]=[]
                self.depth["out"+str(out+1)] = 1

            for _ in range(initial_connections):
                self.makeConnection()

        else:
            # Merge Parents
            parent1 = in_size
            self.input_size = parent1.input_size
            self.output_size = parent1.output_size

            for inp in range(self.input_size):
                self.order.append("inp"+str(inp+1))
                self.remaining_inputs.append("inp"+str(inp+1))
                self.depth["inp"+str(inp+1)] = 0

            for out in range(self.output_size):
                self.remaining_outputs.append("out"+str(out+1))
                self.DNA["out"+str(out+1)]=[]
                self.depth["out"+str(out+1)] = 1

            for node in parent1.DNA:
                for src in parent1.DNA[node]:
                    if random.random()<1:
                        self.splitConnection(src[0],node,src[1],src[2])
                    else:
                        self.modifyWeights(src[0],node,src[1],src[2])

        self.createArchitecture()

########################################################################################################################

    def createArchitecture(self):
        for node in self.DNA:
            if not node.startswith("inp") and len(self.DNA[node]):
                connections =[]
                weights = []
                for connection in self.DNA[node]:
                    connections.append(connection[0])
                    weights.append(connection[1])
                self.node_data[node].append(connections)
                wghts = th.cat(weights,dim=1)
                wghts = nn.Parameter(wghts)
                self.node_data[node].append(wghts)

########################################################################################################################

    def topologicalSortUtil(self,node,visited):
        visited[node] = True

        parent_depth=[0]

        parents = [ v[0] for v in self.DNA[node] if v[-1]=='f']
        for pnode in parents:
            if visited[pnode] == False:
                dep = self.topologicalSortUtil(pnode,visited)
                parent_depth.append(dep)
            else:
                parent_depth.append(self.depth[pnode])

        current_depth = max(parent_depth)+1

        self.order.append(node)
        self.depth[node]=current_depth

        return current_depth

########################################################################################################################

    def topologicalSort(self):
        visited=ddict(lambda : False)
        self.order = []
        self.depth=dict()

        nodes = list(self.DNA.keys())
        for node in nodes:
            if visited[node] == False and len(self.DNA[node])>0:
                self.topologicalSortUtil(node,visited)

        mn_inp_dep = min(self.depth.values())
        for i in range(self.input_size):
            self.depth["inp"+str(i+1)]=mn_inp_dep

        mx_out_dep = max(self.depth.values())
        for i in range(self.output_size):
            self.depth["out"+str(i+1)]=mx_out_dep


########################################################################################################################
    def makeConnection(self,cfrm=None,cto=None,weight=None,direction=None):
        if cfrm ==None:
            mx = 0
            while True:
                if mx>1000:
                    return
                mx+=1
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
                        if self.depth[cfrm]>self.depth[cto]:
                            direction = 'b'
                        if cfrm[:3] == cto[:3] and cto.startswith("out"):
                            direction = 'b'
                        if direction=='b':
                            if random.random()<0.9:
                                continue

                        break
        if weight == None:
            newweight = th.randint(-97,97,(1,1))/97
        else:
            newweight = th.Tensor(weight.tolist().copy())
        self.DNA[cto].append([cfrm,newweight,direction])
        if cto in self.remaining_outputs:
            self.remaining_outputs.remove(cto)
        if cfrm in self.remaining_inputs:
            self.remaining_inputs.remove(cfrm)
        self.topologicalSort()


########################################################################################################################

    def splitConnection(self,frm,to,weight,direction):
        if direction=='b':
            self.makeConnection(frm,to,weight,direction)
            return
        if Brain.HIDDEN_UNIT_RECORD[(frm,to)] == -1:
            newnode = "hid"+str(Brain.HIDDEN_UNIT_ID)
            Brain.HIDDEN_UNIT_ID+=1
            Brain.HIDDEN_UNIT_RECORD[(frm,to)]=newnode
        else:
            newnode = Brain.HIDDEN_UNIT_RECORD[(frm,to)]

        print(frm,newnode,to)

        self.makeConnection(frm,newnode,weight,direction)
        self.makeConnection(newnode,to,None,direction)

########################################################################################################################

    def modifyWeights(self,frm,to,weights,direction):
        weights = weights + th.randint(-97,97,weights.shape)/(97*2)
        self.makeConnection(frm,to,weights,direction)

########################################################################################################################

    

########################################################################################################################

    def evaluteNode(self,node):
        if len(self.DNA[node])>0:
            conns, weights = self.node_data[node]
            self.layer.weight = weights
            inputs = th.cat([self.Memory[con] for con in conns])
            ans = self.layer(inputs)
            self.Memory[node] = nn.ELU()(ans)
            

########################################################################################################################

    def forward(self,inp):
        if len(inp)!=self.input_size:
            return -1

        for i in range(self.input_size):
            self.Memory["inp"+str(i+1)] = th.Tensor([inp[i]])

        for node in self.order:
            if node.startswith("hid"):
                self.evaluteNode(node)

        outputs = [ node for node in self.order if node.startswith("out") ]
        outputs.reverse()

        for out in range(self.output_size):
            self.evaluteNode("out"+str(out+1))

        ans = th.cat([ self.Memory["out"+str(i+1)] for i in range(self.output_size)])
        ans = nn.Softmax(dim=0)(ans).detach().tolist()
        return ans

########################################################################################################################    

    def drawNetwork(self):
        G = nx.DiGraph()
        pos = dict()
        edge_labels = dict()
        ys = ddict(lambda : [])

        #mx_inp_depth = self.depth["inp1"]

        for node, depth in self.depth.items():
            ys[depth].append(node)

        for depth in ys:
            ys[depth].sort(key=lambda x : int(x[3:]), reverse=True)


        for depth in ys:
            for y,node in enumerate(ys[depth]):
                y = (y - (len(ys[depth])-1)/2)
                pos[node]=(depth*24, y*32)

        for node in self.DNA.keys():
            for v in self.DNA[node]:
                G.add_edge(v[0], node,
                        color = 'b' if v[-1] == 'f' and v[1].tolist()[0][0]>=0 else 'r' if v[-1] == 'f' and v[1].tolist()[0][0]<0 else 'y',
                        weight = abs(v[1].tolist()[0][0]+0.5))
                edge_labels[(v[0],node)]="%.2f"%v[1].tolist()[0][0]
        colors = nx.get_edge_attributes(G,'color').values()
        weights = nx.get_edge_attributes(G,'weight').values()

        nx.draw(G,pos,
                edge_color = colors,
                node_size=1300,
                width = list(weights),
                with_labels=True,
                connectionstyle = "arc3,rad=-0.2")
        #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels, font_size=5,label_pos=0.7)
        plt.axis('off')
        plt.show()

########################################################################################################################
if __name__ == "__main__":
    x = Brain(2,2,1)
    y = Brain(x)
    for _ in range(1):
        y = Brain(y)
    y.drawNetwork()
    x.drawNetwork()
    z = Brain(x)
    for _ in range(1):
        z = Brain(z)

    z.drawNetwork()
    x.drawNetwork()
    for _ in range(10):
        ans = y.forward([0.1,0.2])
        print(ans)
