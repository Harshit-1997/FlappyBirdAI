from collections import defaultdict as ddict
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Brain():

    HIDDEN_UNIT_ID=1
    HIDDEN_UNIT_RECORD=ddict(lambda : -1)
    CONNECTION_ID=1
    CONNECTION_RECORD=ddict(lambda : -1)

    def __init__(self, in_size, out_size=2, initial_connections=5):
        super().__init__()
        self.DNA = ddict(lambda : [])
        self.node_data = ddict(lambda : [])
        self.depth = dict()
        self.Memory = ddict(lambda : np.zeros((1,1)))
        self.order = []
        self.remaining_inputs = []
        self.remaining_outputs = []
        self.fitness = 0

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

            self.topologicalSort()
            self.createArchitecture()

        else:
            # Merge Parents
            parent1 = in_size
            parent2 = out_size

            if parent1.fitness<parent2.fitness:
                parent1,parent2 = parent2,parent1

            self.input_size = parent1.input_size
            self.output_size = parent1.output_size
            self.remaining_inputs = parent1.remaining_inputs
            self.remaining_outputs = parent1.remaining_outputs

            for inp in range(self.input_size):
                self.remaining_inputs.append("inp"+str(inp+1))
                self.depth["inp"+str(inp+1)] = 0

            for out in range(self.output_size):
                self.remaining_outputs.append("out"+str(out+1))
                self.DNA["out"+str(out+1)]=[]
                self.depth["out"+str(out+1)] = 1

            self.crossover(parent1.getGenome(),parent2.getGenome())
        
        #self.drawNetwork()

########################################################################################################################

    def crossover(self,parent1,parent2):
        parent1_conns = set(parent1.keys())
        parent2_conns = set(parent2.keys())
        common = parent1_conns.intersection(parent2_conns)
        exs_disj = parent1_conns.difference(parent2_conns)
        for inv in sorted(common.union(exs_disj)):
            if inv in parent2_conns:
                if random.random()<=0.6:
                    self.makeConnection(parent1[inv]['from'],
                                             parent1[inv]['to'],
                                             parent1[inv]['weight'],
                                             parent1[inv]['bias'],
                                             parent1[inv]['direction'],
                                             parent1[inv]['active'])
                else:
                    if not parent2[inv]['active'] and parent1[inv]['active']:
                        parent2[inv]['active']=True
                    self.makeConnection(parent2[inv]['from'],
                                             parent2[inv]['to'],
                                             parent2[inv]['weight'],
                                             parent2[inv]['bias'],
                                             parent2[inv]['direction'],
                                             parent2[inv]['active'])
            else:
                    self.makeConnection(parent1[inv]['from'],
                                             parent1[inv]['to'],
                                             parent1[inv]['weight'],
                                             parent1[inv]['bias'],
                                             parent1[inv]['direction'],
                                             parent1[inv]['active'])
        self.mutate()
        self.topologicalSort()
        self.createArchitecture()

########################################################################################################################

    def createArchitecture(self):
        for node in self.DNA:
            if not node.startswith("inp") and len(self.DNA[node])>0:
                connections =[]
                weights = []
                bias = 0
                for connection in self.DNA[node]:
                    if connection['active']:
                        connections.append(connection["from"])
                        weights.append(connection["weight"])
                        bias = connection['bias']
                if len(connections)>0:
                    self.node_data[node] = []
                    self.node_data[node].append(connections)
                    wghts = np.concatenate(weights,axis=1)
                    self.node_data[node].append(wghts)
                    self.node_data[node].append(bias)

########################################################################################################################

    def getGenome(self):
        genome = dict()
        for node, conns in self.DNA.items():
            for con in conns:
                genome[con['innov']] ={'from':con['from'], 
                                       'to':node, 
                                       'weight':con['weight'], 
                                       'bias':con['bias'], 
                                       'direction':con['direction'], 
                                       'active':con['active']}
        return genome

########################################################################################################################

    def topologicalSortUtil(self,node,visited):
        visited[node] = True

        parent_depth=[0]

        parents = [ v['from'] for v in self.DNA[node] if v['direction']=='f' and v['active']]
        for pnode in parents:
            if visited[pnode] == False:
                dep = self.topologicalSortUtil(pnode,visited)
                parent_depth.append(dep)
            else:
                try:
                    parent_depth.append(self.depth[pnode])
                except KeyError:
                    print(pnode,visited[pnode])


        current_depth = max(parent_depth)+1

        self.order.append(node)
        self.depth[node]=current_depth

        return current_depth

########################################################################################################################

    def topologicalSort(self):
        visited=ddict(lambda : False)
        self.order = []
        self.depth=dict()

        nodes = list( self.DNA.keys())
        for node in nodes:
            if visited[node] == False and not node in self.order:
                self.topologicalSortUtil(node,visited)

        mn_inp_dep = min(self.depth.values())
        for i in range(self.input_size):
            self.depth["inp"+str(i+1)]=mn_inp_dep

        mx_out_dep = max(self.depth.values())
        for i in range(self.output_size):
            self.depth["out"+str(i+1)]=mx_out_dep


########################################################################################################################
    def makeConnection(self,cfrm=None,cto=None,weight=None,bias=None,direction=None,active=None):
        if cfrm ==None:
            mx = 0
            while True:
                if mx>100:
                    return
                mx+=1
                a, b = random.choices(list(enumerate(self.remaining_inputs+self.order+self.remaining_outputs)),k=2)
                ifrm, cfrm = a
                ito, cto = b
                exists = False
                if ifrm != ito and not cto.startswith('inp') and (cfrm in self.order or cfrm in self.remaining_inputs):
                    exists = False
                    for v in self.DNA[cto]:
                        if v['from']==cfrm:
                            exists = True
                    if not exists:
                        direction = 'f'
                        if self.depth[cfrm]>self.depth[cto]:
                            direction = 'b'
                        if cfrm[:3] == cto[:3] and cto.startswith("out"):
                            if random.random()<0.7:
                                continue
                            direction = 'b'
                        if direction=='b':
                            if random.random()<0.1:
                                continue

                        break
        if weight == None:
            newweight = np.random.randint(-97,97,(1,1))/97
        else:
            newweight = weight.copy()

        if bias == None:
            newbias = np.random.randint(-97,97,(1,1))/97
        else:
            newbias = bias.copy()
            
        if active == None:
            active = True

        if Brain.CONNECTION_RECORD[(cfrm,cto)] == -1:
            conn = Brain.CONNECTION_ID
            Brain.CONNECTION_RECORD[(cfrm,cto)]=conn
            Brain.CONNECTION_ID+=1
        else:
            conn = Brain.CONNECTION_RECORD[(cfrm,cto)]


        self.DNA[cto].append( {'innov':conn,
                                'from':cfrm,
                                'weight':newweight,
                                'bias':newbias,
                                'direction':direction,
                                'active':active} )
        if cto in self.remaining_outputs:
            self.remaining_outputs.remove(cto)
        if cfrm in self.remaining_inputs:
            self.remaining_inputs.remove(cfrm)
        self.topologicalSort()

########################################################################################################################

    def modifyWeights(self,frm,to):
        for conn in self.DNA[to]:
            if conn['from']==frm and conn['active']:
                if random.random()>0.98:
                    conn['weight']+=np.random.randint(-97,97,conn['weight'].shape)/(97*4)
                    conn['bias']+=np.random.randint(-97,97,conn['bias'].shape)/(97*4)/10
                else:
                    conn['weight']=np.random.randint(-97,97,conn['weight'].shape)/(97)
                    conn['bias']=np.random.randint(-97,97,conn['bias'].shape)/(97)/10

########################################################################################################################

    def splitConnection(self,frm,to):
        for conn in self.DNA[to]:
            if conn['from']==frm:
                if conn['direction']=='b' or not conn['active']:
                    return
                else:
                    if Brain.HIDDEN_UNIT_RECORD[(frm,to)] == -1:
                        newnode = "hid"+str(Brain.HIDDEN_UNIT_ID)
                        Brain.HIDDEN_UNIT_ID+=1
                        Brain.HIDDEN_UNIT_RECORD[(frm,to)]=newnode
                    else:
                        newnode = Brain.HIDDEN_UNIT_RECORD[(frm,to)]
                    
                    conn['active'] = False
                    self.makeConnection(frm,newnode,conn['weight'],conn['bias'],conn['direction'],True)
                    self.makeConnection(newnode,to,None,None,conn['direction'],True)
                    #print(to,self.DNA[to])

########################################################################################################################

    def mutate(self):
        nodes = list(self.DNA.keys())
        for to in nodes:
            for conn in self.DNA[to]:
                if conn['active']:
                    if random.random()<0.3:
                        if random.random()<0.9:
                            self.modifyWeights(conn['from'],to)
                        elif random.random() < 0.3:
                            self.splitConnection(conn['from'],to)
                        else:
                            self.makeConnection()
                else:
                    if random.random()<0.01:
                        conn['active']=True

########################################################################################################################

    def evaluteNode(self,node):
        if len(self.DNA[node])>0:
                conns, weights, bias = self.node_data[node]
                inputs = np.concatenate([self.Memory[con] for con in conns])
                ans = weights@inputs + bias
                ans[ans<0]=0
                self.Memory[node] = ans

########################################################################################################################

    def softmax(self,x):
        x = np.exp(x)
        ans = [ i/sum(x) for i in x]
        return ans

########################################################################################################################

    def forward(self,inp,mode):
        if len(inp)!=self.input_size:
            return -1

        for i in range(self.input_size):
            self.Memory["inp"+str(i+1)] = np.array([inp[i]]).reshape((1,1))

        for node in self.order:
            self.evaluteNode(node)

        ans = np.concatenate([ self.Memory["out"+str(i+1)] for i in range(self.output_size)])
        if mode == 's':
            ans = self.softmax(ans)
            return np.argmax(ans).tolist()
        else:
            ans = ans.detach()
            return ans.tolist()

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
                G.add_edge(v['from'], node,
                        color = 'b' if v['direction'] == 'f' and v['weight'].tolist()[0][0]>=0 else 'r' if v['direction'] == 'f' and v['weight'].tolist()[0][0]<0 else 'y',
                        weight = abs(v['weight'].tolist()[0][0]+0.5))
                edge_labels[(v['from'],node)]="%.2f"%v['weight'].tolist()[0][0]
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
    x = Brain(6,6,15)
    y = Brain(6,6,15)
    x.drawNetwork()
    y.drawNetwork()
    for _ in range(10):
        x = Brain(x,y)
        x.drawNetwork()
