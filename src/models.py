import torch
from torch import nn
class GraphBranch(nn.Module):
    def __init__(self,n,dim=24):
        super().__init__(); self.emb=nn.Embedding(n,dim); self.self_fc=nn.Linear(dim,dim); self.neigh=nn.Linear(dim,dim); self.norm=nn.LayerNorm(dim)
    def forward(self,edge):
        x=self.emb.weight; agg=torch.zeros_like(x); deg=torch.zeros((len(x),1),device=x.device)
        if edge.numel():
            s,t=edge[0],edge[1]; agg.index_add_(0,t,x[s]); deg.index_add_(0,t,torch.ones((len(t),1),device=x.device))
        return torch.relu(self.norm(self.self_fc(x)+self.neigh(agg/deg.clamp(min=1))))
class IntentGuardNet(nn.Module):
    def __init__(self,n,dim):
        super().__init__(); self.gru=nn.GRU(dim,48,batch_first=True,bidirectional=True)
        self.graph=GraphBranch(n); self.context=nn.Sequential(nn.Linear(dim,32),nn.ReLU(),nn.Linear(32,24),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(144,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,seq,ctx,rec,edge):
        _,h=self.gru(seq); s=torch.cat([h[-2],h[-1]],1); g=self.graph(edge)[rec]; c=self.context(ctx)
        return self.fusion(torch.cat([s,g,c],1)).squeeze(1)
