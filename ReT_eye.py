import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange


EYE_4region_index = [[0,1,2,3,4,5,6,7,8,9,10,11],   #Pupil diameter 
                      [12,13,14,15,16,17,18,19,20,21,22,23,42,43,44,45],  #Pupil dispersion
                      [24,25,26,27,28,29,],  #Fixation duration 
                      [30,31,32,33,34,35,36,37,38,39,40,41,46,47,48,49], #Saccades
                      ]

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.to_k = nn.Linear(self.head_dim,self.head_dim*num_heads)
        self.to_v = nn.Linear(self.head_dim,self.head_dim*num_heads)
        self.to_q = nn.Linear(self.head_dim,self.head_dim*num_heads)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_head_dim(self,x,batch):
        #x:(b t) h (d*h)
        out=[]
        for i in range(self.num_heads):
            out.append(x[:,i,i*self.head_dim:(i+1)*self.head_dim]) #(b t),d
        out = torch.stack(out,dim=1).to(x.device) #(b t),h,d
        out = rearrange(out,'(b t) h d -> b h t d',b=batch)
        return out

    def forward(self, x,uni_causal_mask=None):
        B, T,E = x.shape
        x = rearrange(x,'b t (h d) -> (b t) h d',h=self.num_heads)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.get_head_dim(q,B) #b h t d

        k = self.get_head_dim(k,B)
        v = self.get_head_dim(v,B)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        if uni_causal_mask is not None:
            attn = attn + uni_causal_mask
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, E)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale_init_values=0.1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0


    def forward(self, x,uni_causal_mask=None):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x),uni_causal_mask))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    
class Region_Transformer(nn.Module):
    def __init__(self,depth,num_heads=4,embd_dim=20,fre_dim=5):
        super().__init__()
        self.EYE_4region_index = EYE_4region_index
        self.prj = nn.ModuleList([nn.Linear(12,embd_dim),
                                  nn.Linear(16,embd_dim),
                                  nn.Linear(6,embd_dim),
                                  nn.Linear(16,embd_dim),])
        dim = embd_dim*num_heads
        self.type_emb = nn.Parameter(torch.zeros(1, 1, dim)) 
        self.pos_emb = nn.Parameter(torch.zeros(1, 6, dim))  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                )
                for i in range(depth)
            ]
        )
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_emb, std=0.02)
        trunc_normal_(self.type_emb, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_emb","cls_token"}


    def forward(self,x,uni_causal_mask=None):
        B,T,E=x.shape #b,t,50
        x_trans = []
        for idx, region_index in enumerate(self.EYE_4region_index):
            region_index = torch.tensor(region_index, device=x.device)
            temp_x = torch.index_select(x, 2, region_index).to(x.device)
            temp_x = self.prj[idx](temp_x)
            x_trans.append(temp_x)
        x = torch.cat(x_trans,dim=-1).to(x.device)
        output= []
        cls_tokens = self.cls_token.expand(x.size()[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb.expand(x.size()[0], -1, -1)+self.type_emb.expand(x.size()[0], x.size()[1], -1) 
        for blk in self.blocks:
            x = blk(x,uni_causal_mask)
            output.append(x)
        return output
