import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from ReT_eeg import Region_Transformer as eeg_Region_Transformer
from ReT_eye import Region_Transformer as eye_Region_Transformer



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



class Attention_Fusion(nn.Module):
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
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, cross_attention=False, kv=None):
        B, N, C = x.shape

        q = self.to_q(x)
        if cross_attention:
            k =  self.to_k(kv) 
            v = self.to_v(kv)     
        else:
            k =  self.to_k(x)  
            v = self.to_v(x)          

        q=q.reshape(B, N, self.num_heads, -1).permute(0,2,1,3)
        k=k.reshape(B, N, self.num_heads, -1).permute(0,2,1,3)
        v=v.reshape(B, N, self.num_heads, -1).permute(0,2,1,3)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))

        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, eeg, eye):
        o1 = eeg @ self.weight
        o2 = eye @ self.weight
        o = torch.cat([o1, o2],dim=-1)
        alpha = self.softmax(o)
        eeg = eeg * alpha[:, :, 0].unsqueeze(2)
        eye = eye * alpha[:, :, 1].unsqueeze(2)
        out = eeg + eye
        return out
    

class Fusion_Trans_Block(nn.Module):
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
        self.norm1_1 = norm_layer(dim)
        self.attn = Attention_Fusion(
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


    def forward(self,eeg,eye): 
        x = torch.cat((eeg,eye),dim=1)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1_1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class ReLF(nn.Module):
    def __init__(self,
                 eeg_num_heads = 20,
                 eeg_embd_dim = 4,
                 fre_dim = 5,
                 eye_num_heads = 4,
                 eye_embd_dim = 20,
                 depth=6,
                 num_heads=4,
                 n_class=4,):
        super().__init__()

        assert eeg_num_heads*eeg_embd_dim == eye_num_heads*eye_embd_dim
        dim = eeg_num_heads*eeg_embd_dim
        self.eeg_Transformer = eeg_Region_Transformer(depth,num_heads=eeg_num_heads,embd_dim=eeg_embd_dim,fre_dim=fre_dim)
        self.eye_Transformer = eye_Region_Transformer(depth,num_heads=eye_num_heads,embd_dim=eye_embd_dim)

        self.fusoin0 = Fusion_Trans_Block(dim,num_heads)
        self.fusion1 = Fusion(dim)
        final_dim = dim*3
        self.cls_head = nn.Sequential(
                nn.Linear(final_dim, final_dim),
                nn.GELU(),
                nn.Linear(final_dim, n_class)
            )

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
        return {"Transformer.pos_emb","Transformer.cls_token"}
        
    def forward(self,eeg,eye):

        eegs = self.eeg_Transformer(eeg)
        eyes = self.eye_Transformer(eye)

        out0 = self.fusoin0(eegs[0],eyes[0])
        out1 = self.fusoin0(eegs[2],eyes[2])
        out2 = self.fusoin0(eegs[5],eyes[5])

        out0 = self.fusion1(out0[:,:6,:],out0[:,6:,:])
        out1 = self.fusion1(out1[:,:6,:],out1[:,6:,:])
        out2 = self.fusion1(out2[:,:6,:],out2[:,6:,:])

        out = torch.cat((out0,out1,out2),dim=-1)

        logits = self.cls_head(out.mean(1))

        return logits
        


if __name__ == '__main__':
    eeg=torch.rand(64,5,310)
    eye=torch.rand(64,5,50)
    model=ReLF().to(eeg.device)
    logits = model(eeg,eye)
    print(logits)

    # #load checkpoint for subject-dependent setting
    # checkpoint_path = Path(args.checkpoint)/f'fold_{fold}'/'checkpoint-199.pth'
    # checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    # state_dict = model.state_dict()  
    # no_pretrained_weight = []
    # for k,v in state_dict.items():
    #     if k in checkpoint['model']:
    #         state_dict[k] = checkpoint['model'][k]
    #     else:
    #         no_pretrained_weight.append(k)

    # model.load_state_dict(state_dict) 

    # print('resume checkpoint from %s'%args.checkpoint)  
    # print(no_pretrained_weight)


