import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import torch.nn.functional as F
import numpy as np


class SA_layer(nn.Module):
    def __init__(self, dim=128, head_size=4):
        super(SA_layer, self).__init__()
        self.mha=nn.MultiheadAttention(dim, head_size)
        self.ln1=nn.LayerNorm(dim)
        self.fc1=nn.Linear(dim, dim)
        self.ac=nn.ReLU()
        self.fc2=nn.Linear(dim, dim)
        self.ln2=nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, len_size, fea_dim=x.shape
        x=torch.transpose(x,1,0)
        y,_=self.mha(x,x,x)
        x=self.ln1(x+y)
        x=torch.transpose(x,1,0)
        x=x.reshape(batch_size*len_size, fea_dim)
        x=x+self.fc2(self.ac(self.fc1(x)))
        x=x.reshape(batch_size,len_size, fea_dim)
        x=self.ln2(x)
        return x


class COOI(): # Coordinates On Original Image
    def __init__(self):
        self.stride=32
        self.cropped_size=224
        self.score_filter_size_list=[[3,3],[2,2]]
        self.score_filter_num_list=[3,3]
        self.score_nms_size_list=[[3,3],[3,3]]
        self.score_nms_padding_list=[[1,1],[1,1]]
        self.score_corresponding_patch_size_list=[[224, 224], [112, 112]]
        self.score_filter_type_size=len(self.score_filter_size_list)

    def get_coordinates(self, fm, scale):
        with torch.no_grad():
            batch_size, _, fm_height, fm_width=fm.size() 
            scale_min=torch.min(scale, axis=1, keepdim=True)[0].long() 
            scale_base=(scale-scale_min).long()//2 
            input_loc_list=[]
            fps_loc_list = []
            for type_no in range(self.score_filter_type_size): 
                score_avg=nn.functional.avg_pool2d(fm, self.score_filter_size_list[type_no], stride=1) 
                score_sum=torch.sum(score_avg, dim=1, keepdim=True) 
                _,_,score_height,score_width=score_sum.size()
                patch_height, patch_width=self.score_corresponding_patch_size_list[type_no]

                for filter_no in range(self.score_filter_num_list[type_no]):
                    score_sum_flat=score_sum.view(batch_size, -1)
                    value_max,loc_max_flat=torch.max(score_sum_flat, dim=1)
                    loc_max=torch.stack((loc_max_flat//score_width, loc_max_flat%score_width), dim=1)
                    fps_loc_list.append(loc_max)
                    top_patch=nn.functional.max_pool2d(score_sum, self.score_nms_size_list[type_no], stride=1, padding=self.score_nms_padding_list[type_no])
                    value_max=value_max.view(-1,1,1,1)
                    erase=(top_patch!=value_max).float()
                    score_sum=score_sum*erase

                    # location in the original images
                    loc_rate_h=(2*loc_max[:,0]+fm_height-score_height+1)/(2*fm_height)
                    loc_rate_w=(2*loc_max[:,1]+fm_width-score_width+1)/(2*fm_width)
                    loc_rate=torch.stack((loc_rate_h, loc_rate_w), dim=1)
                    loc_center=(scale_base+scale_min*loc_rate).long()
                    loc_top=loc_center[:,0]-patch_height//2
                    loc_bot=loc_center[:,0]+patch_height//2+patch_height%2
                    loc_lef=loc_center[:,1]-patch_width//2
                    loc_rig=loc_center[:,1]+patch_width//2+patch_width%2
                    loc_tl=torch.stack((loc_top, loc_lef), dim=1)
                    loc_br=torch.stack((loc_bot, loc_rig), dim=1)

                    # For boundary conditions
                    loc_below=loc_tl.detach().clone() # too low
                    loc_below[loc_below>0]=0
                    loc_br-=loc_below
                    loc_tl-=loc_below
                    loc_over=loc_br-scale.long() # too high
                    loc_over[loc_over<0]=0
                    loc_tl-=loc_over
                    loc_br-=loc_over
                    loc_tl[loc_tl<0]=0 # patch too large

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

            input_loc_tensor=torch.stack(input_loc_list, dim=1) # (7,6,4)

            return input_loc_tensor, fps_loc_list


class MultiLevelFusion(nn.Module):
    def __init__(self, mid_dim=128):
        super(MultiLevelFusion, self).__init__()
        self.mid_dim = mid_dim
        self.project_high=nn.Linear(2048, mid_dim)
        self.project_shallow=nn.Linear(64,mid_dim)
        self.project_middle=nn.Linear(512,mid_dim)

        self.mha_list=nn.Sequential(
                         SA_layer(mid_dim, 4),
                         SA_layer(mid_dim, 4),
                         SA_layer(mid_dim, 4)
                       )
 
    def forward(self, shallow_layers, high_layers):
        Bs, Cs, Hs, Ws = shallow_layers.size()#[16, 64, 112, 112]
        Bh, Ch, Hh, Wh = high_layers.size() #[16, 2048, 7, 7]
        shallow_layers = shallow_layers.view(Bs, Cs, -1)#[16,64,112*112]
        shallow_layers = shallow_layers.transpose(1,2)#[16,112*112,64]
        shallow_vecs = self.project_shallow(shallow_layers.reshape(-1,Cs))#[16*112*112, 128]       
        shallow_vecs = shallow_vecs.view(Bs, -1, self.mid_dim)#[16,112*112,128]
        shallow_vecs = shallow_vecs.transpose(1,2)#[16,128,112*112]
        shallow_vecs = shallow_vecs.view(Bs, self.mid_dim, Hs, Ws)#[16,128,112,112]
        shallow_patches = shallow_vecs.unfold(3, 16, 16).unfold(2, 16, 16).permute(0,1,2,3,5,4)#[16,128,7,7,16,16]
        shallow_patches = shallow_patches.reshape(Bs, self.mid_dim, 49, 256)#[16,128,49,256]
        
        high_layers = high_layers.view(Bh, Ch, -1)#[16,2048,7*7]
        high_layers = high_layers.transpose(1,2)#[16,7*7,2048]
        high_vecs = self.project_high(high_layers.reshape(-1, Ch))#[16*7*7,128]
        high_vecs = high_vecs.view(Bh, -1, self.mid_dim)#[16, 7*7, 128]
        high_vecs = high_vecs.transpose(1,2)#[16, 128, 7*7]
        high_patches = high_vecs.view(Bh, self.mid_dim, -1, 1)#[16, 128, 49, 1]

        all_patches = torch.cat((high_patches, shallow_patches), 3)#[16,128,49,257]
        all_patches = all_patches.transpose(1,2)#[16,49,128,273]
        all_patches = all_patches.reshape(Bh*49, self.mid_dim, 257)
        all_patches = all_patches.transpose(1,2)#[16*49,273,128]
        all_embedding=self.mha_list(all_patches)#[16*49, 273, 128]
        all_embedding = all_embedding[:,-1]#[16*49,128]
        fused_feature_maps = all_embedding.reshape(Bh,-1,self.mid_dim)#[16,49,128]
        fused_feature_maps = fused_feature_maps.transpose(1,2)#[16,128,49]
        fused_feature_maps = fused_feature_maps.reshape(Bh, self.mid_dim, Hh, Wh)#[16,128,7,7]
        return fused_feature_maps#[16,128,7,7]


class Patch5Model(nn.Module):
    def __init__(self):
        super(Patch5Model, self).__init__()
        self.resnet = resnet50(pretrained=True) #debug
        self.mid_dims = 128
        self.COOI=COOI()
        self.mha_list=nn.Sequential(
                        SA_layer(128, 4),
                        SA_layer(128, 4),
                        SA_layer(128, 4)
                      )
        self.fc1=nn.Linear(2048, 128)
        self.ac=nn.ReLU()
        self.fc=nn.Linear(128,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.MultiFusion = MultiLevelFusion(self.mid_dims)
    

    def forward(self, input_img, cropped_img, scale):
        x = cropped_img
        batch_size, p, _, _ =x.shape #[batch_size, 3, 224, 224]
        shallow_global_maps, high_global_maps=self.resnet(x)#fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048]
        fused_global_maps = self.MultiFusion(shallow_global_maps, high_global_maps)
        B,C,H,W = fused_global_maps.size()#[16,128,7,7]

        ## global embeddings
        global_embedding = self.avgpool(fused_global_maps)#[16, 128,1,1]
        global_embedding = global_embedding.view(global_embedding.size(0), -1)#[16,128]
        global_embedding = self.ac(global_embedding)#[16,128]
        global_embedding = global_embedding.view(-1, 1, self.mid_dims)#[16, 1, 128]


        input_loc, fps_loc=self.COOI.get_coordinates(fused_global_maps.detach(), scale)

        _,proposal_size,_=input_loc.size()

        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(fused_global_maps.device)  # [N, 4, 3, 224, 224]
        
        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t,l,b,r=input_loc[batch_no, proposal_no]
                img_patch=input_img[batch_no][:, t:b, l:r]
                _, patch_height, patch_width=img_patch.size()
                if patch_height==224 and patch_width==224:
                    window_imgs[batch_no, proposal_no]=img_patch
                else:
                    window_imgs[batch_no, proposal_no:proposal_no+1]=F.interpolate(img_patch[None,...], size=(224, 224), mode='bilinear',align_corners=True)  # [N, 4, 3, 224, 224]

        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 224, 224)  # [N*4, 3, 224, 224]
        _, local_maps=self.resnet(window_imgs.detach()) #[batchsize*self.proposalN, 2048]
        local_embedding = self.avgpool(local_maps)
        local_embedding = local_embedding.view(local_embedding.size(0), -1)#[16*4,2048,1,1]

        local_embedding=self.ac(self.fc1(local_embedding))#[batchsize*self.proposalN, 128]
        local_embedding=local_embedding.view(-1, proposal_size, 128)

        all_embeddings=torch.cat((local_embedding, global_embedding), 1)#[1, 1+self.proposalN, 128]
        all_embeddings=self.mha_list(all_embeddings)
        all_logits=self.fc(all_embeddings[:,-1])
        
        return  all_logits


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model=Patch5Model()
            if torch.cuda.device_count()>1:
                self.model=nn.DataParallel(self.model)

        if opt.continue_train:
            self.model=Patch5Model()


        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if self.isTrain and opt.continue_train:
            print(opt.loadpath)
            self.load_networks(opt.loadpath)     
            if torch.cuda.device_count()>1:
                self.model=nn.DataParallel(self.model)  

        if len(opt.gpu_ids)==0:
            self.model.to('cpu')
        else:
            self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input_img = data[0] # (batch_size, 6, 3, 224, 224)
        self.cropped_img = data[1].to(self.device)
        self.label = data[2].to(self.device).float() #(batch_size)
        self.scale = data[3].to(self.device).float()
    

    def forward(self):
        self.output = self.model(self.input_img, self.cropped_img, self.scale)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

