import torch
import os
import numpy as np
from unet import *
from ct_dataset import *
from functools import partial
from tqdm import tqdm
from torchvision import transforms



gpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")


#dataset  bx2cxhxw
traindata=ctDataset(csv=r'/home/ps/tmp/pycharm_project_773/LIGN-master/LIGN-master/mayo_2021_abdomen_test.csv',transform=transforms.Resize([256,256]))
train_loader = DataLoader(dataset=traindata, batch_size=1, shuffle=True,num_workers=0)



#model
model=UNet( in_channel=1,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=[],
        res_blocks=1,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128).to(device)
class MSConv1(nn.Module):
    def __init__(self, planes, ratio=0.25):
        super(MSConv1, self).__init__()
        out_planes = int(planes * ratio)
        self.conv1 = nn.Conv2d(planes, planes-out_planes, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(planes, out_planes, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x)], dim=1)
class MSRED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(MSRED_CNN, self).__init__()
        #one
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = MSConv1(out_ch)
        self.conv3 = MSConv1(out_ch)
        self.conv4 = MSConv1(out_ch)
        self.conv5 = MSConv1(out_ch)
        #two
        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()



    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out



model.load_state_dict(torch.load(r'/home/ps/tmp/pycharm_project_773/RDM2/abdomen_last.pt'))
model_stage2= MSRED_CNN().to(device)
model_stage2.load_state_dict(torch.load(r'/home/ps/tmp/pycharm_project_773/RDM2/stage2_abdomen_medcnn_last.pt'))


def x_t(x_T,x_0,t,T):
        x_t=(1-(t/T))*x_0+(t/T)*x_T
        return x_t


@torch.no_grad()
def sample(nimg,T):
    x_tt=nimg
    timestep_seq = np.asarray(list(range(0, T)))
    print(nimg.shape)
    [b,c,h,w]=nimg.shape
    for i in tqdm(reversed(range(0, T)), desc='sampling loop time step', total=T):
        t = torch.full((b,), timestep_seq[i], device=device, dtype=torch.long)

        x_ts=x_tt-(1/T)*(nimg-model(x_tt.to(device),t.to(device)))
        x_tt=x_ts
        torch.cuda.empty_cache()

    return x_tt


T=100


#singlesample
#25 dose
cleanimage=read_dcm('/home/ps/tmp/pycharm_project_773/mayo_2021_abdomen/LDCT-and-Projection-data/L299/12-23-2021-NA-NA-31304/1.000000-Full Dose Images-29696/1-059.dcm')
image=read_dcm('/home/ps/tmp/pycharm_project_773/mayo_2021_abdomen/LDCT-and-Projection-data/L299/12-23-2021-NA-NA-31304/1.000000-Low Dose Images-20326/1-059.dcm')
#10 dose
# image=read_dcm('/home/ps/tmp/pycharm_project_773/mayo_2021_chest/LDCT-and-Projection-data/C296/1.000000-Low Dose Images-11387/1-055.dcm')
# cleanimage=read_dcm('/home/ps/tmp/pycharm_project_773/mayo_2021_chest/LDCT-and-Projection-data/C296/1.000000-Full Dose Images-05724/1-055.dcm')
nimg=image
cleanimage=torch.tensor(cleanimage).type(torch.FloatTensor)
cleanimage=transforms.Resize([256,256])(cleanimage)
image=torch.tensor(image).type(torch.FloatTensor)
image=transforms.Resize([256,256])(image)
image=torch.unsqueeze(image, dim=0)
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pm import *
model.eval()
model_stage2.eval()
#model_plot_img('/home/ps/tmp/pycharm_project_773/mayo_2021_chest/LDCT-and-Projection-data/C002/1.000000-Low Dose Images-39882/1-001.dcm',1500,-450)
nimg=image
x_start = nimg.to(device)
x_out=sample(x_start,T)
x_out=model_stage2(x_out)
print(x_out.size())
im_orig=x_out.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
# plt.imshow(im_orig,cmap='gray')
# plt.axis('off')
# plt.savefig('paperimg3')



im_0=nimg.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
plt.imshow(im_0, cmap='gray')
plt.axis('off')
plt.savefig('abdomen_10_LDCT',bbox_inches='tight', pad_inches=0)
plt.show()

cleanimage=cleanimage.squeeze(dim=0).cpu().numpy()
plt.imshow(cleanimage, cmap='gray')
plt.axis('off')
plt.savefig('abdomen_10_NDCT',bbox_inches='tight', pad_inches=0)
plt.show()

# im_orig[im_orig<-1000]=-100
# im_orig[im_orig>1000]=200
# cleanimage[cleanimage<-1000]=-100
# cleanimage[cleanimage>1000]=200
im_orig=ImageRescale(im_orig,[0,1])
im_0=ImageRescale(im_0,[0,1])
cleanimage=ImageRescale(cleanimage,[0,1])
print(im_orig.max())

plt.imshow(im_orig, cmap='gray')
plt.axis('off')
plt.savefig('abdomen_10_RDM2',bbox_inches='tight', pad_inches=0)
plt.show()




to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
def denormalize_(image,norm_range_min,norm_range_max):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image


def trunc(mat,trunc_min,trunc_max):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    return mat


from measure import *
values = range(len(train_loader))
ssim=[]
psnr=[]
rmse=[]
with tqdm(total=len(values)) as pbar:
        for step, img in enumerate(train_loader):

            model.eval()
            model_stage2.eval()
            nimg, cimg = img
            x_out=sample(nimg.to(device),T)
            x_out=model_stage2(x_out)

            x_out=x_out.detach().squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            cimg=cimg.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            nimg=nimg.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
        
            x_out=ImageRescale(x_out, [0, 1])
            cimg=ImageRescale(cimg, [0, 1])
            p,s,m=pm(x_out,cimg)
            psnr.append(p)
            ssim.append(s)
            rmse.append(m)

            pbar.update(1)

print("psnr:", sum(psnr) / len(psnr))
print("ssim:",sum(ssim)/len(ssim))
print("rmse:",sum(rmse)/len(rmse))

