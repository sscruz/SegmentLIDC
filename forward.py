import torch
import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# load and preprocess the files 
images,masks=torch.load( 'inputs.p')
avg=torch.nn.AvgPool2d( 2 )
images= avg(images)
masks = torch.round( avg(masks) ) 

# reshape and convert from grasycale to rgb
shape = images.shape
images = images.view( shape[0], 1, shape[1], shape[2])
images = images.repeat( (1,3,1,1) ) 

from torchvision import transforms
m, s = torch.mean(images, axis=0), torch.std(images, axis=0)
print(m.shape,s.shape)
preprocess = transforms.Compose([
    transforms.Normalize(mean=m, std=s),
])

# get the net
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32,pretrained=True)
batch_size=15
batch = images[0:batch_size,:,:,:]
pred=model( batch )
pred=torch.round( pred ).detach().numpy()

for k in range( batch.shape[0]):
     fig,ax = plt.subplots(1,1,figsize=(5,5))
     ax.imshow(batch[k,0,:,:], cmap=plt.cm.gray, alpha=0.5)

     for c in find_contours(pred[k,0,:,:].astype(float), 0.5):
         plt.plot(c[:,1], c[:,0], '--k', label='Prediction')
     plt.savefig(f"slice_pred_{k}.png")
     plt.close()


