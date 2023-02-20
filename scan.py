import pylidc as pl
from pylidc.utils import consensus
import numpy as np 
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import torch

scan=pl.query(pl.Scan).filter(pl.Scan.patient_id=="LIDC-IDRI-0078").first()
vol = scan.to_volume()
mask=np.zeros_like(vol)
for ann_clust in scan.cluster_annotations():
    cmask,cbbox,_ = consensus(ann_clust, clevel=0.5,
                          pad=[(20,20), (20,20), (0,0)])
    mask[cbbox]+=cmask

    
# for k in range(vol.shape[2]):
#     fig,ax = plt.subplots(1,1,figsize=(5,5))
#     ax.imshow(vol[:,:,k], cmap=plt.cm.gray, alpha=0.5)
#     for c in find_contours(mask[:,:,k].astype(float), 0.5):
#         plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')
        
#     plt.savefig(f"slice_{k}.png")
#     plt.close()
print(np.max(vol), np.min(vol))
vol  = np.transpose( vol , [2,0,1]).astype(np.float32) # each row will be a slice
mask = np.transpose( mask, [2,0,1]).astype(np.float32) # each row will be a slice

t_vol = torch.from_numpy( vol    )
t_mask= torch.from_numpy( mask )

torch.save( (t_vol, t_mask), "inputs.p")
