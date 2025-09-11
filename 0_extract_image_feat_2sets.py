import argparse
import json
# import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # debug
# import sys
# sys.path.append('./')
# import shortuuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import copy
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset 
import monai.transforms as mtf
import numpy as np
import pandas as pd
from abnormality_list import abnormality_list
import monai
# from monai.metrics import ROCAUCMetric
# from monai.data import set_track_meta
from sklearn.metrics import roc_auc_score
import torchvision
from models import i3res
# from eval_path import project_root, answer_path, model_path, model_root, data_root, text_path, data_path
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
)
def seed_everything(seed):
    '''
    setting seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(24)
def ImgABNDataset(args,  mode="train"):
    args = args
    data_root = args.data_root
    mode = mode

    if mode in ['train', 'validation', 'test']:
        with open(args.data_path, 'r') as file:
            json_file = json.load(file)
        data_list = json_file[mode]
    else:
        data_list = []
        for mode in ['train', 'validation', 'test']:
            with open(args.data_path, 'r') as file:
                json_file = json.load(file)
                data_list += json_file[mode]
    
    data_list_df = pd.DataFrame.from_dict(data_list)
    accNum_list = [d[1]['image'].split('/')[1].split('_')[0] for d in  data_list_df.iterrows()]
    image_modal_list = [d[1]['image'].split('/')[1].split('_')[1] for d in  data_list_df.iterrows()]
    data_list_df['AccessionNumber_md5'] = accNum_list
    data_list_df['img_modal'] = image_modal_list
    abn_labels_df = pd.read_csv(os.path.join('./PE_datasets/brown_vqa_v2m.csv'), usecols=['AccessionNumber_md5', ]+ abnormality_list)
    data_list_abn_labels_df = abn_labels_df[abn_labels_df['AccessionNumber_md5'].isin(pd.unique(data_list_df['AccessionNumber_md5']))]
    
    data_list_abn_labels_df = data_list_abn_labels_df.set_index('AccessionNumber_md5')
    data_list_abn_labels_df[abnormality_list] = (data_list_abn_labels_df[abnormality_list] == 'Yes').astype(int)
    data_list_abn_labels_df = data_list_abn_labels_df.reset_index(drop=False)
    data_list_df = pd.merge(data_list_df, data_list_abn_labels_df[['AccessionNumber_md5',]+abnormality_list ], on='AccessionNumber_md5')
    data_list_df['abn_labels'] = data_list_df[abnormality_list].values.tolist()
    
    data_list = []
    for idx in range(len(data_list_df)):
        d = data_list_df.iloc[idx]
        image_path = d["image"]
        image_path = image_path.replace('M3D_format', 'merlin_format')
        
        image_abs_path = os.path.join(data_root, image_path)
        image_abs_path = image_abs_path.replace('.npy', 'img.nii.gz')
        lung_abs_path = image_abs_path.replace('.npy', 'lung.nii.gz')
        data_list.append({"image": image_abs_path, "image_path": image_abs_path, "label": np.array(d['abn_labels']).astype(int)})
    
    # inspecta
    abn_labels_df_inspecta = pd.read_csv(os.path.join('./PE_datasets/inspecta_vqa_v2m.csv'), usecols=['AccessionNumber_md5', ]+ abnormality_list)
    split_inspecta = pd.read_excel(os.path.join('./PE_datasets/inspecta_split.xlsx'), index_col=0)
    if mode in ['train', 'validation', 'test']:
        set_ids = split_inspecta[split_inspecta['trts'] == mode]['impression_id'].tolist()
    else:
        set_ids = split_inspecta['impression_id'].tolist()
    set_abn_labels_df_inspecta = abn_labels_df_inspecta[abn_labels_df_inspecta['AccessionNumber_md5'].isin(set_ids)]
    
    set_abn_labels_df_inspecta = set_abn_labels_df_inspecta.set_index('AccessionNumber_md5')
    set_abn_labels_df_inspecta[abnormality_list] = (set_abn_labels_df_inspecta[abnormality_list] == 'Yes').astype(int)
    set_abn_labels_df_inspecta = set_abn_labels_df_inspecta.reset_index(drop=False)
    set_abn_labels_df_inspecta['abn_labels'] = set_abn_labels_df_inspecta[abnormality_list].values.tolist()
    inspecta_image_root = '/PE_DATA_PATH/inspecta_clean_anomalous_slices'
    
    for idx in range(len(set_abn_labels_df_inspecta)):
        d = set_abn_labels_df_inspecta.iloc[idx]
        accNum = d['AccessionNumber_md5']
        image_abs_path = os.path.join(inspecta_image_root, accNum+'.nii.gz')
        data_list.append({"image": image_abs_path, "image_path": image_abs_path, "label": np.array(d['abn_labels']).astype(int)})
    return data_list
  
train_transform = mtf.Compose(
        [
            LoadImaged(keys=["image", ], ensure_channel_first=True),
            Orientationd(keys=["image", ], axcodes="RAS"),
            Spacingd(keys=["image", ], pixdim=(1.5, 1.5, 3), mode=["bilinear", ]),
            ScaleIntensityRanged(
                keys=["image", ], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            # mtf.CropForegroundd(keys=["image", ], source_key='mask', allow_smaller=True, margin=60),
            SpatialPadd(keys=["image", ], spatial_size=[224, 224, 160]),
            CenterSpatialCropd(
                roi_size=[224, 224, 160],
                keys=["image", ],
            ),
            mtf.RandRotate90d(keys=["image", ],prob=0.9, spatial_axes=(0, 1)),
            mtf.RandScaleIntensityd(keys=["image"],factors=0.1, prob=0.9),
            mtf.RandShiftIntensityd(keys=["image"],offsets=0.1, prob=0.9),
            mtf.ToTensord(keys=["image", ], dtype=torch.float),
        ]
    )

val_transform = mtf.Compose(
        [
            LoadImaged(keys=["image", ], ensure_channel_first=True),
            Orientationd(keys=["image", ], axcodes="RAS"),
            Spacingd(keys=["image", ], pixdim=(1.5, 1.5, 3), mode=["bilinear", ]),
            ScaleIntensityRanged(
                keys=["image", ], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            SpatialPadd(keys=["image", ], spatial_size=[224, 224, 160]),
            CenterSpatialCropd(
                roi_size=[224, 224, 160],
                keys=["image", ],
            ),
            mtf.ToTensord(keys=["image", ], dtype=torch.float),
        ]
    )

class ImageClassifier(torch.nn.Module):
    def __init__(self, args, out_channels):
        super().__init__()
        resnet = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
        self.i3_resnet = i3res.I3ResNet(
            copy.deepcopy(resnet), class_nb=out_channels, conv_class=True, return_skips=True, return_pool=True # extract pooled feature
        )
        del resnet
        
    def forward(self, image):
        contrastive_features, ehr_features, skips, pooled_feat = self.i3_resnet(image)
        # out = F.sigmoid(ehr_features)
        return skips, pooled_feat


def run_model(args):
    train_root = './exps_2sets/i3dResnet_ASL_mean_sampler_2/'+ args.abn_name + '/'
    feat_root = '/media/brownradx/ssd_data0/PE_datas/PE_vlm_data' + '/image_multiscale_feat'
    checkpoint_path = train_root + "/checkpoint/best_metric_model_classification3d_dict.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    BATCHSIZE = 1
    mode = 'train_noA' # no augmentation
    feat_root = feat_root + '_' + mode
    test_list = ImgABNDataset(args, mode='train') 
    test_data = monai.data.Dataset(data=test_list, transform=val_transform)
        
    test_loader = DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=10)
    model = ImageClassifier(args, out_channels=len(abnormality_list))
    model_weights = torch.load(checkpoint_path)
    model = model.to(device, non_blocking=True)
    print(model.load_state_dict(model_weights))
    
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            img_path = batch_data['image_path'][0]
            accNum_modal = os.path.split(img_path)[-1].replace('img.nii.gz', '').replace('.nii.gz', '')
            out_img_folder = os.path.join(feat_root, accNum_modal)
            if os.path.exists(out_img_folder):
                continue
            os.makedirs(out_img_folder)
            images = batch_data['image'].float().cuda()
            labels = batch_data['label'].float().cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                skips, pooled_feat = model(images) # 2048 1024
                # outputs = torch.sigmoid(outputs)
                # feat_arrs = torch.squeeze(pooled_feat).as_tensor().detach().cpu().numpy()
                if mode == 'train':
                    save_size = int(max(1, min(BATCHSIZE, labels[0].sum()) ))
                else:
                    save_size = BATCHSIZE
                for i in range(save_size):
                    npz_name = str(i) + '_' + accNum_modal
                    # np.save(os.path.join(feat_root, npy_name), feat_arrs[i])
                    skips_dict = {}
                    # for l_i in range(1, 6):
                        # skips_dict['scale_'+str(l_i)] = torch.squeeze(skips[l_i][i]).as_tensor().detach().cpu().numpy().astype(np.half)
                    
                    for l_i in range(1, 5):
                        patch_size = int(32/2**l_i)
                        patched_feat = model.i3_resnet.avgpool(patchify_3d(torch.squeeze(skips[l_i][i]))).squeeze().permute(1, 0).reshape(-1, patch_size,patch_size,patch_size)
                        skips_dict['pooled_scale_'+str(l_i)] = patched_feat.as_tensor().detach().cpu().numpy().astype(np.float16)
                    l_i = 5
                    patched_feat = torch.squeeze(skips[l_i][i])
                    skips_dict['scale_'+str(l_i)] = patched_feat.as_tensor().detach().cpu().numpy().astype(np.float16)
                        
                    np.savez(os.path.join(feat_root, accNum_modal, npz_name), **skips_dict)

def patchify_3d(image, patch_size=(7,7,10)):
    """
    Patchify a 3D image into smaller patches.

    Args:
        image (torch.Tensor): The input 3D image tensor (C, D, H, W).
        patch_size (tuple): The size of each patch (patch_d, patch_h, patch_w).

    Returns:
        torch.Tensor: The tensor containing the patches.
    """

    patches = image.unfold(1, patch_size[0], patch_size[0])
    patches = patches.unfold(2, patch_size[1], patch_size[1])
    patches = patches.unfold(3, patch_size[2], patch_size[2])
    patches = patches.permute(0, 1, 2, 3, 4, 5, 6).contiguous()
    patches = patches.view(-1, image.shape[0], patch_size[0], patch_size[1], patch_size[2])
    return patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--abn-name", type=str, default="multiclass")
    
    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--text_path", type=str, default=text_path)
    args = parser.parse_args()
    run_model(args)
