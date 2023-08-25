import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import pdb
import sys

sys.path.append("./")
from utils.config import *


class DFDCDetection(data.Dataset):
    def __init__(self, root, train=True, frame_nums=5, 
                transform=None, target_transform=None, dataset_name='DFDC',Deepfakes_path ='/Celeb-synthesis-mtcnn/*',
        split_path = '/List_of_testing_videos.txt'):
        self.root = root
        self.split_path = split_path
        self.train = train
        self.frame_nums = frame_nums
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.datas = self._split_data()


    def __getitem__(self, index):
        img_path, target, video_fn = self.datas[index]

        img = Image.open(img_path[0])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, video_fn
       
    def __len__(self):
        return len(self.datas)

    def _sampler(self, sample_datas):
        datas = []
        for fn, label, folder in sample_datas:
            if type(fn) != str:  # 会有nan
                continue
            face_paths = glob.glob(os.path.join(self.root, folder, fn.split('.')[0], '*.jpg'))
            if len(face_paths) > self.frame_nums:
                face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0])))#排序
                ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)  # 生成均匀分布的样本
                face_paths = face_paths[ind]
            if self.train:
                datas.extend([[face_path, label, fn] for face_path in face_paths])
            else:
                datas.append([face_paths, label, fn])
        return datas

    def _split_data(self, test_pos_size=2000, seed=0):
        datas = []
        
        data_root = self.root
        with open(self.split_path,'r') as f:
            self.raw_list = f.read().splitlines()
        test_list = [x.split(" ")[0] for x in self.raw_list]
        labels = [x.split(" ")[1] for x in self.raw_list]
        test_list = [data_root+'/'+x.split("/")[1][:-4] for x in test_list]
        self.fake_num = 0
        self.real_num = 0
        if self.train == False:
            
            for path in test_list:
              
                label_str = path.split('/')[5]
                
                label = 1 if len(label_str) >13 else 0
                
                face_paths = glob.glob(os.path.join(path, '*.png'))

                if len(face_paths) < 5:
                    continue
                if len(face_paths) > self.frame_nums:
                    face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0][-4:])))
                    ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)
                    face_paths = face_paths[ind]
                
                    #count real/fake frames 
                    for i in range(len(face_paths)):
                        if label == 1:
                                self.fake_num = self.fake_num+1
                        else:
                            self.real_num = self.real_num+1
                    

                datas.extend([[face_path, label, path] for face_path in face_paths])
        # test_list = [x.split(" ")[1] for x in self.raw_list]
        # test_list = [data_root+'/'+x.split("/")[1][:-4] for x in test_list]
        # ## load dataset metadata
        # metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), low_memory=False)
        # metadata = metadata.set_index('filename', drop=False)
        # metadata = 
        # ## filter out videos without images
        # tmp = self.root + "/*/*"
        # # video_paths = glob.glob(str(self.root/'*'/'*'))
        # video_paths = glob.glob(tmp)
        # vn = [os.path.basename(x) + '.mp4' for x in video_paths if len(os.listdir(x)) > 0]
        # metadata = metadata.loc[vn]  
        # ## random permutation
        # metadata['label'] = metadata['label'].map({'FAKE': 1, 'REAL': 0})
        # metadata = metadata[['filename', 'label', 'original', 'folder']]
        # metadata = metadata.sample(frac=1, random_state=seed) 

        # reals = metadata[metadata['original'].eq('NAN')].drop('original', axis=1)

        # fakes = metadata.drop(reals.filename).set_index('original')
        # if self.train:
        #     train_pos = reals[test_pos_size:] 
        #     train_pos = train_pos[train_pos.filename.isin(fakes.index)] 

        #     train_neg = fakes.loc[train_pos.filename]
        #     datas = self._sampler_new(train_pos,train_neg)
        # else:
        #     test_pos = reals[:test_pos_size]
        #     test_neg = fakes.loc[test_pos.filename].groupby(level=0, group_keys=False).apply(
        #         lambda x: x.sample(1, random_state=seed))
        #     test_datas = np.concatenate([test_pos.values, test_neg.values])

        #     datas = self._sampler(test_datas)
        return datas


if __name__ == '__main__':
    
    from transform import get_transform

    # trans = get_transform(input_size)['train']
    # train_dataset = DFDCDetection(train=True, frame_nums=5, transform=trans)
    # train_dataloader = data.DataLoader(train_dataset, batch_size=train_size, shuffle=True, num_workers=4)

    trans =get_transform(input_size)['test']
    test_dataset = DFDCDetection(train=False, frame_nums=12, transform=trans,root=dfdc_path,split_path=dfdc_data_list)
    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    print(f"test dataset real :{test_dataset.real_num},fake :{test_dataset.fake_num}")

    print(len(test_dataset))
    for i, datas in enumerate(test_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        print(datas[1])