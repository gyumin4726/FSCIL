import os
import sys
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from mmcls.datasets import build_dataset
from mmcls.datasets.pipelines.directional_patch_augment import DirectionalPatchAugment
from mmfscil.datasets.cub import CUBFSCILDataset

def preprocess_and_save_augmented_images(config_file, output_dir):
    """증강된 이미지를 미리 생성하고 저장합니다."""
    # 설정 파일 로드
    cfg = mmcv.Config.fromfile(config_file)
    
    # 전처리용 파이프라인 정의
    preprocess_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=(256, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='DirectionalPatchAugment',
             patch_size=4,
             strength=0.5,
             visualize=False)
    ]
    
    # 데이터셋 설정 수정
    cfg.data.train.dataset.pipeline = preprocess_pipeline
    
    # 데이터셋 생성
    dataset = build_dataset(cfg.data.train.dataset)
    
    # 데이터셋 정보 출력
    print(f'Total number of images in dataset: {len(dataset)}')
    
    
    # 출력 디렉토리 생성
    mmcv.mkdir_or_exist(output_dir)
    
    # 각 이미지에 대해 증강 적용
    for idx in range(len(dataset)):
        # 이미지 로드 및 증강
        results = dataset[idx]
        
        # 증강된 이미지 저장
        img = results['img']
        if isinstance(img, DC):
            img = img.data
            
        # 원본 이미지 경로에서 클래스 정보 추출
        original_path = results['img_info']['filename']
        class_name = os.path.basename(os.path.dirname(original_path))
        image_name = os.path.basename(original_path)
        
        # 클래스별 디렉토리 생성
        class_dir = os.path.join(output_dir, class_name)
        mmcv.mkdir_or_exist(class_dir)
        
        # 저장 경로 설정 (원본과 동일한 이름으로 저장)
        save_path = os.path.join(class_dir, image_name)
        
        # 이미지 저장
        mmcv.imwrite(img, save_path)
        
        if idx % 100 == 0:
            print(f'Processed {idx}/{len(dataset)} images')

if __name__ == '__main__':
    config_file = 'configs/cub/resnet18_etf_bs512_80e_cub_mambafscil.py'
    output_dir = 'data/CUB_200_2011/augmented_images_1'
    preprocess_and_save_augmented_images(config_file, output_dir) 