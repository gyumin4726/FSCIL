import os
import sys
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmcv.image import imrotate
from PIL import Image

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from mmcls.datasets import build_dataset
from mmfscil.datasets.cub import CUBFSCILDataset

def ensure_rgb(img):
    # img가 (H, W, 3)이고, BGR이라면 RGB로 변환
    if img.shape[2] == 3:
        # BGR to RGB
        return img[..., ::-1]
    return img

def rotate_and_crop_resize(img, angle, crop_size, resize_size):
    img = ensure_rgb(img)
    pil_img = Image.fromarray(img)
    rotated = pil_img.rotate(angle, expand=True, fillcolor=(255,255,255))
    width, height = rotated.size
    # CenterCrop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    cropped = rotated.crop((left, top, right, bottom))
    # Resize
    resized = cropped.resize((resize_size, resize_size))
    return resized

def preprocess_and_save_augmented_images(config_file, output_dir):
    """이미지를 회전시키고 저장합니다."""
    # 설정 파일 로드
    cfg = mmcv.Config.fromfile(config_file)
    
    # 원본 이미지 기준으로 파이프라인 설정
    angles = [45]
    preprocess_pipeline = [
        dict(type='LoadImageFromFile')
    ]
    cfg.data.train.dataset.pipeline = preprocess_pipeline
    cfg.data.train.dataset.data_prefix = './data/CUB_200_2011/'  # 원본 이미지 경로
    
    # 데이터셋 생성
    dataset = build_dataset(cfg.data.train.dataset)
    
    # 데이터셋 정보 출력
    print(f'Total number of images in dataset: {len(dataset)}')
    
    # 출력 디렉토리 생성
    mmcv.mkdir_or_exist(output_dir)
    
    # config에서 crop_size, resize_size 가져오기
    crop_size = 224  # 기본값, 필요시 config에서 읽기
    resize_size = 256  # 기본값, 필요시 config에서 읽기

    # 각 이미지에 대해 회전 적용
    for idx in range(len(dataset)):
        # 이미지 로드
        results = dataset[idx]
        img = results['img']
        if isinstance(img, DC):
            img = img.data
        # 원본 이미지 경로에서 클래스 정보 추출
        original_path = results['img_info']['filename']
        class_name = os.path.basename(os.path.dirname(original_path))
        image_name = os.path.basename(original_path)
        image_name_without_ext = os.path.splitext(image_name)[0]
        image_ext = os.path.splitext(image_name)[1]
        # 클래스별 디렉토리 생성
        class_dir = os.path.join(output_dir, class_name)
        mmcv.mkdir_or_exist(class_dir)
        # 각도별로 이미지 저장
        for angle in angles:
            rotated_image_name = f"{image_name_without_ext}_rot{angle}{image_ext}"
            save_path = os.path.join(class_dir, rotated_image_name)
            rotated_img = rotate_and_crop_resize(img, angle, crop_size, resize_size)
            rotated_img.save(save_path)
        if idx % 100 == 0:
            print(f'Processed {idx}/{len(dataset)} images')

if __name__ == '__main__':
    config_file = 'configs/cub/resnet18_etf_bs512_80e_cub_mambafscil.py'
    output_dir = 'data/CUB_200_2011/rotated_images'
    preprocess_and_save_augmented_images(config_file, output_dir) 