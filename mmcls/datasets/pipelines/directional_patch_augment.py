import numpy as np
import cv2
import torch
import logging
from mmcv.parallel import DataContainer as DC
from ..builder import PIPELINES
import mmcv

@PIPELINES.register_module()
class DirectionalPatchAugment:
    """Direction-aware patch augmentation for Vision Mamba.
    
    This augmentation applies different transformations to image patches based on
    scanning directions used in SS2D:
    - h (→): horizontal scanning from left to right
    - h_flip (←): horizontal scanning from right to left
    - v (↓): vertical scanning from top to bottom
    - v_flip (↑): vertical scanning from bottom to top
    
    Each direction applies a different type of transformation:
    - h: saturation adjustment
    - h_flip: contrast adjustment
    - v: brightness adjustment
    - v_flip: blur effect
    
    Args:
        patch_size (int): Size of each patch
        strength (float): Overall strength of augmentations
        visualize (bool): Whether to save visualization
        vis_dir (str): Directory to save visualizations
    """
    def __init__(self,
                 patch_size,
                 strength=0.5,
                 visualize=False,
                 vis_dir='work_dirs/directional_vis'):
        self.patch_size = patch_size
        self.strength = strength
        self.visualize = visualize
        self.vis_dir = vis_dir
        self.logger = logging.getLogger('mmcls')
        
        if self.visualize:
            mmcv.mkdir_or_exist(self.vis_dir)
            
        # 방향별 증강 속성 정의 (SS2D와 일치)
        self.directions = {
            'h': {'attr': 'saturation', 'range': (1.0, 1.0 + strength)},      # →
            'h_flip': {'attr': 'contrast', 'range': (1.0, 1.0 + strength)},   # ←
            'v': {'attr': 'brightness', 'range': (1.0, 1.0 + strength)},      # ↓
            'v_flip': {'attr': 'blur', 'range': (0, int(3 * strength))}       # ↑
        }
        
    def _get_patches(self, img, direction):
        """Get patches in specified direction."""
        H, W = img.shape[:2]
        patches = []
        positions = []
        
        if direction == 'h':  # →
            # 왼쪽에서 오른쪽으로, 위에서 아래로 순차적으로 스캔
            for h in range(0, H - self.patch_size + 1, self.patch_size):
                for w in range(0, W - self.patch_size + 1, self.patch_size):
                    patch = img[h:h+self.patch_size, w:w+self.patch_size].copy()
                    patches.append(patch)
                    positions.append((h, w))
                    
        elif direction == 'h_flip':  # ←
            # h와 동일한 순서로 스캔하되, 패치 순서를 뒤집음
            for h in range(0, H - self.patch_size + 1, self.patch_size):
                for w in range(0, W - self.patch_size + 1, self.patch_size):
                    patch = img[h:h+self.patch_size, w:w+self.patch_size].copy()
                    patches.append(patch)
                    positions.append((h, w))
            patches.reverse()
            positions.reverse()
                    
        elif direction == 'v':  # ↓
            # 열 단위로 스캔 (너비를 먼저 순회)
            for w in range(0, W - self.patch_size + 1, self.patch_size):
                for h in range(0, H - self.patch_size + 1, self.patch_size):
                    patch = img[h:h+self.patch_size, w:w+self.patch_size].copy()
                    patches.append(patch)
                    positions.append((h, w))
                    
        else:  # v_flip: ↑
            # v와 동일한 순서로 스캔하되, 패치 순서를 뒤집음
            for w in range(0, W - self.patch_size + 1, self.patch_size):
                for h in range(0, H - self.patch_size + 1, self.patch_size):
                    patch = img[h:h+self.patch_size, w:w+self.patch_size].copy()
                    patches.append(patch)
                    positions.append((h, w))
            patches.reverse()
            positions.reverse()
                    
        return patches, positions
        
    def _apply_transform(self, patch, attr, value):
        """Apply specific transformation to patch."""
        if attr == 'saturation':
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif attr == 'contrast':
            mean = np.mean(patch, axis=(0, 1), keepdims=True)
            return np.clip(mean + (patch - mean) * value, 0, 255)
            
        elif attr == 'brightness':
            return np.clip(patch * value, 0, 255)
            
        elif attr == 'blur':
            if value > 0:
                kernel_size = 2 * int(value) + 1
                return cv2.GaussianBlur(patch, (kernel_size, kernel_size), 0)
            return patch
            
    def _apply_gradual_transform(self, patches, direction):
        """Apply gradual transformation along direction."""
        n_patches = len(patches)
        if n_patches == 0:
            return patches
            
        attr = self.directions[direction]['attr']
        start_val, end_val = self.directions[direction]['range']
        
        # 선형적으로 증가하는 변화값 생성
        values = np.linspace(start_val, end_val, n_patches)
        
        transformed_patches = []
        for i, patch in enumerate(patches):
            transformed = self._apply_transform(patch, attr, values[i])
            transformed_patches.append(transformed)
            
        return transformed_patches
        
    def _visualize_augmentation(self, original_img, augmented_img, img_id):
        """시각화 저장."""
        if img_id % 100 == 0:  # 100개 단위로만 저장
            H, W = original_img.shape[:2]
            canvas = np.zeros((H * 2, W, 3), dtype=np.uint8)
            canvas[:H] = original_img
            canvas[H:] = augmented_img
            
            # 방향 표시 추가
            arrows = {
                'h': '→', 'h_flip': '←', 'v': '↓', 'v_flip': '↑'
            }
            for i, (direction, info) in enumerate(self.directions.items()):
                text = f"{arrows[direction]} {info['attr']}"
                pos = (10, H + 30 + i * 30)
                cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2)
                
            save_path = f'{self.vis_dir}/aug_{img_id:06d}.jpg'
            mmcv.imwrite(canvas, save_path)
            
    def __call__(self, results):
        """Apply directional patch augmentation."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if isinstance(img, DC):
                img = img.data
            
            original_img = img.copy() if self.visualize else None
            H, W = img.shape[:2]
            
            # 각 방향별로 증강 적용
            for direction in self.directions:
                patches, positions = self._get_patches(img, direction)
                transformed_patches = self._apply_gradual_transform(patches, direction)
                
                # 변환된 패치 복원
                for patch, (h, w) in zip(transformed_patches, positions):
                    img[h:h+self.patch_size, w:w+self.patch_size] = patch
                    
                if self.visualize:
                    # img_id를 직접 results에서 가져옴
                    img_id = results.get('cls_id', 0) * 10000 + results.get('img_id', 0)
                    self._visualize_augmentation(original_img, img, img_id)
                
            if isinstance(results[key], DC):
                img = DC(img, stack=True)
                
            results[key] = img
            
        return results
        
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(patch_size={self.patch_size}, '
        repr_str += f'strength={self.strength}, '
        repr_str += f'visualize={self.visualize})'
        return repr_str 