# Enhanced Mamba-FSCIL with VMamba Backbone and MASC-M

## 1. VMamba Backbone 도입

기존 ResNet18 기반의 백본을 VMamba로 교체하여 성능을 개선했습니다.

### VMamba 백본 상세
- **모델**: VMamba-tiny-s2l5 (ImageNet-1K 사전학습)
  - 모델명: `vmamba_tiny_s2l5` (stage=2, layer=5)
  - 입력 크기: 224×224
  - 출력 채널: [96, 192, 384, 768] (stage 1-4)
  - **사전 학습 가중치**: `vssm_tiny_0230_ckpt_epoch_262.pth`

### 구현 설정
```python
backbone=dict(
    type='VMambaBackbone',
    model_name='vmamba_tiny_s2l5',
    pretrained_path='./vssm_tiny_0230_ckpt_epoch_262.pth',
    out_indices=(0, 1, 2, 3),  # 모든 stage의 특징 추출
    frozen_stages=0,  # 모든 stage 학습 가능
    channel_first=True
)
```

## 2. MASC-M (Multi-Scale Attention Skip Connections)

기존 MAMBA-FSCIL의 3-branch 구조(identity, base, new)를 확장하여 백본의 중간 레이어 특징들을 모두 활용하는 방식을 도입했습니다.

### 핵심 아이디어
1. **다중 스케일 특징 활용**
   - Stage 1 (96채널): 저수준 특징 (텍스처, 엣지)
   - Stage 2 (192채널): 중간 수준 특징 (패턴, 부분적 형태)
   - Stage 3 (384채널): 고수준 특징 (객체 부분, 의미적 정보)
   - Stage 4 (768채널): 최종 특징 (전체적인 객체 표현)

2. **SS2D 기반 특징 처리**
   - 각 스케일의 특징을 SS2D(State Space in 2D) 블록으로 처리
   - 수평/수직 방향의 양방향 처리로 공간 정보 보존
   - 적응형 상태 공간 모델링으로 동적 특징 추출

3. **Cross-Attention 기반 특징 융합**
   - 모든 스케일의 특징을 동적으로 결합
   - 상황에 따라 각 스케일의 중요도를 자동 조절
   - 새로운 클래스 학습시 필요한 특징을 선택적으로 강조

### MASC-M 구현 상세
```python
neck=dict(
    type='MambaNeck',
    version='ss2d',
    in_channels=768,
    out_channels=768,
    feat_size=7,
    num_layers=3,
    use_residual_proj=True,
    # MASC-M 설정
    use_multi_scale_skip=True,
    multi_scale_channels=[96, 192, 384]  # stage 1-3의 채널
)
```

### 작동 방식
1. **특징 추출 단계**
   ```python
   # 각 stage에서 특징 추출
   layer1_out = backbone.layer1(x)  # [B, 96, 56, 56]
   layer2_out = backbone.layer2(x)  # [B, 192, 28, 28]
   layer3_out = backbone.layer3(x)  # [B, 384, 14, 14]
   layer4_out = backbone.layer4(x)  # [B, 768, 7, 7]
   ```

2. **특징 처리 단계**
   ```python
   # 각 스케일 특징을 SS2D로 처리
   for i, feat in enumerate(multi_scale_features):
       adapted_feat = self.multi_scale_adapters[i](feat)
       skip_features.append(adapted_feat)
   ```

3. **융합 단계**
   ```python
   # Cross-attention으로 특징 융합
   attended_features = self.cross_attention(
       query=main_feat,
       key=skip_features,
       value=skip_features
   )
   ```

4. **최종 특징 결합**
   ```python
   # 1. 모든 skip features 수집
   skip_features = [identity_proj]  # identity branch (layer4)
   if self.use_multi_scale_skip:
       for feat in multi_scale_features:  # layer1-3
           adapted_feat = self.multi_scale_adapters[i](feat)
           skip_features.append(adapted_feat)
   if self.use_new_branch:
       skip_features.append(x_new)  # new branch
   
   # 2. Cross-attention으로 가중치 계산
   attention_weights = self.attention_output(attended_features)  # [B, num_features]
   
   # 3. 가중치 적용 및 합산
   weighted_skip_features = []
   for i, feat in enumerate(skip_features):
       weight = attention_weights[:, i:i+1]  # [B, 1]
       weighted_feat = weight * feat  # [B, 1] * [B, 768]
       weighted_skip_features.append(weighted_feat)
   
   weighted_skip = sum(weighted_skip_features)  # [B, 768]
   
   # 4. 최종 출력 생성 
   final_output = x + 0.1 * weighted_skip  # [B, 768]
   ```

## 3. 모델 최적화 개선

기존 구현 대비 두 가지 주요 최적화를 통해 성능을 향상시켰습니다.

### 1. 전체 Stage 학습 활성화
```python
# 기존
frozen_stages=1  # patch embedding과 첫 번째 stage 고정

# 개선
frozen_stages=0  # 모든 stage 학습 가능
```
- **개선 효과**:
  - 저수준 특징(텍스처, 엣지)부터 재학습 가능
  - 새로운 클래스에 대한 적응력 향상
  - Multi-scale 특징과의 시너지 효과

### 2. MLP 프로젝션 강화
```python
# 기존
num_layers=2  # 2-layer MLP projection

# 개선
num_layers=3  # 3-layer MLP projection with enhanced capacity
```
- **구조 비교**:
  ```python
  # 2-layer MLP (기존)
  - Conv2d(in_channels → mid_channels)
  - LayerNorm + LeakyReLU
  - Conv2d(mid_channels → out_channels)

  # 3-layer MLP (개선)
  - Conv2d(in_channels → mid_channels)
  - LayerNorm + LeakyReLU
  - Conv2d(mid_channels → mid_channels)  # 추가된 중간 layer
  - LayerNorm + LeakyReLU
  - Conv2d(mid_channels → out_channels)
  ```
- **개선 효과**:
  - 특징 변환 능력 강화
  - 더 복잡한 패턴 학습 가능
  - Skip connection 특징들의 품질 향상

### 최적화 효과
1. **학습 안정성**
   - 전체 stage 학습으로 end-to-end 최적화
   - 더 깊은 MLP로 안정적인 특징 변환

2. **적응력 향상**
   - 저수준 특징부터 새로운 클래스에 맞춤화
   - 강화된 특징 변환으로 클래스 간 구분력 증가

3. **MASC-M과의 시너지**
   - 모든 stage의 특징이 task에 최적화
   - 향상된 특징 품질로 더 효과적인 multi-scale 융합

### 실행 방법
```bash
sh train_vmamba_fscil.sh
```