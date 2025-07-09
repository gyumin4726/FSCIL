# Skip Connections Architecture Diagrams

## 1. ResNet BasicBlock Skip Connection

```
ResNet BasicBlock Architecture
══════════════════════════════

Input Feature Map (H×W×C)
        │
        ├─────────────────────────┐ Identity Path
        │                         │
        ▼                         │
┌─────────────────┐               │
│   Conv2d 3×3    │               │
│   stride=stride │               │
│   padding=1     │               │
└─────────────────┘               │
        │                         │
        ▼                         │
┌─────────────────┐               │
│   BatchNorm2d   │               │
└─────────────────┘               │
        │                         │
        ▼                         │
┌─────────────────┐               │
│      ReLU       │               │
└─────────────────┘               │
        │                         │
        ▼                         │
┌─────────────────┐               │
│   Conv2d 3×3    │               │
│   stride=1      │               │
│   padding=1     │               │
└─────────────────┘               │
        │                         │
        ▼                         │
┌─────────────────┐               │
│   BatchNorm2d   │               │
└─────────────────┘               │
        │                         │
        ▼                         ▼
┌─────────────────────────────────────┐
│              ADD (+)                │ ← Skip Connection
│        out += shortcut(x)           │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│      ReLU       │
└─────────────────┘
        │
        ▼
Output Feature Map (H'×W'×C')

Note: shortcut(x) = x (if same dimensions) 
      or Conv2d(1×1) + BatchNorm (if different dimensions)
```

## 2. MambaNeck Residual Connection (MASC-M 이전)

```
MambaNeck Residual Connection Architecture
═════════════════════════════════════════

Input Feature Map (B×512×7×7)
        │
        ├─────────────────────────────────────┐ Identity Path
        │                                     │
        ▼                                     ▼
┌─────────────────┐                 ┌─────────────────┐
│   MLP Projection│                 │  AdaptiveAvgPool│
│                 │                 │     (1×1)       │
│ Conv2d 1×1      │                 └─────────────────┘
│ LayerNorm       │                          │
│ LeakyReLU       │                          ▼
│ Conv2d 1×1      │                 ┌─────────────────┐
│ (final, no act) │                 │ Residual Proj   │
│ (num_layers=2)  │                 │ Linear(512→1024)│
└─────────────────┘                 └─────────────────┘
        │                                    │
        ▼                                    │
┌─────────────────┐                          │
│ Positional Embed│                          │
│     (7×7)       │                          │
└─────────────────┘                          │
        │                                    │
        ▼                                    │
┌─────────────────┐                          │
│   Mamba/SS2D    │                          │
│   Processing    │                          │
│                 │                          │
│ 4-direction     │                          │
│ scanning:       │                          │
│ h, h_flip,      │                          │
│ v, v_flip       │                          │
└─────────────────┘                          │
        │                                    │
        ▼                                    │
┌─────────────────┐                          │
│  AdaptiveAvgPool│                          │
│     (1×1)       │                          │
└─────────────────┘                          │
        │                                    │
        ▼                                    ▼
┌─────────────────────────────────────────────────┐
│                  ADD (+)                        │ ← Skip Connection
│         final_output = x + identity_proj        │
│                                                 │
│  점진적 학습 시 (MASC-M 이전):                     │
│  final_output = x + identity_proj + x_new       │
└─────────────────────────────────────────────────┘
        │
        ▼
Output Features (B×1024)
```

## 3. MLPFFNNeck Final Residual Connection

```
MLPFFNNeck Final Residual Connection Architecture
════════════════════════════════════════════════

Input Features (B×C×H×W)  # 예: (B×1024×1×1)
        │
        ▼
┌─────────────────┐
│ AdaptiveAvgPool │
│     (1×1)       │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Flatten: view() │
│ (B×C×1×1)→(B×C) │
└─────────────────┘
        │
        ├─────────────────────────────────────┐ Identity Path
        │                                     │
        ▼                                     │
┌─────────────────┐                          │
│   Main Branch   │                          │
│                 │                          │
│ ┌─────────────┐ │                          │
│ │ ln1: Linear │ │                          │
│ │ (in→in*2)   │ │                          │
│ │ LayerNorm   │ │                          │
│ │ LeakyReLU   │ │                          │
│ └─────────────┘ │                          │
│        │        │                          │
│        ▼        │                          │
│ ┌─────────────┐ │ (if num_layers==3)       │
│ │ ln2: Linear │ │                          │
│ │ (in*2→in*2) │ │                          │
│ │ LayerNorm   │ │                          │
│ │ LeakyReLU   │ │                          │
│ └─────────────┘ │                          │
│        │        │                          │
│        ▼        │                          │
│ ┌─────────────┐ │                          │
│ │ ln3: Linear │ │                          │
│ │ (in*2→out)  │ │                          │
│ │ (no norm/act)│ │                          │
│ └─────────────┘ │                          │
└─────────────────┘                          │
        │                                    │
        ▼                                    ▼
┌─────────────────┐                 ┌─────────────────┐
│ Main Output     │                 │   FFN Branch    │
│      (x)        │                 │                 │
└─────────────────┘                 │ ┌─────────────┐ │
        │                           │ │   Linear    │ │
        │                           │ │ (in→out)    │ │
        │                           │ │ bias=False  │ │
        │                           │ └─────────────┘ │
        │                           └─────────────────┘
        │                                    │
        ▼                                    ▼
┌─────────────────────────────────────────────────┐
│                  ADD (+)                        │ ← Skip Connection
│    if self.use_final_residual:                  │
│        x = x + identity                         │
│    (identity는 FFN으로 처리된 원본 입력)           │
└─────────────────────────────────────────────────┘
        │
        ▼
Final Output Features (B×out_channels)

Note: num_layers=2일 때는 ln2 생략 (ln1 → ln3)
      num_layers=1일 때는 ln1만 사용
```

## MASC-M Enhanced Architecture (현재)

```
MASC-M Enhanced Skip Connection Architecture
═══════════════════════════════════════════

Input Image (224×224×3)
        │
        ▼
┌─────────────────────────────────────────────────┐
│                ResNet-18 Backbone                │
│                                                 │
│ Conv2D: 7×7, stride=2, padding=3 → 112×112×64   │
│ MaxPool2D: 3×3, stride=2, padding=1 → 56×56×64  │
│                                                 │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────┐ │
│ │ Layer1  │  │ Layer2  │  │ Layer3  │  │Layer4│ │
│ │ (64ch)  │  │ (128ch) │  │ (256ch) │  │(512ch│ │
│ │ 56×56   │  │ 28×28   │  │ 14×14   │  │ 7×7) │ │
│ └─────────┘  └─────────┘  └─────────┘  └──────┘ │
│      │            │            │           │    │
└──────┼────────────┼────────────┼───────────┼────┘
       │            │            │           │
       │            │            │           ▼
       │            │            │    ┌─────────────┐
       │            │            │    │ MambaNeck   │
       │            │            │    │ Main Input  │
       │            │            │    │ (x)         │
       │            │            │    └─────────────┘
       │            │            │           │
       ▼            ▼            ▼           │
┌─────────────────────────────────────────────────┐
│              Multi-Scale Adapters               │
│              (Layer1, 2, 3 only)               │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│ │Adapter1 │  │Adapter2 │  │Adapter3 │          │
│ │64→1024  │  │128→1024 │  │256→1024 │          │
│ └─────────┘  └─────────┘  └─────────┘          │
│      │            │            │               │
└──────┼────────────┼────────────┼───────────────┘
       │            │            │           │
       ▼            ▼            ▼           │
┌─────────────────────────────────────────────────┐
│              Skip Features Collection           │
│                                                 │
│    [Identity_Proj(Layer4), MS_Feat1(Layer1),   │
│     MS_Feat2(Layer2), MS_Feat3(Layer3),        │
│     New_Branch*]                               │
│                                                 │
│ • Identity_Proj ← Layer4 (main input)           │
│ • MS_Feat1-3 ← Layer1-3 (multi-scale adapters) │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│           FSCIL-ASAF Attention Module           │
│                                                 │
│ Input: SSM_Output (1024ch)                      │
│        │                                        │
│        ▼                                        │
│ ┌─────────────────┐                             │
│ │ Linear(1024→N)  │  N = num_skip_sources       │
│ └─────────────────┘                             │
│        │                                        │
│        ▼                                        │
│ ┌─────────────────┐                             │
│ │   Softmax       │                             │
│ └─────────────────┘                             │
│        │                                        │
│        ▼                                        │
│ [w1, w2, w3, w4, w5*] ← Attention Weights       │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│            Weighted Skip Fusion                 │ ← Enhanced Skip Connection
│                                                 │
│ skip_weights = skip_attention(x)  # Softmax     │
│ weighted_skip = Σ(wi × skip_feati)              │
│ final_output = x + weighted_skip                │
│                                                 │
│ Skip Features: [identity_proj, ms_feat1,        │
│                ms_feat2, ms_feat3, x_new*]      │
└─────────────────────────────────────────────────┘
       │
       ▼
Final Enhanced Features (1024ch)
```

## Skip Connection Evolution Timeline

```
Skip Connection Evolution in Mamba-FSCIL
════════════════════════════════════════

Phase 1: Basic Skip Connections
┌─────────────────────────────────────────┐
│ ResNet BasicBlock: out += shortcut(x)   │
│ MambaNeck: x + identity_proj            │
│ MLPFFNNeck: x + identity                │
└─────────────────────────────────────────┘
                    │
                    ▼
Phase 2: MASC-M Enhancement
┌─────────────────────────────────────────┐
│ ✓ ResNet BasicBlock (유지)               │
│ ✗ MambaNeck 단순 덧셈 (제거)             │
│ ✓ MLPFFNNeck (유지)                     │
│ ✓ MASC-M Attention Fusion (새로 추가)    │
└─────────────────────────────────────────┘
                    │
                    ▼
Phase 3: Current State
┌─────────────────────────────────────────┐
│ Multi-Scale + Attention-Weighted        │
│ Skip Connection Fusion                  │
│                                         │
│ • 더 풍부한 특징 표현                    │
│ • 동적 가중치 학습                       │
│ • FSCIL 태스크 특화                     │
└─────────────────────────────────────────┘ 