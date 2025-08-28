# MoE-FSCIL: Mixture of Experts for Few-Shot Class-Incremental Learning

## 1. Why VMamba?
ResNet 같은 CNN 백본은 지역적 특징 위주라 SS2D(State Space in 2D)와의 결합이 제한적입니다.  
반면 **VMamba**는 이미지를 시퀀스 단위로 처리하면서 **전역 문맥 + 장기 의존성**을 학습할 수 있어  
SS2D와 구조적으로 잘 맞고, FSCIL 환경에서 더 안정적이고 표현력이 뛰어납니다.

---

## 2. Why Intermediate Representations?
FSCIL에서는 **representation drift** 문제가 핵심입니다.  
- 하위/중간 레이어 표현 → 안정적, 기존 클래스 보존에 도움  
- 최종 표현 → 적응성이 커서 신규 클래스 학습에 유리  

따라서 중간 표현을 같이 활용하면 **Stability–Plasticity 균형**을 잡을 수 있습니다.  
이 프로젝트는 multi-scale skip 구조로 중간 표현을 효과적으로 연결합니다.

---

## 3. Why MoE?
기존 branch 구조(Identity/Base/New)는 고정적이고 확장성에 한계가 있습니다.  
이를 대신해 **Mixture of Experts (MoE)** 구조를 도입했습니다.  

- 입력마다 다른 expert를 동적으로 선택  
- expert 수가 늘어나도 FLOPs는 일정 (sparse activation)  
- load balancing loss로 안정적 학습  

=> 결과적으로 더 단순하고, 확장 가능하며, FSCIL에 특화된 neck 구조가 됩니다.

---

## 4. Run
```bash
sh train_cub.sh
