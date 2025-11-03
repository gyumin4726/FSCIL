# Selective Scanning with Mixture-of-Experts for Few-Shot Class-Incremental Learning

## 1. Why VMamba?
ResNet 같은 CNN 백본은 지역적 특징 위주라 SS2D(State Space in 2D)와의 결합이 제한적입니다.  
반면 **VMamba**는 이미지를 시퀀스 단위로 처리하면서 **전역 문맥 + 장기 의존성**을 학습할 수 있어  
SS2D와 구조적으로 잘 맞고, FSCIL 환경에서 더 안정적이고 표현력이 뛰어납니다.

---

## 2. Why Mixture-of-Experts?


---

## 3. Why Selective Scanning 2D Experts?


---

## 4. Run
```bash
sh train_cub.sh
