# project : Layout Item Detection from Drawings
- **Type :** 공개용 (원본 레포: `private_drawing-detector`)
- **Purpose :** 도면(설계도, 회로도, 시스템 다이어그램 등)에서 Layout Item을 자동으로 인식하고 추출하는 자동화 프로그램 개발
- **Task :** Text 인식/ Diagram 인식 / 매핑

> 🗂️ This repository is a **public version** of the internal project `private_drawing-detector`.  
> 일부 실험 코드 및 데이터셋은 비공개 버전에만 포함됨.

## 📁 Folder Structure
```
project_drawing-detector/
├── diagram_segmentation/      # 도면 내 블록/라인/심볼 등 구조 인식 실험 공간
│   ├── detection_lightSAM.ipynb    
│   ├── detection_openCV_v1.ipynb   
│   └── symbol_detection_v1.ipynb                 
│
├── text_recognition/          # 텍스트 인식 및 OCR 관련 실험 공간
│   ├── ocr_comparison.ipynb     
│   ├── vlm_test_florence.ipynb   
│   └── ...          
│
├── pipeline/                  # OCR 파이프라인 통합 (ing)
│   ├── pipeline_test_v2.ipynb      
│   ├── symbol_region_splitter.py        
│   └── utils.py               
│
└── requirements.txt           # 환경 설정 파일 (Paddle 비활성)

```

## 📌 Notes & Issues 🧷
**PaddleOCR**
- 환경 : CUDA 12.6 (PyTorch), PaddlePaddle GPU 2.6.2, PaddleOCR 2.7.0, python 3.11
- 이슈 : torch / langchain / PyMuPDF 버전 충돌로 설치 불가
- 조치 : python 3.10로 다운그레이드

**DeepSeekOCR** 
- 환경 : CUDA 12.6 (`cu126`), torch==2.6.0, python 3.11
- 이슈 : `flash-attn` 설치 실패 → flash_attention_2 미사용
   - 모델 로드 시 FlashAttention 옵션 제거(`_attn_implementation='flash_attention_2'`)
- 조치 : GPU inference는 HuggingFace Transformers로 대체
- 메모 : flash_attention_2를 사용하기 위해 `nvidia-cuda-toolkit 13.0`(최신) 설치했으나 flash 미사용

**Memo**
- python version은 3.10으로 하는 게 안전한 것 같음.


## 🧪 실험 기록
> 실험 순서 : Text Recognition → Diagram Segmentation → Pipeline (OCR)

<details>
<summary>🧾 Pipeline (OCR)</summary>

이 파이프라인은 도면 전체 OCR 성능이 낮은 문제를 해결하기 위해, **도면을 '심볼 단위'로 분할(crop) → 확대 → OCR 적용 → 통합**하는 방식으로 실험을 설계함.  

📌 *이전 실험(🧾Text Recognition)*   
 도면 전체를 한 번에 인식할 경우 텍스트가 작고, 한글/영문/숫자 혼용으로 인해 인식률이 급격히 저하됨을 확인함.

### 🧱 구성 흐름

| 단계 | 설명 | 비고 |
|------|------|------|
| **1. Symbol Detection** | 도면의 블록/심볼 단위로 ROI(region) 추출 | 좌표 단위 crop |
| **2. Cropping & Scaling** | 각 심볼 영역을 확대(resize)하여 텍스트 크기 보정 | 2배 확대 |
| **3. OCR 적용** | OCR 엔진별 텍스트 추출 (한/영/숫자 혼합 인식) | PaddleOCR, DeepSeek-OCR 실험 |
| **4. 결과 통합** | 각 심볼 OCR 결과를 병합하여 전체 도면 텍스트 구성 | 중복·빈 영역 제거 |



</details>

<details>
<summary>🧾 Text Recognition </summary>

도면 내 텍스트(한글, 영어, 숫자, 기호)를 정확하게 추출하기 위해  
OCR 기반 접근 및 VLM 기반 접근 실험을 수행함.

- **테스트 데이터:** `./data/easy_task_test05_v1.png`  
- **추출 텍스트 수:** 32개  
   

### 🔍 OCR Libraries
| Library | 인식 수 / 정확도 | 주요 이슈 |
|----------|------------------|------------|
| **Tesseract OCR** | 86개 / 12.5% | 실선을 텍스트로 오인식, 한글을 문자 단위로 인식 |
| **EasyOCR** | 36개 / 22% | 숫자 인식률 낮음, 심볼 내부 텍스트 인식 불가 |
| **PaddleOCR** | 40개 / **90%** | 한영 혼합 인식 안정적, 다만 심볼 내부 텍스트는 일부 누락 |

---

### 🔍 VLM Models *(update 2025.10.24)*
| Model | 인식 수 / 정확도 | 특징 / 메모 |
|--------|------------------|--------------|
| **Gemini 2.5 Flash** | 34 / 100% | 텍스트 인식 우수하나 좌표 정보 불명확 — 프롬프트로 개선 가능성 |
| **Florence-2-large** | 33 / 21% | 한글 인식률 낮음, 텍스트 단위가 과도하게 세분화됨 |
| **Qwen2.5-VL-7B-Instruct** | 10 / 28% | JSON 출력 불안정, 한글 인식 낮음 / 텍스트–블록 매핑 우수 |
| **DeepSeek-OCR** | (ing) | 프롬프트 민감도 편차가 큼. 최적의 프롬프트를 찾아야함 |

</details>

<details>
<summary>🧾 추가 실험 Diagram Segmentation </summary>
  
도면 내 Diagram의 블록(Block), 선(Line) 인식 테스트

1️⃣ Hough Transform + OpenCV조합 
- 직선 감지 다수 누락 / 노이즈 라인 검출 다수 발생
- 파라미터 조정이나 함수 수정 필요한 것 같음

2️⃣ SAM 기반 세그멘테이션 실험 :   
- model : Zigeng/SlimSAM-uniform-77(경량화 버전)
   - Bounding box가 영역 외곽에 비정상적으로 생성
   - 해당 모델은 mask만 제공해서 mask 후처리를 메뉴얼하게 함

3️⃣ Segmentation / Detection 모델 검토 중
- **SAM2 기반 세그멘테이션**과 **YOLO-OCR 기반 객체 탐지형 접근**을 비교 검토(예정)

</details>
