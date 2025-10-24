# project : Layout Item Detection from Drawings
- 기록용
- Subject : 도면(설계도, 회로도, 시스템 다이어그램 등)에서 Layout Item을 자동으로 인식하고 추출하는 프로그램 개발
- TASK : Text 인식/ Diagram 인식 / 매핑

## 📁 Folder Structure
```
project/
├── 
├── 
├──    
└── requirements.txt    
```

## 🧪 실험 기록

<details>
<summary>🧾 Text Recognition </summary>

도면 내 텍스트(한글, 영어, 숫자, 기호) 를 정확하게 추출하기 위해 OCR 기반 접근, VLM 기반 접근으로 실험을 진행함.   
- 실험 데이터 : ./data/easy_task_test05_v1.png   
- 추출 텍스트 수 : 32개
   

🧩 OCR Library
   - Tesseract OCR(86개 인식/ 12.5%) : 실선을 텍스트로 오인식, 한글을 문자 단위로 인식
   - EasyOCR(36개 인식/ 22%) : 숫자 인식률 낮음, 심볼의 텍스트 인식 불가
   - **PaddleOCR(40개 인식/ 90%) : 한국어+영어 혼합 인식에서 가장 안정적. 심복의 텍스트는 부분적으로 탐지되지만 명확한 추출 불가**

🧩 VLM(update2025.10.24)
   - Gemini 2.5 Flash(34개 / 100%) : 한국어, 영어 성능이 좋지만, 텍스트 경계 정보(좌표)가 명확하지 않아서 추가 확인이 필요. _이것도 프롬프트 설정 가능하면 좌표도 뽑을 수 있지 않을까?_
   - microsoft/Florence-2-large(33개 / 21%) : 한글 인식률 낮음. 모든 텍스트(문자)를 잡으려고 함.
   - Qwen/Qwen2.5-VL-7B-Instruct(10개 / 28%) : 프롬프트의 실험 필요, json형태로 제대로 나오지 않으면 중간 오류가 잦음. 한국어 인식율 낮음. 다만, text와 블록 인식이 괜찮은편. 10개 추출 중 9개는 정답.
   - [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) (ing): 숫자 인식률 낮음(EasyOCR)

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

3️⃣ SAM 기반 세그멘테이션 실험 : SAM2
- 추후 예정

</details>
