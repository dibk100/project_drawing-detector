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
### 🧾 Text Recognition
도면 내 텍스트(한글, 영어, 숫자, 기호) 를 정확하게 추출하기 위해 OCR 기반 접근, VLM 기반 접근으로 실험을 진행함.

🧩 OCR Library
   - Tesseract OCR, EasyOCR : 한글 인식률 낮음
   - **PaddleOCR** : 한국어+영어 혼합 인식에서 가장 안정적. 다만, 텍스트의 간격이 좁으면 인식률이 떨어짐   

🧩 VLM
   - Gemini 2.5 Flash(api) : 한국어, 영어 성능이 좋지만, 텍스트 경계 정보가 명확하지 않아서 OCR 대비 인식 안정성 낮음
   - microsoft/Florence-2-large(opensource) : 한국어 인식율이 낮음. 
   

> 결론 : ✅ PaddleOCR을 도면용 텍스트 인식 baseline 모델로 채택

### 🧾 추가 실험 Diagram Segmentation :  
도면 내 Diagram의 블록(Block), 선(Line) 인식 테스트

1️⃣ Hough Transform + OpenCV조합 
- 직선 감지 다수 누락 / 노이즈 라인 검출 다수 발생

2️⃣ SAM 기반 세그멘테이션 실험 :   
- model : Zigeng/SlimSAM-uniform-77(경량화 버전)
   - Bounding box가 영역 외곽에 비정상적으로 생성
   - 코드 수정이 필요하거나 추가 실험 필요

3️⃣ SAM 기반 세그멘테이션 실험 : SAM2
