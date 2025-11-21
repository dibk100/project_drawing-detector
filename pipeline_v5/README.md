# project : Layout Item Detection from Drawings
- 기록용(ver5)
- Subject : 도면(설계도, 회로도, 시스템 다이어그램 등)에서 Layout Item을 자동으로 인식하고 추출하는 프로그램 개발
- TASK : Text 인식/ Diagram 인식 / 매핑

## 📁 Folder Structure
```
project/
├── data/            # pdf 파일 1건
├── outputs_png/     # pdf_to_png.py 결과
├── output_v1/
├── output_v2/       # 표삭제된 이미지, 텍스트 삭제된 이미지
├── output_v3_text/  # easyOCR 텍스트 결과(json)
├── step_by_step_preprocessing.ipynb         # v1-v3 실행 파일(전처리만 다룸)
|   
├── test_v1/            # 탐지된 박스 좌표(json)
└── step_by_step_test.ipynb    # **탐지 작업용**
|   
└── requirements_v5.txt    # python=3.10로 설정, easyOCR
```
