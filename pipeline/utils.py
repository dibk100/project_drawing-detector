import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 한글 폰트 설정
font_path = "/home/dibaeck/.fonts/nanum/NanumGothicCoding.ttf"
font_size = 15
font = ImageFont.truetype(font_path, font_size)

def show_image(img, title="Image"):
    # 도면에 시각화하는 함수
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image_highres(img, title="Image", zoom=2.0, dpi=150):
    # 확대 + 고화질화
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    plt.figure(figsize=(w/100*zoom, h/100*zoom), dpi=dpi)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_detections(drawing, boxes, symbol_name):
    """탐지 결과를 이미지에 표시 (한글 지원)"""
    result = drawing.copy()
    
    # OpenCV 이미지 → PIL 이미지
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(result_pil)

    for (x1, y1, x2, y2) in boxes:
        # 사각형 그리기
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # 한글 텍스트 그리기
        draw.text((x1, y1-20), symbol_name, font=font, fill="red")

    # 다시 OpenCV로 변환 후 시각화
    result_cv = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    show_image(result_cv, f"{symbol_name} 탐지 결과")