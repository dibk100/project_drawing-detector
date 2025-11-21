"""
도면 OCR 인식 모듈
easyOCR을 사용하여 도면에서 텍스트를 추출합니다.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import easyocr

try:
    import pypdfium2 as pdfium
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class DrawingOCR:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)  # EasyOCR 초기화

    def process_image(self, image_path: str) -> Dict[str, Any]:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        # PDF 처리
        if image_file.suffix.lower() == '.pdf':
            return self.process_pdf(str(image_file))

        # 이미지 로드
        image_cv = cv2.imread(str(image_file))
        if image_cv is None:
            raise ValueError(f"이미지를 열 수 없습니다: {image_path}")

        text_lines = []
        ocr_results = {}

        # EasyOCR 수행
        results = self.ocr_reader.readtext(image_cv)
        full_text = ""
        for bbox, text, confidence in results:
            # 좌표를 int로 변환
            x_min = int(min([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))

            # polygon 좌표도 float -> float 형식 리스트로 변환
            polygon = [[float(p[0]), float(p[1])] for p in bbox]

            text_lines.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": [x_min, y_min, x_max, y_max],  # 좌상단, 우하단
                "polygon": polygon
            })
            full_text += text + "\n"

        result = {
            "image_path": str(image_file),
            "image_size": list(image_cv.shape[1::-1]),
            "timestamp": datetime.now().isoformat(),
            "text_lines": text_lines,
            "full_text": full_text.strip(),
            "layout": None,
            "tables": None
        }
        return result

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 첫 페이지만 OCR 처리"""
        print(f"\nPDF 처리 중: {Path(pdf_path).name}")
        images = self.convert_pdf_to_images(pdf_path)
        if not images:
            raise ValueError(f"PDF에서 이미지를 추출할 수 없습니다: {pdf_path}")

        # 첫 페이지만 처리
        image = np.array(images[0])
        temp_file = Path(pdf_path).with_suffix('.png')
        cv2.imwrite(str(temp_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return self.process_image(str(temp_file))

    def process_pdf_all_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """PDF 모든 페이지 OCR 처리"""
        print(f"\nPDF 전체 페이지 처리 중: {Path(pdf_path).name}")
        images = self.convert_pdf_to_images(pdf_path)
        if not images:
            raise ValueError(f"PDF에서 이미지를 추출할 수 없습니다: {pdf_path}")

        results = []
        for page_num, image in enumerate(images, 1):
            print(f"\n페이지 {page_num}/{len(images)} OCR 처리 중...")
            temp_file = Path(pdf_path).with_name(f"{Path(pdf_path).stem}_page{page_num}.png")
            cv2.imwrite(str(temp_file), cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            result = self.process_image(str(temp_file))
            result['page_number'] = page_num
            result['total_pages'] = len(images)
            results.append(result)
        return results

    def process_directory(self, directory_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """디렉토리 내 모든 이미지 처리"""
        directory = Path(directory_path)
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.pdf']
        image_files = [f for ext in image_extensions for f in directory.glob(f'*{ext}')]
        print(f"\n총 {len(image_files)}개의 파일 발견")

        results = []
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] 처리 중: {image_file.name}")
            try:
                result = self.process_image(str(image_file))
                results.append(result)
            except Exception as e:
                print(f"오류 발생 ({image_file.name}): {e}")
                results.append({
                    "image_path": str(image_file),
                    "error": str(e)
                })

        if output_path:
            self.save_results(results, output_path)

        return results

    def visualize_result(self, image_path: str, result: Dict[str, Any], output_path: str):
        """OCR 영역을 지우고 이미지 저장"""
        image = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        if result.get('text_lines'):
            for line in result['text_lines']:
                bbox = line.get('bbox')
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # 텍스트 영역을 흰색으로 덮음
                    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 255))

        result_image = Image.alpha_composite(image, overlay).convert('RGB')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_image.save(output_path, format='PNG', quality=95)
        print(f"텍스트 제거 결과 저장: {output_path}")
        return result_image
    
    # 추가
    def visualize_result(self, image_path: str, result: Dict[str, Any], output_path: str):
        """OCR 텍스트 영역을 흰색으로 덮고 저장"""
        image = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        if result.get('text_lines'):
            for line in result['text_lines']:
                bbox = line.get('bbox')
                if bbox and len(bbox) == 4:
                    # bbox = [x_min, y_min, x_max, y_max] 형식으로 바로 사용
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 255))

        result_image = Image.alpha_composite(image, overlay).convert('RGB')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_image.save(output_path, format='PNG')
        print(f"텍스트 제거 결과 저장: {output_path}")
        return result_image

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"결과 저장 완료: {output_path}")

    def print_summary(self, results: List[Dict[str, Any]]):
        print("\n" + "="*60)
        print("OCR 처리 결과 요약")
        print("="*60)
        total_images = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        failed = total_images - successful
        print(f"총 이미지 수: {total_images}, 성공: {successful}, 실패: {failed}")
        total_text_lines = sum(len(r.get('text_lines', [])) for r in results if 'error' not in r)
        print(f"총 텍스트 라인 수: {total_text_lines}")
        print("="*60)
        
    def run_easyocr_and_convert(self, image_path: str, langs=['en'], min_confidence=0.7, visualize=True):
        """
        EasyOCR로 텍스트 감지 후 기존 SuryOCR 스타일 구조로 변환

        Returns:
            text_lines: List[Dict] - 기존 auto_zone 코드에서 사용하는 구조
        """
        reader = easyocr.Reader(langs, gpu=True)

        results = reader.readtext(image_path, detail=1)

        def poly_to_bbox(poly):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            return [min(xs), min(ys), max(xs), max(ys)]

        text_lines = []
        for idx, (poly, text, conf) in enumerate(results):
            if conf < min_confidence:
                continue

            bbox = poly_to_bbox(poly)
            text_lines.append({
                'id': f"text_{idx}",
                'bbox': bbox,
                'polygon': poly,
                'text': text,
                'confidence': conf
            })

        print(f"EasyOCR 감지 완료: {len(text_lines)}개 텍스트 (신뢰도 {min_confidence} 이상)")
        
        if visualize:
            # 이미지 읽기
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.imshow(img_rgb)

            # 사각형 그리기
            for line in text_lines:
                x_min, y_min, x_max, y_max = line['bbox']
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                # ax.text(x_min, y_min - 5, line['text'], color='yellow', fontsize=10, backgroundcolor="black")

            plt.axis('off')
            plt.show()

        return text_lines

def ocr_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='도면 OCR 인식 도구 (PaddleOCR 기반)')
    parser.add_argument('input', help='입력 이미지 파일 또는 디렉토리 경로')
    parser.add_argument('-o', '--output', help='결과 저장 경로 (JSON)', default='ocr_results.json')
    parser.add_argument('--visualize', '-v', action='store_true', help='결과 시각화')
    parser.add_argument('--viz-output', help='시각화 이미지 저장 디렉토리', default='visualized')
    parser.add_argument('--no-gpu', action='store_true', help='GPU 사용 비활성화')
    parser.add_argument('--all-pages', action='store_true', help='PDF 모든 페이지 처리')

    args = parser.parse_args(argv)

    ocr = DrawingOCR(use_gpu=not args.no_gpu)
    input_path = Path(args.input)

    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf' and args.all_pages:
            results = ocr.process_pdf_all_pages(str(input_path))
        else:
            results = [ocr.process_image(str(input_path))]
        ocr.save_results(results, args.output)
    elif input_path.is_dir():
        results = ocr.process_directory(str(input_path), output_path=args.output)
    else:
        print(f"오류: 입력 경로를 찾을 수 없습니다: {input_path}")
        return

    ocr.print_summary(results)

    if args.visualize:
        for result in results:
            if 'error' not in result:
                image_path = result['image_path']
                #DrawingOCR.visualize_result_nb(image_path, result, font_path="/home/dibaeck/.fonts/nanum/NanumGothicCoding.ttf")
                save_path = Path(args.viz_output) / Path(image_path).name
                ocr.visualize_result(image_path, result, str(save_path))
                
    if results and 'error' not in results[0]:
        print("\n첫 번째 이미지 텍스트 샘플:")
        print("-"*60)
        print(results[0].get('full_text', '')[:500])
        if len(results[0].get('full_text', '')) > 500:
            print("...")
