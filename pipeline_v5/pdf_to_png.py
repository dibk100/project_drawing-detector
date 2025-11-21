"""
PDF를 PNG 이미지로 변환하는 도구
모든 페이지를 개별 PNG 파일로 저장합니다.
"""

from pathlib import Path
from typing import List, Optional
import argparse

try:
    import pypdfium2 as pdfium
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("경고: pypdfium2가 설치되지 않았습니다.")
    print("설치: pip install pypdfium2")


class PDFToPNGConverter:
    """PDF를 PNG로 변환하는 클래스"""

    def __init__(self, dpi: int = 200, scale: Optional[float] = None):
        """
        Args:
            dpi: 출력 해상도 (기본값: 200)
            scale: 배율 (dpi 대신 사용 가능, 1.0 = 72dpi)
        """
        if not PDF_SUPPORT:
            raise ImportError("pypdfium2를 설치하세요: pip install pypdfium2")

        self.dpi = dpi
        self.scale = scale if scale is not None else dpi / 72.0

    def convert_pdf_to_images(self, pdf_path: str, page_numbers: Optional[List[int]] = None):
        """
        PDF를 PIL Image 리스트로 변환

        Args:
            pdf_path: PDF 파일 경로
            page_numbers: 변환할 페이지 번호 리스트 (None이면 모든 페이지)

        Returns:
            PIL Image 리스트
        """
        pdf = pdfium.PdfDocument(pdf_path)
        images = []
        total_pages = len(pdf)

        print(f"PDF 총 페이지 수: {total_pages}")

        # 변환할 페이지 결정
        if page_numbers is None:
            pages_to_convert = range(total_pages)
        else:
            pages_to_convert = [p - 1 for p in page_numbers if 0 < p <= total_pages]

        # 각 페이지 변환
        for page_idx in pages_to_convert:
            page = pdf[page_idx]
            bitmap = page.render(scale=self.scale)
            pil_image = bitmap.to_pil()
            images.append(pil_image)

            print(f"페이지 {page_idx + 1}/{total_pages} 변환 완료 - 크기: {pil_image.size}")

        return images

    def save_as_png(self,
                    pdf_path: str,
                    output_dir: str = "output",
                    prefix: str = "",
                    page_numbers: Optional[List[int]] = None,
                    quality: int = 95):
        """
        PDF를 PNG 파일로 저장

        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리
            prefix: 파일명 접두사
            page_numbers: 변환할 페이지 번호 리스트 (None이면 모든 페이지)
            quality: PNG 품질 (1-100)

        Returns:
            저장된 파일 경로 리스트
        """
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # PDF 파일명 (확장자 제외)
        pdf_name = Path(pdf_path).stem

        # 접두사 설정
        if prefix:
            file_prefix = prefix
        else:
            file_prefix = pdf_name

        # 이미지 변환
        images = self.convert_pdf_to_images(pdf_path, page_numbers)

        # PNG로 저장
        saved_files = []
        for idx, image in enumerate(images, 1):
            # 페이지 번호 (실제 페이지 번호)
            if page_numbers:
                page_num = page_numbers[idx - 1]
            else:
                page_num = idx

            # 파일명 생성
            filename = f"{file_prefix}_page_{page_num:03d}.png"
            filepath = output_path / filename

            # PNG 저장
            image.save(filepath, "PNG", quality=quality, optimize=True)
            saved_files.append(str(filepath))

            print(f"저장: {filepath}")

        print(f"\n총 {len(saved_files)}개 페이지 변환 완료")
        print(f"출력 디렉토리: {output_path.absolute()}")

        return saved_files

    def save_single_image(self,
                         pdf_path: str,
                         output_path: str,
                         page_number: int = 1,
                         quality: int = 95):
        """
        PDF의 특정 페이지를 단일 PNG로 저장

        Args:
            pdf_path: PDF 파일 경로
            output_path: 출력 파일 경로
            page_number: 페이지 번호 (1부터 시작)
            quality: PNG 품질
        """
        images = self.convert_pdf_to_images(pdf_path, [page_number])

        if not images:
            raise ValueError(f"페이지 {page_number}를 변환할 수 없습니다.")

        # 출력 디렉토리 생성
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 저장
        images[0].save(output_file, "PNG", quality=quality, optimize=True)
        print(f"저장: {output_file}")

        return str(output_file)


def convert_multiple_pdfs(pdf_files: List[str],
                         output_dir: str = "output",
                         dpi: int = 200,
                         page_numbers: Optional[List[int]] = None):
    """
    여러 PDF 파일을 일괄 변환

    Args:
        pdf_files: PDF 파일 경로 리스트
        output_dir: 출력 디렉토리
        dpi: 해상도
        page_numbers: 변환할 페이지 번호 리스트
    """
    converter = PDFToPNGConverter(dpi=dpi)

    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        print(f"처리 중: {pdf_file}")
        print(f"{'='*60}")

        try:
            converter.save_as_png(
                pdf_file,
                output_dir=output_dir,
                page_numbers=page_numbers
            )
        except Exception as e:
            print(f"오류 발생 ({pdf_file}): {e}")

    print(f"\n모든 PDF 변환 완료!")


def pdf_to_png_main(argv=None):
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='PDF를 PNG 이미지로 변환')
    parser.add_argument('input', help='PDF 파일 또는 디렉토리 경로')
    parser.add_argument('-o', '--output', help='출력 디렉토리', default='output_png')
    parser.add_argument('--dpi', type=int, help='해상도 (기본: 200)', default=200)
    parser.add_argument('--prefix', help='출력 파일명 접두사', default='')
    parser.add_argument('--pages', help='변환할 페이지 (예: 1,3,5-10)', default=None)
    parser.add_argument('--quality', type=int, help='PNG 품질 (1-100, 기본: 95)', default=95)
    parser.add_argument('--single-page', type=int, help='단일 페이지만 변환', default=None)

    args = parser.parse_args(argv)

    # 입력 경로 확인
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {input_path}")
        return

    # 페이지 범위 파싱
    page_numbers = None
    if args.pages:
        page_numbers = parse_page_range(args.pages)
        print(f"변환할 페이지: {page_numbers}")

    # 컨버터 초기화
    converter = PDFToPNGConverter(dpi=args.dpi)

    # 파일 처리
    if input_path.is_file():
        # 단일 파일
        if args.single_page:
            # 단일 페이지
            output_file = f"{args.prefix or input_path.stem}_page_{args.single_page:03d}.png"
            output_path = Path(args.output) / output_file
            converter.save_single_image(
                str(input_path),
                str(output_path),
                args.single_page,
                args.quality
            )
        else:
            # 모든 페이지 또는 지정된 페이지
            converter.save_as_png(
                str(input_path),
                args.output,
                args.prefix,
                page_numbers,
                args.quality
            )

    elif input_path.is_dir():
        # 디렉토리 내 모든 PDF 파일
        pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))

        if not pdf_files:
            print(f"PDF 파일을 찾을 수 없습니다: {input_path}")
            return

        print(f"총 {len(pdf_files)}개의 PDF 파일 발견")
        convert_multiple_pdfs(
            [str(f) for f in pdf_files],
            args.output,
            args.dpi,
            page_numbers
        )

    else:
        print(f"오류: 유효하지 않은 경로: {input_path}")


def parse_page_range(page_str: str) -> List[int]:
    """
    페이지 범위 문자열을 파싱

    예: "1,3,5-10" -> [1, 3, 5, 6, 7, 8, 9, 10]

    Args:
        page_str: 페이지 범위 문자열

    Returns:
        페이지 번호 리스트
    """
    pages = []

    for part in page_str.split(','):
        if '-' in part:
            # 범위 (예: 5-10)
            start, end = part.split('-')
            pages.extend(range(int(start), int(end) + 1))
        else:
            # 단일 페이지
            pages.append(int(part))

    return sorted(list(set(pages)))  # 중복 제거 및 정렬


