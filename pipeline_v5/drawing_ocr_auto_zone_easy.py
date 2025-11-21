"""
자동 영역 감지 도면 OCR
이미지에서 사각형과 선을 자동으로 감지하여 텍스트를 구분합니다.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import json
import cv2
import numpy as np

from drawing_ocr_easy import DrawingOCR


class AutoZoneDetector:
    """자동 영역 감지 클래스"""

    def __init__(self,
                 min_box_area: int = 1000,       # 너무 작은 박스 제거 500 -> 1000
                 min_line_length: int = 100,
                 canny_threshold1: int = 50,
                 canny_threshold2: int = 150,
                 debug: bool = False):
        """
        Args:
            min_box_area: 최소 사각형 면적
            min_line_length: 최소 선 길이
            canny_thresholㅇd1: Canny edge 낮은 임계값
            canny_threshold2: Canny edge 높은 임계값
            debug: 디버깅 모드 (중간 이미지 저장)
        """
        self.min_box_area = min_box_area
        self.min_line_length = min_line_length
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.debug = debug

    def detect_rectangles(self, image_path: str, text_regions: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        이미지에서 사각형 감지
        * 이미지에서 사각형 감지 (가로 길이가 특이하게 긴 박스 제외)
        """
        # 이미지 로드
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 텍스트 영역 제거
        if text_regions:
            for text in text_regions:
                bbox = text.get('bbox')
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    margin = 3
                    cv2.rectangle(gray,
                                (max(0, int(x1)-margin), max(0, int(y1)-margin)),
                                (min(gray.shape[1], int(x2)+margin), min(gray.shape[0], int(y2)+margin)),
                                255, -1)
            if self.debug:
                debug_dir = Path("debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / "0_text_removed.png"), gray)
                print(f"디버그: {len(text_regions)}개 텍스트 영역 제거됨")

        # 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge 감지
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

        # 수평/수직 선 강조
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        h_dilated = cv2.dilate(edges, h_kernel, iterations=1)
        v_dilated = cv2.dilate(edges, v_kernel, iterations=1)
        combined = cv2.add(h_dilated, v_dilated)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 후보 네모의 가로 길이 계산 (평균 + 2*표준편차 기준)
        widths = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_box_area:
                epsilon = 0.015 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                if len(approx) == 4:
                    _, _, w, _ = cv2.boundingRect(approx)
                    widths.append(w)
        max_width_allowed = (np.mean(widths) + 2 * np.std(widths)) if widths else float('inf')

        rectangles = []
        rejected_boxes = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if area < self.min_box_area:
                if self.debug and len(approx) >= 3:
                    x, y, w, h = cv2.boundingRect(approx)
                    rejected_boxes.append({
                        'reason': 'area_too_small',
                        'area': int(area),
                        'vertices': len(approx),
                        'bbox': [int(x), int(y), int(x + w), int(y + h)]
                    })
                continue

            if len(approx) != 4:
                if self.debug and area >= self.min_box_area:
                    x, y, w, h = cv2.boundingRect(approx)
                    reason = f'polygon_{len(approx)}_vertices'
                    rejected_boxes.append({
                        'reason': reason,
                        'area': int(area),
                        'vertices': len(approx),
                        'bbox': [int(x), int(y), int(x + w), int(y + h)]
                    })
                continue

            x, y, w, h = cv2.boundingRect(approx)

            # 가로 길이가 평균 + 2*표준편차 이상이면 제외
            if w > max_width_allowed:
                if self.debug:
                    rejected_boxes.append({
                        'reason': 'too_wide',
                        'area': int(area),
                        'vertices': len(approx),
                        'bbox': [int(x), int(y), int(x + w), int(y + h)]
                    })
                continue

            rectangles.append({
                'id': f'box_{idx}',
                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                'area': int(area),
                'width': int(w),
                'height': int(h),
                'vertices': 4,
                'type': 'rectangle'
            })

        # 디버깅: 거부된 박스 시각화
        if self.debug and rejected_boxes:
            debug_dir = Path("output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_img = img.copy()
            for box in rejected_boxes[:20]:
                bbox = box['bbox']
                color = (0, 0, 255) if box['reason'] in ['area_too_small', 'too_wide'] else (255, 0, 0)
                cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = f"{box['reason'][:10]} A:{box['area']} V:{box['vertices']}"
                cv2.putText(debug_img, label, (bbox[0], bbox[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imwrite(str(debug_dir / "10_rejected_boxes.png"), debug_img)
            print(f"디버그: {len(rejected_boxes)}개 도형 거부됨 (빨강=면적부족/너무넓음, 파랑=사각형아님)")

        rectangles.sort(key=lambda x: x['area'], reverse=True)
        print(f"감지된 사각형: {len(rectangles)}개")
        return rectangles
    # def detect_rectangles(self, image_path: str, text_regions: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
    #     """
    #     이미지에서 사각형 감지
    #     * 이미지에서 사각형 감지 (가로 길이가 특이하게 긴 박스 제외)

    #     Args:
    #         image_path: 이미지 경로
    #         text_regions: OCR로 감지된 텍스트 영역 리스트 (bbox 포함)

    #     Returns:
    #         감지된 사각형 리스트
    #     """
    #     # 이미지 로드
    #     img = cv2.imread(image_path)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #     # 텍스트 영역을 원본에서 제거 (흰색으로 채움)
    #     if text_regions:
    #         for text in text_regions:
    #             bbox = text.get('bbox')
    #             if bbox and len(bbox) == 4:
    #                 x1, y1, x2, y2 = bbox
    #                 # 텍스트 영역을 흰색으로 완전히 채움 (약간 확장)
    #                 margin = 3
    #                 cv2.rectangle(gray,
    #                             (max(0, int(x1)-margin), max(0, int(y1)-margin)),
    #                             (min(gray.shape[1], int(x2)+margin), min(gray.shape[0], int(y2)+margin)),
    #                             255, -1)  # 흰색으로 채움

    #         if self.debug:
    #             debug_dir = Path("output/debug")
    #             debug_dir.mkdir(parents=True, exist_ok=True)
    #             cv2.imwrite(str(debug_dir / "0_text_removed.png"), gray)
    #             print(f"디버그: {len(text_regions)}개 텍스트 영역 제거됨")

    #     # 노이즈 제거
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #     # Edge 감지
    #     edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

    #     # 형태학적 연산으로 점선을 실선으로 연결
    #     # 수평 방향 커널 (점선을 수평으로 연결)
    #     h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    #     h_dilated = cv2.dilate(edges, h_kernel, iterations=1)

    #     # 수직 방향 커널 (점선을 수직으로 연결)
    #     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    #     v_dilated = cv2.dilate(edges, v_kernel, iterations=1)

    #     # 수평과 수직 결합
    #     combined = cv2.add(h_dilated, v_dilated)

    #     # 추가로 작은 간격 메우기 (closing 연산)
    #     kernel = np.ones((3, 3), np.uint8)
    #     dilated = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    #     # 덩어리 제거: 수평선과 수직선만 추출
    #     # 1. 수평선 추출
    #     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    #     horizontal_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horizontal_kernel)

    #     # 2. 수직선 추출
    #     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    #     vertical_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)

    #     # 3. 수평선 + 수직선 결합 (덩어리는 제거됨)
    #     lines_only = cv2.add(horizontal_lines, vertical_lines)

    #     # 4. 선을 약간 굵게 (박스 감지를 위해)
    #     thicken_kernel = np.ones((3, 3), np.uint8)
    #     lines_thickened = cv2.dilate(lines_only, thicken_kernel, iterations=1)

    #     # 디버깅: 중간 이미지 저장
    #     if self.debug:
    #         debug_dir = Path("output/debug")
    #         debug_dir.mkdir(parents=True, exist_ok=True)
    #         cv2.imwrite(str(debug_dir / "1_edges.png"), edges)
    #         cv2.imwrite(str(debug_dir / "2_h_dilated.png"), h_dilated)
    #         cv2.imwrite(str(debug_dir / "3_v_dilated.png"), v_dilated)
    #         cv2.imwrite(str(debug_dir / "4_combined.png"), combined)
    #         cv2.imwrite(str(debug_dir / "5_dilated_final.png"), dilated)
    #         cv2.imwrite(str(debug_dir / "6_horizontal_lines.png"), horizontal_lines)
    #         cv2.imwrite(str(debug_dir / "7_vertical_lines.png"), vertical_lines)
    #         cv2.imwrite(str(debug_dir / "8_lines_only.png"), lines_only)
    #         cv2.imwrite(str(debug_dir / "9_lines_thickened.png"), lines_thickened)

    #     # 윤곽선 찾기 (선만 남긴 이미지에서)
    #     contours, _ = cv2.findContours(lines_thickened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     rectangles = []
    #     rejected_boxes = []  # 디버깅용

    #     for idx, contour in enumerate(contours):
    #         # 면적 필터링
    #         area = cv2.contourArea(contour)

    #         # 도형 근사 (닫힌 도형 확인)
    #         epsilon = 0.015 * cv2.arcLength(contour, True)
    #         approx = cv2.approxPolyDP(contour, epsilon, True)

    #         # 면적 필터링
    #         if area < self.min_box_area:
    #             if self.debug and len(approx) >= 3:  # 3개 이상 꼭지점
    #                 x, y, w, h = cv2.boundingRect(approx)
    #                 rejected_boxes.append({
    #                     'reason': 'area_too_small',
    #                     'area': int(area),
    #                     'vertices': len(approx),
    #                     'bbox': [int(x), int(y), int(x + w), int(y + h)]
    #                 })
    #             continue

    #         # 사각형만 찾기 (4개 꼭지점)
    #         vertices = len(approx)
    #         if vertices == 4:  # 사각형만
    #             x, y, w, h = cv2.boundingRect(approx)

    #             rectangles.append({
    #                 'id': f'box_{idx}',
    #                 'bbox': [int(x), int(y), int(x + w), int(y + h)],
    #                 'area': int(area),
    #                 'width': int(w),
    #                 'height': int(h),
    #                 'vertices': 4,
    #                 'type': 'rectangle'
    #             })
    #         else:
    #             # 디버깅: 사각형이 아닌 도형
    #             if self.debug and area >= self.min_box_area:
    #                 x, y, w, h = cv2.boundingRect(approx)
    #                 if vertices < 3:
    #                     reason = 'not_closed_shape'
    #                 elif vertices == 3:
    #                     reason = 'triangle'
    #                 elif vertices > 4:
    #                     reason = f'polygon_{vertices}_vertices'
    #                 else:
    #                     reason = 'unknown'

    #                 rejected_boxes.append({
    #                     'reason': reason,
    #                     'area': int(area),
    #                     'vertices': vertices,
    #                     'bbox': [int(x), int(y), int(x + w), int(y + h)]
    #                 })

    #     # 디버깅: 거부된 박스 시각화
    #     if self.debug and rejected_boxes:
    #         debug_img = img.copy()
    #         for box in rejected_boxes[:20]:  # 상위 20개만
    #             bbox = box['bbox']
    #             color = (0, 0, 255) if box['reason'] == 'area_too_small' else (255, 0, 0)
    #             cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    #             label = f"{box['reason'][:10]} A:{box['area']} V:{box['vertices']}"
    #             cv2.putText(debug_img, label, (bbox[0], bbox[1]-5),
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    #         cv2.imwrite(str(debug_dir / "10_rejected_boxes.png"), debug_img)
    #         print(f"디버그: {len(rejected_boxes)}개 도형 거부됨 (빨강=면적부족, 파랑=사각형아님)")

    #         # 거부 이유별 통계
    #         reason_stats = {}
    #         for b in rejected_boxes:
    #             reason = b['reason']
    #             reason_stats[reason] = reason_stats.get(reason, 0) + 1

    #         print(f"  거부 이유별 통계:")
    #         for reason, count in sorted(reason_stats.items()):
    #             print(f"    - {reason}: {count}개")

    #     # 면적 순으로 정렬
    #     rectangles.sort(key=lambda x: x['area'], reverse=True)

    #     print(f"감지된 사각형: {len(rectangles)}개")

    #     return rectangles

    def detect_paths(self, image_path: str, rectangles: List[Dict[str, Any]] = None,
                     text_regions: List[Dict[str, Any]] = None, distance_threshold: int = 10) -> List[Dict[str, Any]]:
        """
        박스에서 텍스트로 향하는 경로 추출 (직선이 아닌 실제 경로)

        Args:
            image_path: 이미지 경로
            rectangles: 감지된 사각형 리스트 (박스 기준점)
            text_regions: OCR로 감지된 텍스트 영역 리스트
            distance_threshold: 박스/텍스트와의 거리 임계값 (픽셀)

        Returns:
            박스-텍스트 간 경로 리스트
        """
        if not rectangles or not text_regions:
            print("경로 추출: 박스 또는 텍스트 영역이 없음")
            return []

        # 이미지 로드
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge 감지
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)

        # 박스와 텍스트 영역을 제외한 선만 추출하기 위해 마스크 생성
        mask = np.ones_like(gray) * 255

        # 박스와 텍스트는 흰색으로 채워서 제외
        for rect in rectangles:
            bbox = rect['bbox']
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 0, -1)

        for text in text_regions:
            bbox = text.get('bbox')
            if bbox and len(bbox) == 4:
                cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), 0, -1)

        # 마스크 적용: 박스/텍스트 내부는 제거
        edges_masked = cv2.bitwise_and(edges, mask)

        if self.debug:
            debug_dir = Path("output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "path_edges_masked.png"), edges_masked)

        detected_paths = []

        # 각 박스의 변에서 시작하는 경로 찾기
        for rect_idx, rect in enumerate(rectangles):
            bbox = rect['bbox']
            bx1, by1, bx2, by2 = bbox

            # 박스의 4개 변에서 시작점 찾기
            box_edges = self._get_box_edge_points(bbox, edges_masked, distance_threshold)

            # 각 시작점에서 텍스트 영역으로의 경로 추적
            for edge_point in box_edges:
                for text_idx, text in enumerate(text_regions):
                    text_bbox = text.get('bbox')
                    if not text_bbox or len(text_bbox) != 4:
                        continue

                    # 경로 추적
                    path = self._trace_path(edge_point, text_bbox, edges_masked)

                    if path and len(path) >= 2:  # 유효한 경로
                        detected_paths.append({
                            'id': f'path_{rect_idx}_{text_idx}',
                            'source_box': rect['id'],
                            'target_text': text_idx,
                            'path': path,
                            'length': self._calculate_path_length(path),
                            'type': 'box_to_text'
                        })

        print(f"감지된 경로: {len(detected_paths)}개 (박스-텍스트 연결)")
        return detected_paths

    def _get_box_edge_points(self, bbox: List[int], edges: np.ndarray, threshold: int) -> List[Tuple[int, int]]:
        """
        박스 변에서 선이 시작되는 점들 찾기 (최적화: 샘플링)

        Args:
            bbox: 박스 좌표 [x1, y1, x2, y2]
            edges: 엣지 이미지
            threshold: 박스 변으로부터의 거리 임계값

        Returns:
            시작점 리스트 [(x, y), ...]
        """
        bx1, by1, bx2, by2 = bbox
        edge_points = []
        sample_interval = 10  # 10픽셀마다 샘플링 (속도 향상)

        # 각 변에서 엣지 픽셀 찾기 (샘플링 적용)
        # 상단 변
        for x in range(max(0, bx1), min(edges.shape[1], bx2), sample_interval):
            for dy in range(-threshold, threshold+1):
                y = by1 + dy
                if 0 <= y < edges.shape[0] and edges[y, x] > 0:
                    edge_points.append((x, y))
                    break

        # 하단 변
        for x in range(max(0, bx1), min(edges.shape[1], bx2), sample_interval):
            for dy in range(-threshold, threshold+1):
                y = by2 + dy
                if 0 <= y < edges.shape[0] and edges[y, x] > 0:
                    edge_points.append((x, y))
                    break

        # 좌측 변
        for y in range(max(0, by1), min(edges.shape[0], by2), sample_interval):
            for dx in range(-threshold, threshold+1):
                x = bx1 + dx
                if 0 <= x < edges.shape[1] and edges[y, x] > 0:
                    edge_points.append((x, y))
                    break

        # 우측 변
        for y in range(max(0, by1), min(edges.shape[0], by2), sample_interval):
            for dx in range(-threshold, threshold+1):
                x = bx2 + dx
                if 0 <= x < edges.shape[1] and edges[y, x] > 0:
                    edge_points.append((x, y))
                    break

        return edge_points

    def _trace_path(self, start: Tuple[int, int], target_bbox: List[float],
                    edges: np.ndarray, max_length: int = 500) -> List[Tuple[int, int]]:
        """
        시작점에서 목표 영역까지의 경로 추적 (최적화)

        Args:
            start: 시작점 (x, y)
            target_bbox: 목표 텍스트 박스 [x1, y1, x2, y2]
            edges: 엣지 이미지
            max_length: 최대 경로 길이 (기본값: 500)

        Returns:
            경로 점들의 리스트 [(x, y), ...]
        """
        tx1, ty1, tx2, ty2 = [int(v) for v in target_bbox]
        path = [start]
        visited = set([start])
        current = start

        # 목표 중심점
        target_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)

        # 시작점과 목표까지의 거리가 너무 멀면 조기 종료
        init_dist = np.sqrt((start[0] - target_center[0])**2 + (start[1] - target_center[1])**2)
        if init_dist > max_length * 2:  # 너무 멀면 포기
            return []

        prev_dist = init_dist

        for step in range(max_length):
            # 목표 영역에 도달했는지 확인
            if tx1 <= current[0] <= tx2 and ty1 <= current[1] <= ty2:
                return path

            # 다음 이동 가능한 점 찾기 (8방향)
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = current[0] + dx, current[1] + dy

                    # 경계 체크
                    if not (0 <= nx < edges.shape[1] and 0 <= ny < edges.shape[0]):
                        continue

                    # 방문 체크
                    if (nx, ny) in visited:
                        continue

                    # 엣지가 있는지 체크
                    if edges[ny, nx] > 0:
                        # 목표까지의 거리 계산
                        dist = np.sqrt((nx - target_center[0])**2 + (ny - target_center[1])**2)
                        neighbors.append(((nx, ny), dist))

            if not neighbors:
                # 경로가 막혔으면 중단
                break

            # 목표에 가장 가까운 점 선택
            neighbors.sort(key=lambda x: x[1])
            next_point = neighbors[0][0]
            next_dist = neighbors[0][1]

            # 진행이 없으면 조기 종료 (20스텝 동안 거리가 줄어들지 않음)
            if step > 20 and next_dist >= prev_dist:
                return []

            path.append(next_point)
            visited.add(next_point)
            current = next_point
            prev_dist = next_dist

        # 목표에 도달하지 못한 경우 빈 경로 반환
        if not (tx1 <= current[0] <= tx2 and ty1 <= current[1] <= ty2):
            return []

        return path

    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """
        경로의 총 길이 계산

        Args:
            path: 경로 점들의 리스트

        Returns:
            경로 길이
        """
        if len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx*dx + dy*dy)

        return length

    def detect_lines(self, image_path: str, rectangles: List[Dict[str, Any]] = None, distance_threshold: int = 10) -> List[Dict[str, Any]]:
        """
        박스에서 시작하는 직선 감지 (수평선, 수직선)

        Args:
            image_path: 이미지 경로
            rectangles: 감지된 사각형 리스트 (박스 기준점)
            distance_threshold: 박스와의 거리 임계값 (픽셀)

        Returns:
            박스에서 시작하는 선 리스트
        """
        # 이미지 로드
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge 감지
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)

        # Hough Line Transform으로 선 감지
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=self.min_line_length,
            maxLineGap=10
        )

        detected_lines = []
        rejected_lines = []  # 디버깅용

        if lines is not None:
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]

                # 선의 방향 계산
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # 수평선 또는 수직선 판단
                if dx > dy * 3:  # 수평선
                    line_type = 'horizontal'
                    length = dx
                elif dy > dx * 3:  # 수직선
                    line_type = 'vertical'
                    length = dy
                else:
                    continue  # 대각선은 제외

                # 박스 기준점 체크 (rectangles가 제공된 경우만)
                if rectangles:
                    is_near_box = self._is_line_near_rectangles(
                        [int(x1), int(y1)], [int(x2), int(y2)],
                        rectangles, distance_threshold
                    )

                    if not is_near_box:
                        if self.debug:
                            rejected_lines.append({
                                'start': [int(x1), int(y1)],
                                'end': [int(x2), int(y2)],
                                'type': line_type,
                                'reason': 'not_near_box'
                            })
                        continue

                detected_lines.append({
                    'id': f'line_{idx}',
                    'start': [int(x1), int(y1)],
                    'end': [int(x2), int(y2)],
                    'length': int(length),
                    'type': line_type
                })

        # 디버깅: 거부된 선 통계
        if self.debug and rectangles:
            print(f"디버그: 박스 근처가 아닌 선 {len(rejected_lines)}개 제외됨")

        # 길이 순으로 정렬
        detected_lines.sort(key=lambda x: x['length'], reverse=True)

        print(f"감지된 선: {len(detected_lines)}개 (박스에서 시작하는 수평/수직선)")
        return detected_lines

    def _is_line_near_rectangles(self, start: List[int], end: List[int],
                                  rectangles: List[Dict[str, Any]], threshold: int) -> bool:
        """
        선이 사각형 근처에 있는지 확인 (선의 시작점 또는 끝점이 박스 변에 인접)

        Args:
            start: 선의 시작점 [x, y]
            end: 선의 끝점 [x, y]
            rectangles: 사각형 리스트
            threshold: 거리 임계값

        Returns:
            박스 근처에 있으면 True
        """
        for rect in rectangles:
            bbox = rect['bbox']
            bx1, by1, bx2, by2 = bbox

            # 선의 시작점 또는 끝점이 박스 변 근처에 있는지 체크
            for point in [start, end]:
                px, py = point

                # 박스의 4개 변과의 거리 계산
                # 상단 변
                if abs(py - by1) <= threshold and bx1 - threshold <= px <= bx2 + threshold:
                    return True
                # 하단 변
                if abs(py - by2) <= threshold and bx1 - threshold <= px <= bx2 + threshold:
                    return True
                # 좌측 변
                if abs(px - bx1) <= threshold and by1 - threshold <= py <= by2 + threshold:
                    return True
                # 우측 변
                if abs(px - bx2) <= threshold and by1 - threshold <= py <= by2 + threshold:
                    return True

        return False

    def merge_nearby_lines(self, lines: List[Dict[str, Any]], threshold: int = 10) -> List[Dict[str, Any]]:
        """
        가까운 선들을 병합

        Args:
            lines: 선 리스트
            threshold: 병합 임계값 (픽셀)

        Returns:
            병합된 선 리스트
        """
        if not lines:
            return []

        # 수평선과 수직선 분리
        h_lines = [l for l in lines if l['type'] == 'horizontal']
        v_lines = [l for l in lines if l['type'] == 'vertical']

        # 수평선 병합
        merged_h = self._merge_horizontal_lines(h_lines, threshold)

        # 수직선 병합
        merged_v = self._merge_vertical_lines(v_lines, threshold)

        return merged_h + merged_v

    def _merge_horizontal_lines(self, lines: List[Dict[str, Any]], threshold: int) -> List[Dict[str, Any]]:
        """수평선 병합"""
        if not lines:
            return []

        # y 좌표로 그룹화
        groups = []
        for line in lines:
            y_avg = (line['start'][1] + line['end'][1]) / 2
            x1 = min(line['start'][0], line['end'][0])
            x2 = max(line['start'][0], line['end'][0])

            # 기존 그룹에 추가 가능한지 확인
            added = False
            for group in groups:
                if abs(group['y'] - y_avg) < threshold:
                    # 같은 그룹
                    group['x1'] = min(group['x1'], x1)
                    group['x2'] = max(group['x2'], x2)
                    group['count'] += 1
                    added = True
                    break

            if not added:
                groups.append({
                    'y': y_avg,
                    'x1': x1,
                    'x2': x2,
                    'count': 1
                })

        # 결과 생성
        merged = []
        for idx, group in enumerate(groups):
            merged.append({
                'id': f'h_line_{idx}',
                'start': [int(group['x1']), int(group['y'])],
                'end': [int(group['x2']), int(group['y'])],
                'length': int(group['x2'] - group['x1']),
                'type': 'horizontal'
            })

        return merged

    def _merge_vertical_lines(self, lines: List[Dict[str, Any]], threshold: int) -> List[Dict[str, Any]]:
        """수직선 병합"""
        if not lines:
            return []

        # x 좌표로 그룹화
        groups = []
        for line in lines:
            x_avg = (line['start'][0] + line['end'][0]) / 2
            y1 = min(line['start'][1], line['end'][1])
            y2 = max(line['start'][1], line['end'][1])

            # 기존 그룹에 추가 가능한지 확인
            added = False
            for group in groups:
                if abs(group['x'] - x_avg) < threshold:
                    # 같은 그룹
                    group['y1'] = min(group['y1'], y1)
                    group['y2'] = max(group['y2'], y2)
                    group['count'] += 1
                    added = True
                    break

            if not added:
                groups.append({
                    'x': x_avg,
                    'y1': y1,
                    'y2': y2,
                    'count': 1
                })

        # 결과 생성
        merged = []
        for idx, group in enumerate(groups):
            merged.append({
                'id': f'v_line_{idx}',
                'start': [int(group['x']), int(group['y1'])],
                'end': [int(group['x']), int(group['y2'])],
                'length': int(group['y2'] - group['y1']),
                'type': 'vertical'
            })

        return merged


class DrawingOCRAutoZone(DrawingOCR):
    """자동 영역 감지 OCR 클래스"""

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        self.detector = AutoZoneDetector()
       
        self.rectangles = []
        self.lines = []
        self.paths = []  # 경로 정보 저장
        self.text_regions = []  # 텍스트 영역 저장

    def auto_detect_zones(self, image_path: str, merge_lines: bool = True, remove_text: bool = True, detect_paths: bool = True, ocr_langs=['en']):
        """
        EasyOCR 기반 자동 영역 감지

        Args:
            image_path: 이미지 경로
            merge_lines: 가까운 선들을 병합할지 여부
            remove_text: 텍스트 영역을 먼저 제거할지 여부
            detect_paths: 박스-텍스트 간 경로 추출 여부
            ocr_langs: OCR 언어 리스트 (EasyOCR)
        """
        from PIL import Image
        print(f"\n자동 영역 감지 중 (EasyOCR): {Path(image_path).name}")

        text_regions = None

        # 텍스트 먼저 감지 (박스 감지 전에 제거하기 위해)
        if remove_text or detect_paths:
            print("EasyOCR로 텍스트 영역 감지 중...")

            # EasyOCR 실행 및 기존 스타일로 변환
            text_lines = self.run_easyocr_and_convert(image_path, langs=ocr_langs, min_confidence=0)

            # 텍스트 영역 리스트 생성
            text_regions = [{'bbox': tl['bbox'], 'confidence': tl['confidence']} for tl in text_lines]
            self.text_regions = text_regions

            print(f"텍스트 영역 {len(text_regions)}개 감지됨 (신뢰도 70% 이상)")

        # 사각형 감지 (텍스트 영역 제거 후)
        self.rectangles = self.detector.detect_rectangles(image_path, text_regions if remove_text else None)

        # 경로 감지 (박스-텍스트 연결)
        if detect_paths and text_regions:
            print("박스-텍스트 간 경로 추출 중...")
            self.paths = self.detector.detect_paths(image_path, self.rectangles, text_regions, distance_threshold=10)
        else:
            self.paths = []

        # 선 감지 (박스에서 시작하는 선만)
        lines = self.detector.detect_lines(image_path, rectangles=self.rectangles, distance_threshold=10)

        # 선 병합
        if merge_lines:
            self.lines = self.detector.merge_nearby_lines(lines)
        else:
            self.lines = lines

        print(f"최종 감지: 사각형 {len(self.rectangles)}개, 선 {len(self.lines)}개, 경로 {len(self.paths)}개")

        # EasyOCR 결과를 클래스에 저장 (분류 시 사용)
        self.text_lines = text_lines

    def classify_text_by_auto_zones(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        감지된 영역을 기반으로 텍스트 분류

        Args:
            result: OCR 결과

        Returns:
            영역별 분류 결과
        """
        classified = {
            'boxes': {},
            'line_regions': {},
            'outside': []
        }

        text_lines = result.get('text_lines', [])

        # 1. 사각형 영역별 분류
        for rect in self.rectangles:
            rect_id = rect['id']
            classified['boxes'][rect_id] = {
                'bbox': rect['bbox'],
                'area': rect['area'],
                'texts': []
            }

            for line in text_lines:
                if self._is_text_in_box(line, rect['bbox']):
                    classified['boxes'][rect_id]['texts'].append(line)

        # 2. 선 기준 영역별 분류
        for line in self.lines:
            line_id = line['id']

            if line['type'] == 'horizontal':
                # 선 위/아래 영역
                classified['line_regions'][f"{line_id}_above"] = {
                    'line': line,
                    'region': 'above',
                    'texts': []
                }
                classified['line_regions'][f"{line_id}_below"] = {
                    'line': line,
                    'region': 'below',
                    'texts': []
                }

                for text in text_lines:
                    if self._is_text_above_line(text, line):
                        classified['line_regions'][f"{line_id}_above"]['texts'].append(text)
                    elif self._is_text_below_line(text, line):
                        classified['line_regions'][f"{line_id}_below"]['texts'].append(text)

            elif line['type'] == 'vertical':
                # 선 좌/우 영역
                classified['line_regions'][f"{line_id}_left"] = {
                    'line': line,
                    'region': 'left',
                    'texts': []
                }
                classified['line_regions'][f"{line_id}_right"] = {
                    'line': line,
                    'region': 'right',
                    'texts': []
                }

                for text in text_lines:
                    if self._is_text_left_of_line(text, line):
                        classified['line_regions'][f"{line_id}_left"]['texts'].append(text)
                    elif self._is_text_right_of_line(text, line):
                        classified['line_regions'][f"{line_id}_right"]['texts'].append(text)

        return classified

    def _is_text_in_box(self, text: Dict[str, Any], box: List[int]) -> bool:
        """텍스트가 박스 내부에 있는지 확인"""
        bbox = text.get('bbox')
        if not bbox or len(bbox) != 4:
            return False

        tx1, ty1, tx2, ty2 = bbox
        bx1, by1, bx2, by2 = box

        # 중심점 기준
        cx = (tx1 + tx2) / 2
        cy = (ty1 + ty2) / 2

        return bx1 <= cx <= bx2 and by1 <= cy <= by2

    def _is_text_above_line(self, text: Dict[str, Any], line: Dict[str, Any]) -> bool:
        """텍스트가 수평선 위에 있는지 확인"""
        bbox = text.get('bbox')
        if not bbox or len(bbox) != 4:
            return False

        _, ty1, _, ty2 = bbox
        line_y = line['start'][1]

        # 텍스트 하단이 선보다 위에 있어야 함
        return ty2 < line_y

    def _is_text_below_line(self, text: Dict[str, Any], line: Dict[str, Any]) -> bool:
        """텍스트가 수평선 아래에 있는지 확인"""
        bbox = text.get('bbox')
        if not bbox or len(bbox) != 4:
            return False

        _, ty1, _, _ = bbox
        line_y = line['start'][1]

        # 텍스트 상단이 선보다 아래에 있어야 함
        return ty1 > line_y

    def _is_text_left_of_line(self, text: Dict[str, Any], line: Dict[str, Any]) -> bool:
        """텍스트가 수직선 왼쪽에 있는지 확인"""
        bbox = text.get('bbox')
        if not bbox or len(bbox) != 4:
            return False

        tx1, _, tx2, _ = bbox
        line_x = line['start'][0]

        # 텍스트 우측이 선보다 왼쪽에 있어야 함
        return tx2 < line_x

    def _is_text_right_of_line(self, text: Dict[str, Any], line: Dict[str, Any]) -> bool:
        """텍스트가 수직선 오른쪽에 있는지 확인"""
        bbox = text.get('bbox')
        if not bbox or len(bbox) != 4:
            return False

        tx1, _, _, _ = bbox
        line_x = line['start'][0]

        # 텍스트 좌측이 선보다 오른쪽에 있어야 함
        return tx1 > line_x

    def process_with_auto_zones(self,
                                image_path: str,
                                detect_tables: bool = True,
                                detect_layout: bool = True,
                                merge_lines: bool = True) -> Dict[str, Any]:
        """
        자동 영역 감지 + OCR 처리

        Args:
            image_path: 이미지 경로
            detect_tables: 테이블 인식 여부
            detect_layout: 레이아웃 분석 여부
            merge_lines: 선 병합 여부

        Returns:
            OCR 결과 + 자동 영역 분류
        """
        # 자동 영역 감지
        self.auto_detect_zones(image_path, merge_lines)

        # OCR 처리
        result = self.process_image(image_path, detect_tables, detect_layout)

        # 자동 영역별 분류
        auto_zones = self.classify_text_by_auto_zones(result)
        result['auto_zones'] = auto_zones

        # 통계 추가
        result['auto_zone_stats'] = self._calculate_zone_stats(auto_zones)

        return result

    def _calculate_zone_stats(self, auto_zones: Dict[str, Any]) -> Dict[str, Any]:
        """영역별 통계 계산"""
        stats = {
            'total_boxes': len(auto_zones.get('boxes', {})),
            'total_line_regions': len(auto_zones.get('line_regions', {})),
            'box_details': {},
            'line_details': {}
        }

        # 박스별 통계
        for box_id, box_data in auto_zones.get('boxes', {}).items():
            texts = box_data.get('texts', [])
            stats['box_details'][box_id] = {
                'text_count': len(texts),
                'area': box_data.get('area', 0),
                'full_text': '\n'.join([t.get('text', '') for t in texts])
            }

        # 선 영역별 통계
        for region_id, region_data in auto_zones.get('line_regions', {}).items():
            texts = region_data.get('texts', [])
            stats['line_details'][region_id] = {
                'text_count': len(texts),
                'full_text': '\n'.join([t.get('text', '') for t in texts])
            }

        return stats

    def visualize_auto_zones(self,
                            image_path: str,
                            result: Dict[str, Any],
                            output_path: str,
                            show_boxes: bool = True,
                            show_lines: bool = True,
                            show_paths: bool = True,
                            show_text: bool = True,
                            show_text_content: bool = True):
        """
        자동 감지된 영역 시각화

        Args:
            image_path: 원본 이미지
            result: OCR 결과
            output_path: 출력 경로
            show_boxes: 사각형 표시
            show_lines: 선 표시
            show_text: 텍스트 박스 표시
            show_text_content: 텍스트 내용 표시
        """
        # 이미지 로드
        image = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # 폰트
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # 1. 사각형 그리기
        if show_boxes:
            auto_zones = result.get('auto_zones', {})
            boxes = auto_zones.get('boxes', {})

            for idx, rect in enumerate(self.rectangles[:20]):  # 상위 10개만
                bbox = rect['bbox']
                box_id = rect['id']

                # 박스 내 텍스트 수 확인
                box_data = boxes.get(box_id, {})
                text_count = len(box_data.get('texts', []))

                # 도형 타입에 따른 색상
                shape_type = rect.get('type', 'rectangle')
                if shape_type == 'rectangle':
                    color = (0, 200, 100, 200)  # 녹색
                elif shape_type == 'triangle':
                    color = (200, 100, 0, 200)  # 주황색
                elif shape_type in ['pentagon', 'hexagon']:
                    color = (100, 0, 200, 200)  # 보라색
                else:
                    color = (200, 200, 0, 200)  # 노란색

                # 도형 테두리
                draw.rectangle(bbox, outline=color[:3] + (255,), width=4)

                # 라벨 배경
                vertices = rect.get('vertices', 4)
                label = f"{shape_type.upper()}-{idx + 1}"
                detail = f"{text_count}개 텍스트 | {rect['area']:,}px² | {vertices}꼭지점"

                # 상단 라벨
                label_bbox = draw.textbbox((bbox[0], bbox[1] - 45), label, font=font)
                draw.rectangle(label_bbox, fill=(0, 200, 100, 220))
                draw.text((bbox[0], bbox[1] - 45), label, fill=(255, 255, 255, 255), font=font)

                # 상세 정보
                draw.text((bbox[0], bbox[1] - 25), detail, fill=(0, 200, 100, 255), font=font_small)

        # 2. 경로 그리기 (박스-텍스트 연결)
        if show_paths and self.paths:
            for idx, path_info in enumerate(self.paths[:50]):  # 상위 50개만
                path = path_info['path']
                if len(path) < 2:
                    continue

                # 경로를 곡선으로 그리기
                color = (255, 100, 255, 200)  # 자주색

                # 경로의 모든 점을 연결해서 그리기
                for i in range(len(path) - 1):
                    draw.line([path[i], path[i+1]], fill=color, width=2)

                # 시작점과 끝점에 원 그리기
                start_point = path[0]
                end_point = path[-1]

                # 시작점 (박스 쪽)
                draw.ellipse([start_point[0]-3, start_point[1]-3,
                             start_point[0]+3, start_point[1]+3],
                            fill=(0, 255, 0, 255), outline=(0, 200, 0, 255))

                # 끝점 (텍스트 쪽)
                draw.ellipse([end_point[0]-3, end_point[1]-3,
                             end_point[0]+3, end_point[1]+3],
                            fill=(255, 0, 0, 255), outline=(200, 0, 0, 255))

                # 중간 지점에 레이블
                mid_idx = len(path) // 2
                mid_point = path[mid_idx]

                label = f"PATH-{idx + 1}"
                detail = f"{path_info['length']:.0f}px"

                # 레이블 배경
                label_bbox = draw.textbbox((mid_point[0] + 5, mid_point[1] - 15), label, font=font_small)
                draw.rectangle(label_bbox, fill=(0, 0, 0, 180))
                draw.text((mid_point[0] + 5, mid_point[1] - 15), label, fill=color, font=font_small)
                draw.text((mid_point[0] + 5, mid_point[1] + 2), detail, fill=color, font=font_small)

        # 3. 선 그리기
        if show_lines:
            for idx, line in enumerate(self.lines[:20]):  # 상위 20개만
                if line['type'] == 'horizontal':
                    color = (255, 50, 50, 255)  # 밝은 빨강
                    label_prefix = "H-LINE"
                else:
                    color = (50, 100, 255, 255)  # 밝은 파랑
                    label_prefix = "V-LINE"

                # 선 그리기
                draw.line([tuple(line['start']), tuple(line['end'])],
                         fill=color, width=3)

                # 레이블 위치
                mid_x = (line['start'][0] + line['end'][0]) // 2
                mid_y = (line['start'][1] + line['end'][1]) // 2

                # 레이블
                label = f"{label_prefix}-{idx + 1}"
                detail = f"{line['length']}px"

                # 레이블 배경
                label_bbox = draw.textbbox((mid_x + 5, mid_y - 15), label, font=font_medium)
                draw.rectangle(label_bbox, fill=(0, 0, 0, 180))
                draw.text((mid_x + 5, mid_y - 15), label, fill=color, font=font_medium)

                # 길이 정보
                draw.text((mid_x + 5, mid_y + 2), detail, fill=color, font=font_small)

        # 4. 텍스트 박스 그리기 (신뢰도 70% 이상만)
        if show_text and 'text_lines' in result:
            for text_line in result['text_lines']:
                bbox = text_line.get('bbox')
                text_content = text_line.get('text', '')
                confidence = text_line.get('confidence', 0)

                # 70% 미만은 이미 필터링되어 있지만 이중 체크
                if confidence < 0.7:
                    continue

                if bbox and len(bbox) == 4:
                    # 신뢰도에 따른 색상 (70% 이상만 표시)
                    if confidence > 0.9:
                        text_color = (0, 255, 200, 220)  # 청록색 (높은 신뢰도)
                    else:
                        text_color = (255, 165, 0, 220)  # 주황색 (중간 신뢰도 70~90%)

                    # 텍스트 박스
                    draw.rectangle(bbox, outline=text_color[:3] + (255,), width=2)

                    # 텍스트 내용 표시
                    if show_text_content and text_content:
                        # 텍스트 줄이기
                        display_text = text_content[:30] + "..." if len(text_content) > 30 else text_content

                        # 텍스트 배경
                        text_pos = (bbox[0], bbox[3] + 2)
                        text_bbox_bg = draw.textbbox(text_pos, display_text, font=font_small)
                        draw.rectangle(text_bbox_bg, fill=(0, 0, 0, 200))

                        # 텍스트
                        draw.text(text_pos, display_text, fill=text_color[:3] + (255,), font=font_small)

        # 합성 및 저장
        result_image = Image.alpha_composite(image, overlay)
        result_image = result_image.convert('RGB')

        output_file = Path(output_path)

        # 확장자가 없으면 .png 추가
        if not output_file.suffix:
            output_file = output_file / "auto_zones_viz.png"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_image.save(output_file, quality=95)

        print(f"자동 영역 시각화 저장: {output_path}")

        # 개별 요소만 있는 이미지들도 저장
        base_path = output_file.parent
        base_name = output_file.stem

        # 1. 박스만 있는 이미지
        if show_boxes:
            boxes_only_path = base_path / f"{base_name}_boxes_only.png"
            self._save_boxes_only(image_path, result, boxes_only_path)
            print(f"박스만 시각화 저장: {boxes_only_path}")

        # 2. 선만 있는 이미지
        if show_lines:
            lines_only_path = base_path / f"{base_name}_lines_only.png"
            self._save_lines_only(image_path, result, lines_only_path)
            print(f"선만 시각화 저장: {lines_only_path}")

        # 3. 경로만 있는 이미지
        if show_paths and self.paths:
            paths_only_path = base_path / f"{base_name}_paths_only.png"
            self._save_paths_only(image_path, result, paths_only_path)
            print(f"경로만 시각화 저장: {paths_only_path}")

        # 4. 텍스트만 있는 이미지
        if show_text:
            text_only_path = base_path / f"{base_name}_text_only.png"
            self._save_text_only(image_path, result, text_only_path, show_text_content)
            print(f"텍스트만 시각화 저장: {text_only_path}")

    def _save_boxes_only(self, image_path: str, result: Dict[str, Any], output_path: Path):
        """박스만 표시하는 이미지 저장"""
        image = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        auto_zones = result.get('auto_zones', {})
        boxes = auto_zones.get('boxes', {})

        for idx, rect in enumerate(self.rectangles[:10]):
            bbox = rect['bbox']
            box_id = rect['id']
            box_data = boxes.get(box_id, {})
            text_count = len(box_data.get('texts', []))

            color = (0, 200, 100, 200)
            draw.rectangle(bbox, outline=color[:3] + (255,), width=4)

            label = f"BOX-{idx + 1}"
            detail = f"{text_count}개 텍스트 | {rect['area']:,}px²"

            label_bbox = draw.textbbox((bbox[0], bbox[1] - 45), label, font=font)
            draw.rectangle(label_bbox, fill=(0, 200, 100, 220))
            draw.text((bbox[0], bbox[1] - 45), label, fill=(255, 255, 255, 255), font=font)
            draw.text((bbox[0], bbox[1] - 25), detail, fill=(0, 200, 100, 255), font=font_small)

        result_image = Image.alpha_composite(image, overlay).convert('RGB')
        result_image.save(output_path, quality=95)

    def _save_lines_only(self, image_path: str, result: Dict[str, Any], output_path: Path):
        """선만 표시하는 이미지 저장"""
        image = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        for idx, line in enumerate(self.lines[:20]):
            if line['type'] == 'horizontal':
                color = (255, 50, 50, 255)
                label_prefix = "H-LINE"
            else:
                color = (50, 100, 255, 255)
                label_prefix = "V-LINE"

            draw.line([tuple(line['start']), tuple(line['end'])], fill=color, width=3)

            mid_x = (line['start'][0] + line['end'][0]) // 2
            mid_y = (line['start'][1] + line['end'][1]) // 2

            label = f"{label_prefix}-{idx + 1}"
            detail = f"{line['length']}px"

            label_bbox = draw.textbbox((mid_x + 5, mid_y - 15), label, font=font_medium)
            draw.rectangle(label_bbox, fill=(0, 0, 0, 180))
            draw.text((mid_x + 5, mid_y - 15), label, fill=color, font=font_medium)
            draw.text((mid_x + 5, mid_y + 2), detail, fill=color, font=font_small)

        result_image = Image.alpha_composite(image, overlay).convert('RGB')
        result_image.save(output_path, quality=95)

    def _save_paths_only(self, image_path: str, result: Dict[str, Any], output_path: Path):
        """경로만 표시하는 이미지 저장"""
        image = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font_small = ImageFont.load_default()

        for idx, path_info in enumerate(self.paths[:50]):
            path = path_info['path']
            if len(path) < 2:
                continue

            # 경로를 곡선으로 그리기
            color = (255, 100, 255, 220)  # 자주색

            # 경로의 모든 점을 연결
            for i in range(len(path) - 1):
                draw.line([path[i], path[i+1]], fill=color, width=2)

            # 시작점과 끝점
            start_point = path[0]
            end_point = path[-1]

            draw.ellipse([start_point[0]-3, start_point[1]-3,
                         start_point[0]+3, start_point[1]+3],
                        fill=(0, 255, 0, 255), outline=(0, 200, 0, 255))

            draw.ellipse([end_point[0]-3, end_point[1]-3,
                         end_point[0]+3, end_point[1]+3],
                        fill=(255, 0, 0, 255), outline=(200, 0, 0, 255))

            # 레이블
            mid_idx = len(path) // 2
            mid_point = path[mid_idx]

            label = f"PATH-{idx + 1}: {path_info['source_box']} → Text-{path_info['target_text']}"
            detail = f"{path_info['length']:.0f}px"

            label_bbox = draw.textbbox((mid_point[0] + 5, mid_point[1] - 15), label, font=font_small)
            draw.rectangle(label_bbox, fill=(0, 0, 0, 180))
            draw.text((mid_point[0] + 5, mid_point[1] - 15), label, fill=color, font=font_small)
            draw.text((mid_point[0] + 5, mid_point[1] + 2), detail, fill=color, font=font_small)

        result_image = Image.alpha_composite(image, overlay).convert('RGB')
        result_image.save(output_path, quality=95)

    def _save_text_only(self, image_path: str, result: Dict[str, Any], output_path: Path, show_text_content: bool = True):
        """텍스트만 표시하는 이미지 저장"""
        image = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font_small = ImageFont.load_default()

        if 'text_lines' in result:
            for text_line in result['text_lines']:
                bbox = text_line.get('bbox')
                text_content = text_line.get('text', '')
                confidence = text_line.get('confidence', 0)

                # 70% 미만은 이미 필터링되어 있지만 이중 체크
                if confidence < 0.7:
                    continue

                if bbox and len(bbox) == 4:
                    if confidence > 0.9:
                        text_color = (0, 255, 200, 220)  # 청록색 (90% 이상)
                    else:
                        text_color = (255, 165, 0, 220)  # 주황색 (70~90%)

                    draw.rectangle(bbox, outline=text_color[:3] + (255,), width=2)

                    if show_text_content and text_content:
                        display_text = text_content[:30] + "..." if len(text_content) > 30 else text_content
                        text_pos = (bbox[0], bbox[3] + 2)
                        text_bbox_bg = draw.textbbox(text_pos, display_text, font=font_small)
                        draw.rectangle(text_bbox_bg, fill=(0, 0, 0, 200))
                        draw.text(text_pos, display_text, fill=text_color[:3] + (255,), font=font_small)

        result_image = Image.alpha_composite(image, overlay).convert('RGB')
        result_image.save(output_path, quality=95)

    def print_auto_zone_summary(self, result: Dict[str, Any]):
        """자동 영역 결과 요약 출력"""
        if 'auto_zone_stats' not in result:
            return

        stats = result['auto_zone_stats']

        print("\n" + "="*60)
        print("자동 감지 영역 분류 결과")
        print("="*60)

        print(f"\n감지된 사각형: {stats['total_boxes']}개")
        for box_id, details in list(stats['box_details'].items())[:5]:
            print(f"\n[{box_id}] - {details['text_count']}개 텍스트, 면적: {details['area']}px²")
            if details['full_text']:
                print(details['full_text'][:200])

        print(f"\n감지된 선 영역: {stats['total_line_regions']}개")
        for region_id, details in list(stats['line_details'].items())[:5]:
            print(f"\n[{region_id}] - {details['text_count']}개 텍스트")
            if details['full_text']:
                print(details['full_text'][:200])

        print("\n" + "="*60)


def auto_zone_main(argv=None):
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='자동 영역 감지 도면 OCR')
    parser.add_argument('input', help='입력 이미지 파일')
    parser.add_argument('-o', '--output', help='결과 JSON 파일', default='auto_zones_result.json')
    parser.add_argument('--visualize', '-v', action='store_true', help='결과 시각화')
    parser.add_argument('--viz-output', help='시각화 출력 파일', default='auto_zones_viz.png')
    parser.add_argument('--no-gpu', action='store_true', help='GPU 비활성화')
    parser.add_argument('--no-merge', action='store_true', help='선 병합 비활성화')
    parser.add_argument('--debug', '-d', action='store_true', help='디버깅 모드 활성화 (중간 이미지 저장)')
    parser.add_argument('--min-box-area', type=int, default=1000, help='최소 박스 면적 (기본값: 1000)')

    args = parser.parse_args(argv)

    # OCR 초기화 (디버깅 모드 포함)
    ocr = DrawingOCRAutoZone(use_gpu=not args.no_gpu)
    ocr.detector.debug = args.debug
    ocr.detector.min_box_area = args.min_box_area

    if args.debug:
        print(f"🐛 디버깅 모드 활성화됨 (debug={ocr.detector.debug})")
        print(f"   - 최소 박스 면적: {ocr.detector.min_box_area}px²")
        print(f"   - 디버그 이미지 저장 위치: output/debug/")

    # 이미지 처리
    result = ocr.process_with_auto_zones(
        args.input,
        detect_tables=True,
        detect_layout=True,
        merge_lines=not args.no_merge
    )

    # 결과 저장
    ocr.save_results([result], args.output)

    # 요약 출력
    ocr.print_auto_zone_summary(result)

    # 시각화
    if args.visualize:
        ocr.visualize_auto_zones(
            args.input,
            result,
            args.viz_output
        )

