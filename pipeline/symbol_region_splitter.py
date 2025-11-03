import cv2
import numpy as np
from utils import show_image_highres

def non_max_suppression_fast(boxes, overlapThresh=0.3):
    # Issue : detections 리스트에는 중복이 많음(비슷한 위치에 중복으로 잡히는 것 같음)
    # 중복 부분 제거하기 위한 작업
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]

        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs]
        idxs = idxs[overlap <= overlapThresh]

    return boxes[pick].astype(int)

def match_symbol(drawing_gray, symbol_gray, threshold=0.75):
    # 도면에서 심볼 매칭 테스트 
    # 모든 탐지 좌표 수집(중복 제거 함수 활용)
    
    res = cv2.matchTemplate(drawing_gray, symbol_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    h, w = symbol_gray.shape

    rectangles = [[pt[0], pt[1], pt[0]+w, pt[1]+h] for pt in zip(*loc[::-1])]
    filtered_boxes = non_max_suppression_fast(rectangles)
    print(f"탐지된 심볼 수 (중복 제거 후): {len(filtered_boxes)}")
    return filtered_boxes
    
def group_close_coords(coords, threshold=50):
    if not coords:
        return []
    groups = []
    current_group = [coords[0]]
    for val in coords[1:]:
        if abs(val - current_group[-1]) <= threshold:
            current_group.append(val)
        else:
            groups.append(int(np.mean(current_group)))
            current_group = [val]
    groups.append(int(np.mean(current_group)))
    return groups

def split_and_extract_regions(drawing, centers_x, centers_y, fallback=20, zoom_factor=2.0):
    height, width = drawing.shape[:2]
    grouped_x = group_close_coords(sorted(centers_x))
    grouped_y = group_close_coords(sorted(centers_y))

    split_x = [0] + grouped_x + [width]
    split_y = [0] + grouped_y + [height]

    regions = []
    region_id = 1

    for i in range(len(split_y)-1):
        for j in range(len(split_x)-1):
            y1, y2 = split_y[i], split_y[i+1]
            x1, x2 = split_x[j], split_x[j+1]

            if (y2 - y1) < fallback:
                y2 = min(height, y1 + fallback)
            if (x2 - x1) < fallback:
                x2 = min(width, x1 + fallback)

            roi = drawing[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_zoom = cv2.resize(roi, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
            # show_image_highres(roi_zoom, f"Region {region_id}", zoom=1.0, dpi=150)        # 잠시 시각화 off
            regions.append((region_id, roi_zoom))
            region_id += 1

    return regions