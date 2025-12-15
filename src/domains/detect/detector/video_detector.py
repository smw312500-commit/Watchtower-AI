import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

models_folder = Path(os.environ.get("MODELS_FOLDER", "models"))


class BaseVideoDetector:
    def __init__(self, model_src: Path, **kwargs):
        self.model = YOLO(model_src, **kwargs)

    def detect(self, src: Path, dest: Path, **kwargs):
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter.fourcc(*"avc1")
        out = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, **kwargs)
            out.write(results[0].plot())

        out.release()
        cap.release()


class VideoDetectorYolo11n(BaseVideoDetector):
    def __init__(self):
        super().__init__(models_folder / "yolo11n.pt", verbose=False)


class VideoDetectorFireDetectV1(BaseVideoDetector):
    def __init__(self):
        super().__init__(models_folder / "fire_detect_v251205_1.pt", verbose=False)


# 갓길 정차 차량 감지 로직
class VideoDetectorShoulderStop:
    def __init__(self):
        self.model = YOLO(models_folder / "yolov8s.pt", verbose=False)

        self.STOP_DISTANCE = 3
        self.STOP_TIME = 0.05
        self.SKIP = 4

        # 고정좌표
        self.poly = np.array(
            [
                (459, 359),
                (519, 364),
                (283, 580),
                (110, 544),
            ],
            dtype=np.int32,
        )

        self.car_status = {}
        self.stopped_ids = set()
        self.stop_time_map = {}
        self.frame_count = 0
        self.last_boxes = []
        self.last_ids = []
        self.last_roi_occupied = False

    def detect(self, src: Path, dest: Path, **kwargs):
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            str(dest),
            cv2.VideoWriter.fourcc(*"avc1"),
            fps,
            (width, height),
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            if self.frame_count % self.SKIP != 0:
                annotated = frame.copy()

                for box, tid in zip(self.last_boxes, self.last_ids):
                    x1, y1, x2, y2 = map(int, box)

                    color = (0, 0, 255) if tid in self.stopped_ids else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID:{tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                roi_color = (0, 0, 255) if self.last_roi_occupied else (0, 255, 0)
                cv2.polylines(annotated, [self.poly], True, roi_color, 2)

                out.write(annotated)
                continue

            annotated = frame.copy()
            current_time = time.time()
            roi_occupied = False

            results = self.model.track(
                frame,
                persist=True,
                conf=0.05,
                verbose=False,
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                ids = results[0].boxes.id.int().cpu().tolist()

                self.last_boxes = boxes
                self.last_ids = ids

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    corners = [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2),
                    ]

                    inside = any(
                        cv2.pointPolygonTest(self.poly, corner, False) >= 0
                        for corner in corners
                    )

                    if not inside:
                        continue

                    roi_occupied = True

                    # 정차 판단 부분
                    if tid not in self.car_status:
                        self.car_status[tid] = (cx, cy, current_time)
                    else:
                        px, py, last_time = self.car_status[tid]
                        move_dist = abs(cx - px) + abs(cy - py)

                        if move_dist < self.STOP_DISTANCE:
                            if current_time - last_time >= self.STOP_TIME:
                                self.stopped_ids.add(tid)
                                self.stop_time_map.setdefault(tid, current_time)
                        else:
                            self.car_status[tid] = (cx, cy, current_time)

                    color = (0, 0, 255) if tid in self.stopped_ids else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID:{tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            self.last_roi_occupied = roi_occupied

            roi_color = (0, 0, 255) if roi_occupied else (0, 255, 0)
            cv2.polylines(annotated, [self.poly], True, roi_color, 2)
            out.write(annotated)

        cap.release()
        out.release()

class VideoDetectorWrongWay(BaseVideoDetector):

    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck (COCO)

    def __init__(self):
        # 팀 코드 구조: models 폴더의 yolo11n.pt를 기본으로 사용 :contentReference[oaicite:3]{index=3}
        super().__init__(models_folder / "yolo11n.pt", verbose=False)

    # -----------------------------
    # Helpers (ported from wrong_way7.py)
    # -----------------------------
    def _get_centers(self, result, allowed_classes=None):
        centers = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if allowed_classes is not None and cls_name not in allowed_classes:
                continue

            if box.id is None:
                continue

            track_id = int(box.id)
            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            centers[track_id] = (cx, cy)
        return centers

    def _get_tracks(self, result, allowed_classes=None):
        tracks = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            if allowed_classes is not None and cls_name not in allowed_classes:
                continue

            if box.id is None:
                continue

            track_id = int(box.id)
            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            tracks[track_id] = (cx, cy, bw, bh)
        return tracks

    def _lane_direction_step(
            self,
            track_id,
            move_dir,
            lane_map,
            lane_acc,
            normal_up,
            normal_down,
            cnt_thresh=10,
    ):
        if track_id in lane_map:
            return lane_map[track_id]

        acc_vec, cnt = lane_acc.get(track_id, (np.zeros(2, dtype=np.float32), 0))
        acc_vec += move_dir
        cnt += 1
        lane_acc[track_id] = (acc_vec, cnt)

        if cnt >= cnt_thresh:
            norm = np.linalg.norm(acc_vec)
            if norm < 1e-6:
                return None

            avg_vec = acc_vec / norm
            scores = {}
            if normal_up is not None:
                scores["up"] = float(np.dot(avg_vec, normal_up))
            if normal_down is not None:
                scores["down"] = float(np.dot(avg_vec, normal_down))

            if not scores:
                return None

            lane = max(scores, key=scores.get)
            lane_map[track_id] = lane
            lane_acc.pop(track_id, None)
            return lane

        return None

    def _learn_direction(
            self,
            video_path: str,
            warmup_seconds=45,
            max_match_dist=80.0,
            min_step=1.0,
            refine_iters=2,
            hysteresis_H=0.15,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        warmup_frames = int(fps * float(warmup_seconds))

        frame_count = 0
        prev_centers = None
        vectors = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count > warmup_frames:
                break

            # 차량만 추적 (네 기존 로직 동일) :contentReference[oaicite:4]{index=4}
            result = self.model.track(
                frame, persist=True, verbose=False, classes=self.VEHICLE_CLASSES
            )[0]
            centers_dict = self._get_centers(result)

            if len(centers_dict) == 0:
                prev_centers = None
                continue

            centers = np.array(list(centers_dict.values()), dtype=np.float32)

            if prev_centers is not None:
                for prev in prev_centers:
                    dists = np.linalg.norm(centers - prev, axis=1)
                    j = int(np.argmin(dists))
                    if dists[j] < max_match_dist:
                        dx, dy = centers[j] - prev
                        step = float(np.hypot(dx, dy))
                        if step < float(min_step):
                            continue
                        vectors.append([dx, dy])

            prev_centers = centers

        cap.release()

        if len(vectors) == 0:
            return None, None

        vecs = np.stack(vectors, axis=0).astype(np.float32)

        angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
        median_angle = float(np.median(angles))
        labels = np.zeros(len(vecs), dtype=np.int32)  # 0=up, 1=down
        labels[angles > median_angle] = 1

        def compute_normal(group_vecs):
            if len(group_vecs) == 0:
                return None
            m = np.mean(group_vecs, axis=0)
            n = float(np.linalg.norm(m))
            if n < 1e-6:
                return None
            return (m / n).astype(np.float32)

        normal_up = compute_normal(vecs[labels == 0])
        normal_down = compute_normal(vecs[labels == 1])
        if normal_up is None or normal_down is None:
            return normal_up, normal_down

        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        uvecs = vecs / norms

        for _ in range(int(refine_iters)):
            up_scores = np.dot(uvecs, normal_up)
            down_scores = np.dot(uvecs, normal_down)
            diff = up_scores - down_scores

            new_labels = labels.copy()
            new_labels[diff > float(hysteresis_H)] = 0
            new_labels[diff < -float(hysteresis_H)] = 1

            labels = new_labels

            new_up = compute_normal(uvecs[labels == 0])
            new_down = compute_normal(uvecs[labels == 1])

            if new_up is None or new_down is None:
                break

            normal_up, normal_down = new_up, new_down

        return normal_up, normal_down

    def _build_lane_masks(
            self,
            video_path: str,
            normal_up,
            normal_down,
            paint_seconds=45,
            min_step=1.0,
            cnt_thresh=10,
            thickness_scale=0.95,
            thickness_min=9,
            thickness_max=45,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        paint_frames = int(fps * float(paint_seconds))
        target_fps = 15.0
        frame_step = max(1, int(round(fps / target_fps)))

        ret, frame0 = cap.read()
        if not ret:
            cap.release()
            return None, None

        H, W = frame0.shape[:2]
        mask_up = np.zeros((H, W), dtype=np.uint8)
        mask_down = np.zeros((H, W), dtype=np.uint8)

        prev = {}
        lane_map = {}
        lane_acc = {}
        life_cnt = {}

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if (frame_idx % frame_step) != 0:
                continue
            if frame_idx > paint_frames:
                break

            result = self.model.track(
                frame, persist=True, verbose=False, classes=self.VEHICLE_CLASSES
            )[0]
            tracks = self._get_tracks(result)

            for tid, (cx, cy, bw, bh) in tracks.items():
                if tid not in prev:
                    prev[tid] = (cx, cy)
                    life_cnt[tid] = 1
                    continue

                px, py = prev[tid]
                dx, dy = cx - px, cy - py
                prev[tid] = (cx, cy)

                step = float(np.hypot(dx, dy))
                if step < float(min_step):
                    continue

                if step > 120.0:
                    continue

                life_cnt[tid] += 1
                if life_cnt[tid] < 10:
                    continue

                move_dir = np.array([dx, dy], dtype=np.float32) / step

                base = max(bw, bh)
                if base < 18:
                    continue

                lane = self._lane_direction_step(
                    tid,
                    move_dir,
                    lane_map,
                    lane_acc,
                    normal_up,
                    normal_down,
                    cnt_thresh=cnt_thresh,
                )
                if lane is None:
                    continue

                thickness = int(base * thickness_scale)
                thickness = max(thickness_min, min(thickness_max, thickness))

                p1 = (int(px), int(py))
                p2 = (int(cx), int(cy))

                if lane == "up":
                    cv2.line(mask_up, p1, p2, 255, thickness, cv2.LINE_AA)
                else:
                    cv2.line(mask_down, p1, p2, 255, thickness, cv2.LINE_AA)

        cap.release()
        return mask_up, mask_down

    # -----------------------------
    # Required interface (team pipeline)
    # -----------------------------
    def detect(self, src: Path, dest: Path, **kwargs):
        video_path = str(src)

        # 1) Learn directions
        normal_up, normal_down = self._learn_direction(video_path)
        if normal_up is None or normal_down is None:
            raise RuntimeError("learn_direction failed (not enough motion vectors)")

        # 2) Build lane masks
        mask_up, mask_down = self._build_lane_masks(
            video_path, normal_up, normal_down
        )
        if mask_up is None or mask_down is None:
            raise RuntimeError("build_lane_masks failed")

        # 3) Run detection and write output video (no cv2.imshow; 팀 구조는 파일 출력) :contentReference[oaicite:5]{index=5}
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter.fourcc(*"avc1")
        out = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))

        target_fps = 15.0
        frame_step = max(1, int(round(fps / target_fps)))
        threshold_frames = int(target_fps * 0.25)  # 네 기존 로직과 동일 컨셉 :contentReference[oaicite:6]{index=6}
        min_step = 1.0
        dot_threshold = 0.1

        prev_centers = {}
        wrong_counter = {}

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if (frame_idx % frame_step) != 0:
                continue

            results = self.model.track(frame, persist=True, verbose=False)
            result = results[0]
            annotated = result.plot()
            centers = self._get_centers(result)

            # overlay masks (up=green, down=blue) :contentReference[oaicite:7]{index=7}
            annotated[:, :, 1] = np.maximum(annotated[:, :, 1], mask_up)
            annotated[:, :, 0] = np.maximum(annotated[:, :, 0], mask_down)

            for track_id, (cx, cy) in centers.items():
                if track_id not in prev_centers:
                    prev_centers[track_id] = (cx, cy)
                    wrong_counter[track_id] = 0
                    continue

                prev_cx, prev_cy = prev_centers[track_id]
                dx = cx - prev_cx
                dy = cy - prev_cy
                prev_centers[track_id] = (cx, cy)

                step = float((dx * dx + dy * dy) ** 0.5)
                if step < min_step:
                    continue

                move_dir = np.array([dx, dy], dtype=np.float32) / (step + 1e-12)

                cx_i, cy_i = int(cx), int(cy)
                r = 6
                y1 = max(0, cy_i - r)
                y2 = min(mask_up.shape[0], cy_i + r + 1)
                x1 = max(0, cx_i - r)
                x2 = min(mask_up.shape[1], cx_i + r + 1)

                up_score = int(np.sum(mask_up[y1:y2, x1:x2]))
                down_score = int(np.sum(mask_down[y1:y2, x1:x2]))

                if up_score > down_score:
                    expected_normal = normal_up
                    lane_name = "up"
                elif down_score > up_score:
                    expected_normal = normal_down
                    lane_name = "down"
                else:
                    continue

                sim = float(np.dot(move_dir, expected_normal))
                if sim < -dot_threshold:
                    wrong_counter[track_id] += 1
                else:
                    wrong_counter[track_id] = max(0, wrong_counter[track_id] - 1)

                if wrong_counter[track_id] > threshold_frames:
                    # 화면 표기(로그는 팀 스타일에 맞춰 나중에 바꿔도 됨)
                    cv2.putText(
                        annotated,
                        f"WRONG WAY! ID:{track_id} lane:{lane_name}",
                        (max(10, cx_i - 60), max(30, cy_i - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    wrong_counter[track_id] = 0

                # 상태 텍스트
                color = (0, 0, 255) if wrong_counter[track_id] > 0 else (0, 255, 0)
                cv2.putText(
                    annotated,
                    f"ID:{track_id} {lane_name} cnt:{wrong_counter[track_id]}",
                    (cx_i, max(20, cy_i - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            out.write(annotated)

        out.release()
        cap.release()
