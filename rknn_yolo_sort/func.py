import cv2
import numpy as np
from rknnlite.api import RKNNLite
from types import SimpleNamespace
from BoTSORT.tracker.bot_sort import BoTSORT
from deep_sort_realtime.embedder.embedder_mobilenet2_onnx import RKNNEmbedder
import time 
from collections import deque
from rknnpool import rknnPoolExecutor

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    expx = np.exp(x)
    return expx / np.sum(expx, axis=axis, keepdims=True)

def nms_numpy(boxes, scores, iou_th=0.45):
    if boxes.shape[0] == 0: return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(ovr <= iou_th)[0]
        order = order[inds + 1]
    return keep

def _decode_dfl(reg_hw64, stride):
    H, W, _ = reg_hw64.shape
    reg  = reg_hw64.reshape(H, W, 4, DFL_DIM)
    prob = softmax(reg, axis=-1)
    idx  = np.arange(DFL_DIM, dtype=np.float32)
    dist = (prob * idx).sum(-1) * stride
    return dist.reshape(-1, 4)

def rk_yolov8_decode_and_nms(outputs, orig_hw, conf_th=CONF_TH, iou_th=IOU_TH,
                             keep_classes={0}, class_agnostic=True):
    H0, W0 = orig_hw
    groups = {}
    for out in outputs:
        n, c, h, w = out.shape
        arr = out[0].astype(np.float32)
        groups.setdefault((h, w), {})[c] = arr

    all_boxes, all_scores, all_cls = [], [], []
    for (h, w), parts in groups.items():
        print(len(groups))
        print(len(parts))
        if not {64, 80, 1}.issubset(parts):  # 回归/分类/obj 都要有
            continue
        stride = STRIDES[h]

        reg  = np.transpose(parts[64], (1, 2, 0))  # (H,W,64)
        dist = _decode_dfl(reg, stride)

        gy, gx = np.indices((h, w))
        cx = ((gx + 0.5) * stride).reshape(-1, 1)
        cy = ((gy + 0.5) * stride).reshape(-1, 1)

        l, t, r, b = np.split(dist, 4, axis=1)
        x1 = cx - l; y1 = cy - t; x2 = cx + r; y2 = cy + b
        boxes = np.concatenate([x1, y1, x2, y2], axis=1)  # (H*W,4)

        obj = sigmoid(parts[1].reshape(-1, 1))
        cls = sigmoid(np.transpose(parts[80], (1, 2, 0))).reshape(-1, NUM_CLASS)
        scores_all = obj * cls

        if keep_classes is None:
            scores = scores_all.max(1)
            cls_ids = scores_all.argmax(1)
        else:
            keep_idx = np.array(sorted(list(keep_classes)), dtype=int)
            sub_scores = scores_all[:, keep_idx]
            cls_local = sub_scores.argmax(1)
            scores    = sub_scores[np.arange(sub_scores.shape[0]), cls_local]
            cls_ids   = keep_idx[cls_local]

        mask = scores >= conf_th
        if not np.any(mask): continue
        all_boxes.append(boxes[mask]); all_scores.append(scores[mask]); all_cls.append(cls_ids[mask])

    if not all_boxes: return []
    boxes  = np.concatenate(all_boxes,  axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_ids= np.concatenate(all_cls,    axis=0)

    keep_idx = nms_numpy(boxes, scores, iou_th) if class_agnostic else []
    if not class_agnostic:
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            keep_idx.extend(idx[nms_numpy(boxes[idx], scores[idx], iou_th)])
    keep_idx = np.array(keep_idx, dtype=int) if not class_agnostic else keep_idx

    boxes, scores, cls_ids = boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]
    scale_w, scale_h = W0 / IMG_SIZE, H0 / IMG_SIZE
    boxes[:, [0, 2]] *= scale_w; boxes[:, [1, 3]] *= scale_h
    boxes = boxes.round().astype(int)

    return [([int(x1), int(y1), int(x2), int(y2)], float(sc), int(ci))
            for (x1, y1, x2, y2), sc, ci in zip(boxes, scores, cls_ids)]

# --- YOLO worker: (fid, frame_bgr) -> (fid, dets, frame_bgr) ---
def det_func(rknn_det, payload):
    fid, frame_bgr = payload
    H0, W0 = frame_bgr.shape[:2]
    # 走 NN 要 NHWC uint8；保持你原先的 640x640 resize（或 letterbox）
    inp = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
    inp = inp[:, :, ::-1]            # BGR->RGB
    inp = inp[None, ...].astype(np.uint8)  # [1,H,W,3] NHWC uint8
    out = rknn_det.inference(inputs=[inp], data_format=['nhwc'])
    dets = rk_yolov8_decode_and_nms(out, (H0, W0),
                                    conf_th=CONF_TH, iou_th=IOU_TH, keep_classes={0})
    # 返回原图用于后续画框/裁剪
    return (fid, dets, frame_bgr)

# --- ReID worker: (fid, crops_bgr_list) -> (fid, embeds_np) ---
def reid_func(rknn_reid, payload):
    fid, crops = payload
    if not crops:
        return (fid, np.zeros((0, 128), np.float32))  # 按你的 ReID 维度
    batch = []
    for im in crops:
        im = cv2.resize(im, (224, 224))
        # 你的 ReID 如果是 BGR & 手动归一化，这里保持一致；否则改成 RGB/减均值除方差
        batch.append(im)
    # 如果 reid 模型不支持 batch，就循环跑
    embeds = []
    for im in batch:
        x = im[None, ...].astype(np.uint8)  # NHWC
        feat = rknn_reid.inference(inputs=[x], data_format=['nhwc'])[0].astype(np.float32).squeeze()
        embeds.append(feat)
    embeds = np.stack(embeds, axis=0) if embeds else np.zeros((0, 128), np.float32)
    return (fid, embeds)