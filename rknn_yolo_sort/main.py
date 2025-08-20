import cv2, time
import numpy as np
from collections import deque
from rknnpool import rknnPoolExecutor
from types import SimpleNamespace
from BoTSORT.tracker.bot_sort import BoTSORT
from func import det_func,reid_func
# —— 按你的路径改 —— 
DET_RKNN  = '/home/cat/yolov8s_fp16.rknn'
REID_RKNN = '/home/cat/deep_sort_realtime/embedder/weights/mobilenetv2_embedder_224224_i8.rknn'
CAP_PROP_FPS = 30
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('无法打开摄像头')

    # —— BoT-SORT 初始化（沿用你原来的 args）——
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    cap_fps = cap_fps if cap_fps and cap_fps > 0 else 30
    args = SimpleNamespace(
        track_high_thresh=0.3, track_low_thresh=0.05, new_track_thresh=0.3,
        match_thresh=0.8, with_reid=True, proximity_thresh=0.6, appearance_thresh=0.25,
        cmc_method='none', mot20=False, fast_reid_config='', fast_reid_weights='',
        device='cpu', track_buffer=40, name='rk3588', ablation=False,
        size_primary_period=10, size_alpha_mix=0.5
    )
    tracker = BoTSORT(args, frame_rate=cap_fps)

    # —— 两个 RKNN 线程池：Det 多核、ReID 1 核（可按机子调）——
    pool_det  = rknnPoolExecutor(rknnModel=DET_RKNN,  TPEs=3, func=det_func)
    pool_reid = rknnPoolExecutor(rknnModel=REID_RKNN, TPEs=1, func=reid_func)

    # —— 预热（减少起步抖动）——
    fid = -1
    warm = pool_det.TPEs + 1
    for _ in range(warm):
        ok, fr = cap.read()
        if not ok: break
        fid += 1
        pool_det.put((fid, fr))

    pending_det  = {}   # fid -> (dets, frame_bgr, kept_idx)
    pending_reid = {}   # fid -> embeds
    next_fid = 0

    prev_time = time.time()
    fps_win = deque(maxlen=30)
    TIMEOUT_S = 0.08   # 对齐超时，避免卡死
    last_show = time.time()

    print('🚀 开始摄像头实时推理，按 ESC 退出')
    try:
        while True:
            # 采集并投喂 det
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            pool_det.put((fid, frame))

            # 尽量多取 det 结果
            while True:
                res, ok = pool_det.get()
                if not ok:
                    break
                dfid, dets, fr = res

                # 与 ReID 对齐：对 dets 进行同一套筛选，并记录 kept_idx
                H0, W0 = fr.shape[:2]
                crops, kept_idx = [], []
                if dets:
                    for i, ((x1,y1,x2,y2), sc, _cid) in enumerate(dets):
                        if sc < 0.5 or (x2-x1) < 16 or (y2-y1) < 16:
                            continue
                        x1 = max(0, min(x1, W0-1)); y1 = max(0, min(y1, H0-1))
                        x2 = max(0, min(x2, W0-1)); y2 = max(0, min(y2, H0-1))
                        if x2 > x1 and y2 > y1:
                            crops.append(fr[y1:y2, x1:x2]); kept_idx.append(i)

                pending_det[dfid] = (dets, fr, kept_idx)
                # 投喂 ReID
                pool_reid.put((dfid, crops))

            # 尽量多取 reid 结果
            while True:
                res, ok = pool_reid.get()
                if not ok:
                    break
                rfid, embeds = res
                pending_reid[rfid] = embeds

            # 按 fid 对齐：同一 fid 的 det 和 reid 都到齐再喂 tracker
            while next_fid in pending_det and next_fid in pending_reid:
                dets, fr, kept_idx = pending_det.pop(next_fid)
                embeds = pending_reid.pop(next_fid)

                # 只用与 ReID 一致的索引，保证数量/顺序一致
                sel_dets = [dets[i] for i in kept_idx] if dets and kept_idx else []
                if sel_dets:
                    output_results = np.array(
                        [[x1,y1,x2,y2,float(sc),0] for ((x1,y1,x2,y2), sc, _cid) in sel_dets],
                        dtype=np.float32
                    )
                    # 容错：长度不一致时截断到最短
                    if embeds.shape[0] != output_results.shape[0]:
                        m = min(embeds.shape[0], output_results.shape[0])
                        embeds = embeds[:m]
                        output_results = output_results[:m]
                else:
                    output_results = np.empty((0,6), np.float32)
                    # 若无框，也要给一个空 embeds
                    embeds = np.zeros((0, embeds.shape[1] if embeds.ndim==2 else 128), np.float32)

                tracker.set_external_embeds(embeds)
                tracks = tracker.update(output_results, fr)

                # 画框＋ID
                for t in tracks:
                    if hasattr(t,'tlwh'):
                        x,y,w,h = t.tlwh; x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
                    else:
                        x1,y1,x2,y2 = map(int, t.tlbr)
                    tid = int(getattr(t,'track_id', getattr(t,'id',-1)))
                    cv2.rectangle(fr,(x1,y1),(x2,y2),(0,255,0),1)
                    cv2.putText(fr,f'ID {tid}',(x1,max(0,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

                # FPS
                now = time.time()
                dt  = now - prev_time
                prev_time = now
                if dt > 0:
                    fps_win.append(1.0/dt)
                cv2.putText(fr, f'FPS: { (sum(fps_win)/len(fps_win)):.1f }',
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1)

                cv2.imshow('det+reid dual-pool', fr)
                last_show = now
                if (cv2.waitKey(1) & 0xFF) == 27:
                    raise KeyboardInterrupt

                next_fid += 1

            # 超时跳过，避免卡在缺失的某个 fid
            if time.time() - last_show > TIMEOUT_S:
                next_fid += 1
                last_show = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pool_det.release()
        pool_reid.release()
        print('✅ 结束')

if __name__ == '__main__':
    main()
