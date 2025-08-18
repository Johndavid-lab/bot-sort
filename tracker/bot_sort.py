import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import sys
sys.path.append("/home/cat/BoTSORT") 
import logging
import os
from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

#from fast_reid.fast_reid_interfece import FastReIDInterface


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 配置 logging
log_path = os.path.join(log_dir, "miss_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),  # 写入文件
        logging.StreamHandler()  # 同时输出到控制台（可选）
    ]
)

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50,size_primary_period=10):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        self.area_prev = None            # 上一次成功匹配时的面积
        self.last_match_frame = None    # 上一次成功匹配的帧号

        self.size_ema_periods = (10, 20)
        self.size_ema_alpha   = {p: 2.0 / (p + 1) for p in self.size_ema_periods}  # EMA 系数
        self.size_ema         = {p: None for p in self.size_ema_periods}            # {period: (ew,eh)}
        self.size_primary_period = size_primary_period
        self.size_ema_min_conf = 0.45      # 置信度低的检测不写入 EMA，防脏

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        ew, eh = float(self.tlwh[2]), float(self.tlwh[3])
        for p in self.size_ema_periods:
            self.size_ema[p] = (ew, eh)
        self.prev_tlwh = None

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        # 刷新尺寸均线
        ew, eh = float(self.tlwh[2]), float(self.tlwh[3])
        for p in self.size_ema_periods:
            self.size_ema[p] = (ew, eh)
        self.prev_tlwh = None

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.prev_tlwh = self.tlwh.copy() if (self.mean is not None) else None

        # 目标帧信息
        self.frame_id = frame_id
        self.tracklet_len += 1

        # 取检测 tlwh（后面做 EMA 和观测构造）
        new_tlwh = new_track.tlwh
        dw, dh = float(new_tlwh[2]), float(new_tlwh[3])

        # EMA：用这次检测的尺寸更新各周期均线（置信度门控）
        det_conf = float(getattr(new_track, 'score', getattr(self, 'score', 1.0)))
        self._size_ema_update(dw, dh, det_conf)

        # 从主均线取平滑尺寸，若还没初始化则回退检测尺寸
        sw, sh = self._size_from_primary_ema_or(dw, dh)

        # 单帧尺寸“夹紧”避免跳变过大：相对当前预测尺寸限制倍率
        clip = 1.25
        pw, ph = float(self.tlwh[2]), float(self.tlwh[3])
        sw = np.clip(sw, pw / clip, pw * clip)
        sh = np.clip(sh, ph / clip, ph * clip)

        # 构造观测：中心用检测中心，尺寸用 EMA 尺寸
        xywh = self.tlwh_to_xywh(new_tlwh)
        xywh[2], xywh[3] = sw, sh

        # KF 更新（只更新一次）
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, xywh)

        # 特征 EMA
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # 状态字段
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        # 记录“上一帧命中”的面积和帧号（供 miss 日志用）
        self.area_prev = float(self.tlwh[2] * self.tlwh[3])
        self.last_match_frame = frame_id

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

    def _size_ema_update(self, w: float, h: float, det_conf: float = 1.0):
        if det_conf < getattr(self, 'size_ema_min_conf', 0.45):
            return
        for p in self.size_ema_periods:
            a = self.size_ema_alpha[p]
            cur = self.size_ema[p]
            if cur is None:
                self.size_ema[p] = (float(w), float(h))
            else:
                ew, eh = cur
                self.size_ema[p] = (a*float(w) + (1.0-a)*ew,
                                    a*float(h) + (1.0-a)*eh)

    def _size_from_primary_ema_or(self, default_w: float, default_h: float):
        p = getattr(self, 'size_primary_period', 10)
        cur = self.size_ema.get(p, None)
        return (float(default_w), float(default_h)) if cur is None else (float(cur[0]), float(cur[1]))

class BoTSORT(object):
    def __init__(self, args, frame_rate=30):
        # 轨迹容器
        self.tracked_stracks  = []  # type: list[STrack]
        self.lost_stracks     = []  # type: list[STrack]
        self.removed_stracks  = []  # type: list[STrack]
        BaseTrack.clear_count()

        # 基本配置
        self.frame_id = 0
        self.args     = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh  = args.track_low_thresh
        self.new_track_thresh  = args.new_track_thresh

        self.buffer_size   = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.frame_rate = frame_rate
        # ReID / 距离门限
        self.proximity_thresh  = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        self.encoder = None           # 不使用内置 FastReID
        self.feat_dim = getattr(args, "feat_dim", 128)  # 你的 RKNNEmbedder 输出维度（改成一致）

        # 相机运动补偿（GMC/CMC）
        self.gmc = GMC(method=getattr(args, "cmc_method", "none"),
                       verbose=[getattr(args, "name", ""), getattr(args, "ablation", False)])

        # 外部特征缓存（与本帧 input detections 一一对应的全量 embeddings）
        self._ext_embeds = None

        self.size_primary_period = getattr(args, "size_primary_period", 10)
        
    # ----- 外部特征入口：在调用 update() 前先传进来 -----
    def set_external_embeds(self, feats):
        """
        feats: (N, D) float32，顺序与本帧 output_results 一一对应。
        如果传 None，表示本帧不使用外部特征。
        """
        if feats is None:
            self._ext_embeds = None
            return
        feats = np.asarray(feats, dtype=np.float32)
        n = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        self._ext_embeds = feats / n

    # ----- 主更新函数：input => output_results: Nx6 [x1,y1,x2,y2,score,cls] -----
    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks    = []
        lost_stracks      = []
        removed_stracks   = []

        # --- 标准化输入 ---
        if output_results is None:
            output_results = np.empty((0, 6), dtype=np.float32)
        else:
            output_results = np.asarray(output_results, dtype=np.float32)

        if len(output_results):
            if output_results.shape[1] >= 6:
                bboxes  = output_results[:, :4]
                scores  = output_results[:, 4]
                classes = output_results[:, 5]
            elif output_results.shape[1] == 5:
                bboxes  = output_results[:, :4]
                scores  = output_results[:, 4]
                classes = np.zeros((len(scores),), dtype=np.float32)
            else:
                bboxes  = np.empty((0, 4), dtype=np.float32)
                scores  = np.empty((0,),    dtype=np.float32)
                classes = np.empty((0,),    dtype=np.float32)

            # 低阈值过滤
            lowest_inds = scores > self.track_low_thresh
            bboxes  = bboxes[lowest_inds]
            scores  = scores[lowest_inds]
            classes = classes[lowest_inds]

            # 高阈值筛选（用于第一阶段关联）
            remain_inds  = scores > self.track_high_thresh
            dets         = bboxes[remain_inds]          # (K,4) tlbr
            scores_keep  = scores[remain_inds]          # (K,)
            classes_keep = classes[remain_inds]         # (K,)
        else:
            bboxes  = np.empty((0, 4), dtype=np.float32)
            scores  = np.empty((0,),    dtype=np.float32)
            classes = np.empty((0,),    dtype=np.float32)
            dets         = np.empty((0, 4), dtype=np.float32)
            scores_keep  = np.empty((0,),    dtype=np.float32)
            classes_keep = np.empty((0,),    dtype=np.float32)

        # -------------------- 提特征 + 构造 detections --------------------
        if len(dets) > 0:
            # 裁剪（与 dets 对齐）
            dets_int = dets.astype(int).copy()
            H, W     = img.shape[:2]
            dets_int[:, [0, 2]] = np.clip(dets_int[:, [0, 2]], 0, W - 1)
            dets_int[:, [1, 3]] = np.clip(dets_int[:, [1, 3]], 0, H - 1)
            crops = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in dets_int]

            if getattr(self.args, "with_reid", False):
                feats_keep = None

                # 外部特征优先：_ext_embeds 与原始 output_results 一一对应
                if self._ext_embeds is not None:
                    feats_all = self._ext_embeds
                    self._ext_embeds = None  # 用过即清

                    if len(feats_all) == len(scores):
                        feats_low  = feats_all[lowest_inds]
                        feats_keep = feats_low[remain_inds]
                    elif len(feats_all) == len(bboxes):
                        feats_low  = feats_all[lowest_inds]
                        feats_keep = feats_low[remain_inds]
                    elif len(feats_all) == len(dets):
                        feats_keep = feats_all
                    else:
                        feats_keep = None

                    # 长度不一致（极端情况） → 截断到最小长度
                    if feats_keep is not None and len(feats_keep) != len(dets):
                        m = min(len(feats_keep), len(dets))
                        feats_keep   = feats_keep[:m]
                        dets         = dets[:m]
                        scores_keep  = scores_keep[:m]
                        classes_keep = classes_keep[:m]

                    if feats_keep is not None:
                        n = np.linalg.norm(feats_keep, axis=1, keepdims=True) + 1e-12
                        feats_keep = feats_keep / n

                # 回退到内置 encoder
                if feats_keep is None:
                    if self.encoder is not None and len(crops):
                        feats_keep = self.encoder.inference(crops).astype(np.float32, copy=False)
                        n = np.linalg.norm(feats_keep, axis=1, keepdims=True) + 1e-12
                        feats_keep = feats_keep / n
                    else:
                        # 改成你的 ReID 维度（如 128/256）
                        feats_keep = np.zeros((len(dets), 128), dtype=np.float32)

                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f,size_primary_period=self.size_primary_period)
                              for tlbr, s, f in zip(dets, scores_keep, feats_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s,size_primary_period=self.size_primary_period)
                              for tlbr, s in zip(dets, scores_keep)]
        else:
            detections = []

        # -------------------- BoT-SORT 两阶段关联流程 --------------------
        # 划分已确认/未确认轨迹
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 预测位置（KF）
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # CMC
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # 第一阶段：与高分框关联
        ious_dists      = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not getattr(self.args, "mot20", False):
            ious_dists = matching.fuse_score(ious_dists, detections)

        if getattr(self.args, "with_reid", False):
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det   = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二阶段：与低分框再关联
        if len(scores):
            inds_high   = scores < self.track_high_thresh
            inds_low    = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second   = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second= classes[inds_second]
        else:
            dets_second   = np.empty((0, 4), dtype=np.float32)
            scores_second = np.empty((0,),    dtype=np.float32)
            classes_second= np.empty((0,),    dtype=np.float32)

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                                 for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det   = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        time_sec = self.frame_id / self.frame_rate

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                tlwh = track.tlwh
                x, y, w, h = tlwh
                area_now = float(w * h)
                area_prev = getattr(track, 'area_prev', None)
                ratio = (area_now / area_prev) if (area_prev is not None and area_prev > 1e-6) else float('nan')

                prev_tlwh = getattr(track, "prev_tlwh", None)
                if prev_tlwh is not None:
                    px, py, pw, ph = prev_tlwh
                    prev_str = f"prev_x={px:.1f} prev_y={py:.1f} prev_w={pw:.1f} prev_h={ph:.1f}"
                else:
                    prev_str = "prev=NA"

                logging.info(
                    f"[MISS_START] t={time_sec:.2f}s "
                    f"frame={self.frame_id} "
                    f"id={int(getattr(track,'track_id',-1))} "
                    f"x={x:.1f} y={y:.1f} w={w:.1f} h={h:.1f} "
                    f"{prev_str} "
                    f"area_now={area_now:.1f} "
                    f"area_prev={area_prev if area_prev is not None else 'NA'} "
                    f"ratio={ratio:.3f}"
                )

                track.mark_lost()
                lost_stracks.append(track)
        # 处理未确认轨迹
        detections_u = [detections[i] for i in u_detection]
        ious_dists   = matching.iou_distance(unconfirmed, detections_u)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not getattr(self.args, "mot20", False):
            ious_dists = matching.fuse_score(ious_dists, detections_u)

        if getattr(self.args, "with_reid", False):
            emb_dists = matching.embedding_distance(unconfirmed, detections_u) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_u[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 初始化新轨
        for inew in u_detection:
            track = detections_u[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # 超时清理
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        


        # 合并结果集
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # 输出全部已存在轨迹（如需仅输出 is_activated，可在此过滤）
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
