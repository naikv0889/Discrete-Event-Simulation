
"""
manual_line_sim.py — Discrete‑event simulator for a 6‑station manual assembly line
with synthetic camera/wearable signals + an experiment harness for sweeps/replications.
"""
from __future__ import annotations
import math
import random
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd

@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str
    payload: Any = field(compare=False, default=None)

@dataclass
class StationSpec:
    name: str
    mean_ct: float
    cv: float = 0.2
    defect_risk: float = 0.005

@dataclass
class BufferSpec:
    capacity: int  # 0 = no buffer

class RingCounter:
    def __init__(self): self._c = 0
    def next(self) -> int:
        self._c += 1; return self._c

class Sensors:
    def __init__(self, camera_dropout: float = 0.0, rng: random.Random | None = None):
        self.camera_dropout = camera_dropout
        self.rng = rng or random.Random()
    def camera_frame(self, t: float, station_idx: int) -> Dict[str, Any]:
        if self.rng.random() < self.camera_dropout: return {"has_frame": False}
        a, b = max(0.5, 5 - station_idx*0.5), max(0.5, 2 + station_idx*0.3)
        conf = np.random.beta(a, b)
        return {"has_frame": True, "action": "assemble_step", "confidence": float(conf)}
    def wearables(self, t: float, op_id: int) -> Dict[str, Any]:
        base_hr = 72 + 3*np.sin(t/600.0 + op_id)
        fatigue = min(1.0, t / 14400.0)
        hr = base_hr + 12*fatigue + np.random.normal(0, 1.5)
        hrv = 60 - 20*fatigue + np.random.normal(0, 3)
        posture_risk = min(1.0, 0.2 + 0.6*fatigue + np.random.normal(0, 0.05))
        return {"HR": float(hr), "HRV": float(hrv), "posture_risk": float(max(0.0, posture_risk))}

class ManualLineSim:
    def __init__(self, station_specs: List[StationSpec], buffer_specs: List[BufferSpec],
                 sim_time_s: float = 3600.0, seed: int | None = None, camera_dropout: float = 0.0,
                 defect_inflation_from_posture: float = 0.5):
        assert len(buffer_specs) == len(station_specs) - 1, "buffers between stations = N-1"
        self.station_specs = station_specs; self.buffer_specs = buffer_specs
        self.sim_time_s = sim_time_s; self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else 123)
        self.seq = RingCounter(); self.now = 0.0; self.event_q: List[Event] = []
        self.buffers: List[List[int]] = [[] for _ in buffer_specs]
        self.in_service: List[Optional[int]] = [None] * len(station_specs)
        self.item_id_counter = 0; self.completed_items: List[int] = []
        self.sensors = Sensors(camera_dropout=camera_dropout, rng=self.rng)
        self.logs: List[Dict[str, Any]] = []; self.operator_ids = list(range(len(station_specs)))
        self.defect_inflation_from_posture = defect_inflation_from_posture

    def schedule(self, delay: float, kind: str, payload: Any = None):
        t = self.now + delay; self.event_q.append(Event(t, self.seq.next(), kind, payload))
        # maintain heap property manually
        import heapq as _hq; _hq.heapify(self.event_q)

    def draw_ct(self, spec: StationSpec) -> float:
        mean, cv = spec.mean_ct, spec.cv
        if cv <= 0: return mean
        sigma2 = math.log(1 + cv**2); mu = math.log(mean) - 0.5*sigma2
        return float(np.random.lognormal(mean=mu, sigma=math.sqrt(sigma2)))

    def start_item(self):
        self.item_id_counter += 1; item_id = self.item_id_counter
        if self.in_service[0] is None:
            self.in_service[0] = item_id; ct = self.draw_ct(self.station_specs[0])
            self.schedule(ct, "finish", (0, item_id)); self.log_event("start", 0, item_id)

    def try_pull_from_buffer(self, si: int):
        if si == 0:
            if self.in_service[0] is None: self.start_item()
            return
        buf = self.buffers[si-1]
        if self.in_service[si] is None and buf:
            item_id = buf.pop(0); self.in_service[si] = item_id
            ct = self.draw_ct(self.station_specs[si]); self.schedule(ct, "finish", (si, item_id))
            self.log_event("start", si, item_id)

    def push_to_buffer_or_downstream(self, si: int, item_id: int):
        if si == len(self.station_specs) - 1:
            self.completed_items.append(item_id); self.log_event("complete", si, item_id)
            self.in_service[si] = None; self.try_pull_from_buffer(si); return
        buf = self.buffers[si]; cap = self.buffer_specs[si].capacity
        if cap == 0:
            if self.in_service[si + 1] is None:
                self.in_service[si] = None; self.in_service[si + 1] = item_id
                ct = self.draw_ct(self.station_specs[si + 1])
                self.schedule(ct, "finish", (si + 1, item_id)); self.log_event("start", si + 1, item_id)
                self.try_pull_from_buffer(si)
            else:
                self.schedule(0.01, "retry_push", (si, item_id))
            return
        if len(buf) < cap:
            buf.append(item_id); self.log_event("to_buffer", si, item_id, extra={"buffer_len": len(buf)})
            self.in_service[si] = None; self.try_pull_from_buffer(si); self.try_pull_from_buffer(si+1)
        else:
            self.schedule(0.01, "retry_push", (si, item_id))

    def qc_defect_probability(self, item_id: int) -> float:
        base = sum(s.defect_risk for s in self.station_specs)
        df = pd.DataFrame([r for r in self.logs if r.get("item_id") == item_id])
        posture = df.get("posture_risk", pd.Series([], dtype=float)).fillna(0.3).mean() if len(df) else 0.3
        return min(0.8, base * (1.0 + self.defect_inflation_from_posture * posture))

    def log_event(self, event: str, si: int, item_id: int, extra=None):
        rec = {"t": self.now, "event": event, "station": si, "station_name": self.station_specs[si].name, "item_id": item_id}
        if extra: rec.update(extra)
        cam = self.sensors.camera_frame(self.now, si); wear = self.sensors.wearables(self.now, op_id=self.operator_ids[si])
        rec.update({"cam_has_frame": cam.get("has_frame", False), "cam_conf": cam.get("confidence", np.nan),
                    "HR": wear["HR"], "HRV": wear["HRV"], "posture_risk": wear["posture_risk"]})
        self.logs.append(rec)

    def run(self) -> Dict[str, Any]:
        import heapq
        self.schedule(0.0, "start", None)
        while True:
            if not self.event_q:
                if self.in_service[0] is None: self.start_item()
                self.schedule(0.01, "tick", None)
            ev = heapq.heappop(self.event_q); self.now = ev.time
            if self.now > self.sim_time_s: break
            if ev.kind == "finish":
                si, item_id = ev.payload; self.log_event("finish", si, item_id)
                if si == len(self.station_specs) - 1:
                    p_def = self.qc_defect_probability(item_id); defect = (self.rng.random() < p_def)
                    self.logs.append({"t": self.now, "event": "qc", "station": si,
                                      "station_name": self.station_specs[si].name,
                                      "item_id": item_id, "defect": defect, "p_def": p_def})
                self.push_to_buffer_or_downstream(si, item_id)
            elif ev.kind == "retry_push":
                si, item_id = ev.payload; self.push_to_buffer_or_downstream(si, item_id)
            elif ev.kind == "tick":
                for si in range(len(self.station_specs)): self.try_pull_from_buffer(si)
            elif ev.kind == "start":
                self.try_pull_from_buffer(0)
        return self.summarize()

    def summarize(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.logs)
        completed = df[df["event"] == "complete"]["item_id"].nunique()
        horizon_h = max(1e-6, self.sim_time_s/3600.0); tput = completed / horizon_h
        qc = df[df["event"] == "qc"]; fpy = float(1.0 - qc["defect"].astype(int).mean()) if len(qc) else float("nan")
        ct_stats = {}
        for si, name in enumerate([s.name for s in self.station_specs]):
            starts = df[(df["event"] == "start") & (df["station"] == si)][["item_id", "t"]]
            finishes = df[(df["event"] == "finish") & (df["station"] == si)][["item_id", "t"]]
            merged = pd.merge(starts, finishes, on="item_id", suffixes=("_start", "_finish"))
            if len(merged):
                d = (merged["t_finish"] - merged["t_start"]).describe()[["mean","std","50%"]].to_dict()
                ct_stats[name] = {"mean": float(d["mean"]), "std": float(d["std"]), "p50": float(d["50%"])}
            else:
                ct_stats[name] = {"mean": float("nan"), "std": float("nan"), "p50": float("nan")}
        import math as _m
        bneck = max(ct_stats.items(), key=lambda kv: (kv[1]["mean"] if not _m.isnan(kv[1]["mean"]) else -1))[0]
        cam = df.get("cam_has_frame", pd.Series([], dtype=float)).dropna()
        cam_rate = float(cam.mean()) if len(cam) else float("nan")
        return {"throughput_per_hour": float(tput), "completed": int(completed), "first_pass_yield": float(fpy) if not _m.isnan(fpy) else float("nan"),
                "bottleneck_station": bneck, "ct_stats": ct_stats, "camera_frame_rate": cam_rate, "logs": df}

class ExperimentHarness:
    def __init__(self, base_station_specs: List[StationSpec]):
        self.base_station_specs = base_station_specs
    def build_sim(self, buffers: List[int], sim_time_s: float, seed: int, camera_dropout: float, s3_mean: Optional[float] = None):
        specs = [StationSpec(s.name, s.mean_ct if s3_mean is None or i != 2 else s3_mean, s.cv, s.defect_risk)
                 for i, s in enumerate(self.base_station_specs)]
        buf_specs = [BufferSpec(c) for c in buffers]
        return ManualLineSim(specs, buf_specs, sim_time_s=sim_time_s, seed=seed, camera_dropout=camera_dropout)
    def run_one(self, buffers: List[int], sim_time_s: float, seed: int, camera_dropout: float, s3_mean: Optional[float] = None):
        sim = self.build_sim(buffers, sim_time_s, seed, camera_dropout, s3_mean); out = sim.run()
        row = {"buffers": tuple(buffers), "camera_dropout": camera_dropout, "s3_mean": s3_mean if s3_mean is not None else self.base_station_specs[2].mean_ct,
               "throughput_per_hour": out["throughput_per_hour"], "first_pass_yield": out["first_pass_yield"],
               "bottleneck_station": out["bottleneck_station"], "camera_frame_rate": out["camera_frame_rate"]}
        return {"row": row, "detail": out}
    def run_scenarios(self, scenarios: List[Dict[str, Any]], replications: int = 10, sim_time_s: float = 3600.0):
        results = []; base_seed = 123
        for sc_idx, sc in enumerate(scenarios):
            for r in range(replications):
                seed = base_seed + sc_idx*1000 + r
                res = self.run_one(buffers=sc["buffers"], sim_time_s=sim_time_s, seed=seed,
                                   camera_dropout=sc.get("camera_dropout", 0.0), s3_mean=sc.get("s3_mean"))
                res["scenario"] = sc; res["replication"] = r; results.append(res)
        return results
    def aggregate(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        rows = [r["row"] for r in results]; df = pd.DataFrame(rows)
        grouped = df.groupby(["buffers", "camera_dropout", "s3_mean"], dropna=False)
        summary = grouped.agg(tput_mean=("throughput_per_hour", "mean"), tput_std=("throughput_per_hour", "std"),
                              fpy_mean=("first_pass_yield", "mean"), cam_ok_mean=("camera_frame_rate", "mean"),
                              n=("throughput_per_hour", "count")).reset_index()
        return summary

def default_line() -> List[StationSpec]:
    return [
        StationSpec("S1_kitting",     22.0, 0.25, 0.002),
        StationSpec("S2_fasteners",   18.0, 0.20, 0.002),
        StationSpec("S3_alignment",   35.0, 0.30, 0.004),
        StationSpec("S4_torque",      20.0, 0.18, 0.003),
        StationSpec("S5_QC_pack",     15.0, 0.15, 0.001),
        StationSpec("S6_palletize",   12.0, 0.15, 0.001),  # <-- NEW station (edit name/params as you like)
    ]