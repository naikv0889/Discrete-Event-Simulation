"""app.py — Streamlit dashboard for ManualLineSim with batch scenarios tab."""
import time
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from manual_line_sim import StationSpec, BufferSpec, ManualLineSim, default_line

st.set_page_config(page_title="Manual Assembly Line — Simulator Dashboard", layout="wide")
st.title("📦 Manual Assembly Line — Simulator Dashboard")

# ---------------------------
# Sidebar configuration
# ---------------------------
st.sidebar.header("⚙️ Configuration")
with st.sidebar:
    st.markdown("**Simulation horizon** (seconds)")
    sim_time_s = st.slider("sim_time_s", min_value=30, max_value=43_200, value=3600, step=30)

    st.markdown("**Random seed**")
    seed = st.number_input("seed", min_value=0, max_value=10_000, value=42, step=1)

    st.markdown("**Camera dropout (probability per logged event)**")
    camera_dropout = st.slider("camera_dropout", min_value=0.0, max_value=0.9, value=0.1, step=0.05)

    st.markdown("**Buffer capacities (between S1-S2, S2-S3, S3-S4, S4-S5, S5-S6)**")
    buf1 = st.number_input("Buffer 1", min_value=0, max_value=50, value=1, step=1)
    buf2 = st.number_input("Buffer 2", min_value=0, max_value=50, value=2, step=1)
    buf3 = st.number_input("Buffer 3", min_value=0, max_value=50, value=2, step=1)
    buf4 = st.number_input("Buffer 4", min_value=0, max_value=50, value=1, step=1)
    buf5 = st.number_input("Buffer 5", min_value=0, max_value=50, value=1, step=1)

    st.markdown("**Station specs** (mean CT seconds, CV, base defect risk)")
    base_specs = default_line()
    station_specs: List[StationSpec] = []
    for s in base_specs:
        st.markdown(f"**{s.name}**")
        mean_ct = st.number_input(
            f"mean_ct_{s.name}",
            min_value=5.0,
            max_value=120.0,
            value=float(s.mean_ct),
            step=1.0,
        )
        cv = st.slider(
            f"cv_{s.name}",
            min_value=0.0,
            max_value=1.0,
            value=float(s.cv),
            step=0.05,
        )
        dr = st.number_input(
            f"defect_risk_{s.name}",
            min_value=0.0,
            max_value=0.05,
            value=float(s.defect_risk),
            step=0.001,
            format="%.3f",
        )
        station_specs.append(StationSpec(s.name, mean_ct=mean_ct, cv=cv, defect_risk=dr))

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        start_btn = st.button("▶️ Start Simulation", use_container_width=True, type="primary")
    with run_col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)

# ---------------------------
# Session state
# ---------------------------
if "sim_result" not in st.session_state or reset_btn:
    st.session_state.sim_result = None
    st.session_state.logs_df = pd.DataFrame()
    st.session_state.wall_time_s = None

# ---------------------------
# Run simulation
# ---------------------------
if start_btn:
    buffers = [buf1, buf2, buf3, buf4, buf5]
    sim = ManualLineSim(
        station_specs,
        [BufferSpec(c) for c in buffers],
        sim_time_s=sim_time_s,
        seed=seed,
        camera_dropout=camera_dropout,
    )
    t0 = time.time()
    with st.status("🚀 Starting simulation...", expanded=True) as status:
        status.write("Initializing model and event queue…")
        status.update(label="▶️ Running discrete-event engine…", state="running")
        result = sim.run()
        status.update(label="📊 Summarizing KPIs…", state="running")
    wall_elapsed = time.time() - t0
    st.toast(f"Simulation complete in {wall_elapsed:.2f}s", icon="✅")
    st.session_state.sim_result = result
    st.session_state.logs_df = result["logs"].copy()
    st.session_state.wall_time_s = wall_elapsed

res = st.session_state.sim_result
logs = st.session_state.logs_df
wall_time = st.session_state.get("wall_time_s", None)

# ---------------------------
# KPIs
# ---------------------------
with st.container():
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    if res is not None:
        kpi1.metric("Throughput / hr", f"{res['throughput_per_hour']:.1f}")
        kpi2.metric("Completed units", f"{res['completed']}")
        fpy = res.get("first_pass_yield")
        if fpy is not None and not np.isnan(fpy):
            fpy = float(np.clip(fpy, 0.0, 1.0))
        kpi3.metric("FPY", "—" if (fpy is None or np.isnan(fpy)) else f"{100 * fpy:.1f}%")
        kpi4.metric("Bottleneck", res.get("bottleneck_station", "—"))
        cam_rate = res.get("camera_frame_rate")
        kpi5.metric("Camera frames OK", "—" if cam_rate is None or np.isnan(cam_rate) else f"{100 * cam_rate:.1f}%")
        kpi6.metric("Run time (wall)", "—" if wall_time is None else f"{wall_time:.2f}s")
    else:
        kpi1.metric("Throughput / hr", "—")
        kpi2.metric("Completed units", "—")
        kpi3.metric("FPY", "—")
        kpi4.metric("Bottleneck", "—")
        kpi5.metric("Camera frames OK", "—")
        kpi6.metric("Run time (wall)", "—")

st.divider()

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(
    ["Production", "Buffers & Inventory", "Quality & Sensors", "Timeline", "Config Snapshot", "Scenarios (batch)"]
)

# ---- Tab 0: Production
with tabs[0]:
    st.subheader("Production Details")
    if logs.empty:
        st.info("Run a simulation to see production details.")
    else:
        res_ct = []
        for si, name in enumerate([s.name for s in station_specs]):
            starts = logs[(logs["event"] == "start") & (logs["station"] == si)][["item_id", "t"]]
            finishes = logs[(logs["event"] == "finish") & (logs["station"] == si)][["item_id", "t"]]
            merged = pd.merge(starts, finishes, on="item_id", suffixes=("_start", "_finish"))
            if len(merged):
                ct = (merged["t_finish"] - merged["t_start"]).describe()
                res_ct.append(
                    {"station": name, "mean_ct": ct["mean"], "p50": ct["50%"], "std": ct["std"], "n": len(merged)}
                )
            else:
                res_ct.append({"station": name, "mean_ct": np.nan, "p50": np.nan, "std": np.nan, "n": 0})

        df_ct = pd.DataFrame(res_ct)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df_ct, use_container_width=True)
        with c2:
            st.bar_chart(df_ct.set_index("station")["mean_ct"], use_container_width=True)

        st.markdown("**Recent station events**")
        last_events = (
            logs[logs["event"].isin(["start", "finish", "complete", "to_buffer"])]
            .sort_values("t", ascending=False)
            .head(30)
        )
        st.dataframe(
            last_events[["t", "event", "station_name", "item_id", "cam_has_frame", "cam_conf", "HR", "HRV", "posture_risk"]],
            use_container_width=True,
            height=400,
        )

# ---- Tab 1: Buffers & Inventory
with tabs[1]:
    st.subheader("Buffers & Inventory")
    if logs.empty:
        st.info("Run a simulation to see buffer dynamics.")
    else:
        n_buffers = len(station_specs) - 1
        buf_plots = st.columns(n_buffers)
        for i in range(n_buffers):
            dfb = logs[(logs["event"] == "to_buffer") & (logs["station"] == i)][["t", "buffer_len"]].copy()
            dfb = dfb.sort_values("t")
            with buf_plots[i]:
                st.markdown(f"**Buffer {i + 1} occupancy**")
                if len(dfb):
                    st.line_chart(dfb.set_index("t"), use_container_width=True)
                else:
                    st.caption("No pushes recorded for this buffer.")

        starts = logs[logs["event"] == "start"]["station"].value_counts().rename("starts")
        finishes = logs[logs["event"] == "finish"]["station"].value_counts().rename("finishes")
        wip_df = pd.concat([starts, finishes], axis=1).fillna(0)
        wip_df["approx_in_service"] = (wip_df["starts"] - wip_df["finishes"]).clip(lower=0)
        wip_df = wip_df.reindex(range(len(station_specs))).fillna(0)
        wip_df.index = [s.name for s in station_specs]

        st.markdown("**Approx. items in service by station (end of run)**")
        st.dataframe(wip_df[["approx_in_service"]], use_container_width=True)

# ---- Tab 2: Quality & Sensors
with tabs[2]:
    st.subheader("Quality & Sensors")
    if logs.empty:
        st.info("Run a simulation to see QC and sensor summaries.")
    else:
        qc = logs[logs["event"] == "qc"]
        if len(qc):
            defect_rate = 100 * qc["defect"].mean()
            st.metric("Defect rate", f"{defect_rate:.2f}%")
            st.dataframe(
                qc[["t", "item_id", "defect", "p_def"]].sort_values("t", ascending=False),
                use_container_width=True,
                height=260,
            )
        else:
            st.caption("No QC records (run longer or check configuration).")

        st.markdown("**Camera health**")
        cam_ok = logs["cam_has_frame"].dropna()
        cam_ok_rate = cam_ok.mean() if len(cam_ok) else np.nan
        st.write("Frames OK:", "—" if np.isnan(cam_ok_rate) else f"{100 * cam_ok_rate:.1f}%")

        st.markdown("**Notifications** (auto-generated)")
        notes = []
        cam_drop = logs[~logs["cam_has_frame"].astype(bool)]
        for _, r in cam_drop.head(50).iterrows():
            notes.append({"t": r["t"], "type": "camera", "msg": f"No frame at {r['station_name']} for item {int(r['item_id'])}"})
        high_posture = logs[logs["posture_risk"] > 0.8]
        for _, r in high_posture.head(50).iterrows():
            notes.append({"t": r["t"], "type": "ergonomics", "msg": f"High posture risk ({r['posture_risk']:.2f}) at {r['station_name']}"})
        if len(qc):
            for _, r in qc[qc["defect"] == True].head(50).iterrows():
                notes.append({"t": r["t"], "type": "quality", "msg": f"Defect detected on item {int(r['item_id'])} (p={r['p_def']:.2f})"})
        notif_df = pd.DataFrame(notes).sort_values("t", ascending=False)
        if len(notif_df):
            st.dataframe(notif_df, use_container_width=True, height=260)
        else:
            st.caption("No notifications triggered.")

# ---- Tab 3: Timeline
with tabs[3]:
    st.subheader("Event Timeline (last 200)")
    if logs.empty:
        st.info("Run a simulation to view timeline.")
    else:
        show_cols = ["t", "event", "station_name", "item_id", "cam_has_frame", "cam_conf", "HR", "HRV", "posture_risk", "buffer_len"]
        tl = logs.sort_values("t").tail(200)[show_cols]
        st.dataframe(tl, use_container_width=True, height=420)

        st.markdown("**Starts over time**")
        starts = logs[logs["event"] == "start"]["t"].value_counts().sort_index().cumsum().rename("starts_cum").to_frame()
        st.line_chart(starts, use_container_width=True)

# ---- Tab 4: Config snapshot
with tabs[4]:
    st.subheader("Configuration Snapshot & Export Logs")
    st.json(
        {
            "sim_time_s": sim_time_s,
            "seed": seed,
            "camera_dropout": camera_dropout,
            "buffers": [buf1, buf2, buf3, buf4, buf5],
            "stations": [s.__dict__ for s in station_specs],
        }
    )
    if not logs.empty:
        csv = logs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download logs (CSV)", data=csv, file_name="sim_logs.csv", mime="text/csv")

# ---- Tab 5: Batch scenarios
with tabs[5]:
    st.subheader("Batch Scenarios — compare policies")
    st.caption("Define buffer layouts and parameter grids; run replications; compare throughput/FPY.")

    buf_str = st.text_input(
        "Buffer layouts (one per line; values S1-S2,S2-S3,S3-S4,S4-S5,S5-S6)",
        value="1,2,2,0,1\n1,3,3,1,1",
    )
    drop_str = st.text_input("Camera dropout values (comma-separated)", value="0.0,0.1,0.3")
    s3_str = st.text_input("S3 mean CT values (sec, comma-separated)", value="30,35,40")
    reps = st.number_input("Replications per scenario", min_value=3, max_value=50, value=10, step=1)
    sim_override = st.number_input(
        "Simulation horizon for batch (sec)", min_value=30, max_value=43_200, value=int(sim_time_s), step=30
    )
    run_batch = st.button("▶️ Run batch", type="primary")

    if run_batch:
        buf_lines = [line.strip() for line in buf_str.splitlines() if line.strip()]
        buf_lists = []
        valid = True

        for line in buf_lines:
            try:
                parts = [int(x.strip()) for x in line.split(",")]
                assert len(parts) == 5
                buf_lists.append(parts)
            except Exception:
                st.error(f"Invalid buffer line: '{line}'. Expect 'a,b,c,d,e'.")
                valid = False
                break

        if valid:
            try:
                drops = [float(x.strip()) for x in drop_str.split(",")]
                s3_means = [float(x.strip()) for x in s3_str.split(",")]
            except Exception:
                st.error("Check dropout/S3 lists; they must be numbers.")
                valid = False

        if valid:
            with st.status("Running batch scenarios…", expanded=True) as status:
                status.write("Preparing scenarios…")
                scenarios = []
                for buf in buf_lists:
                    for d in drops:
                        for s3m in s3_means:
                            scenarios.append({"buffers": buf, "camera_dropout": d, "s3_mean": s3m})
                status.write(f"Total scenarios: {len(scenarios)} × {reps} reps")

                harness = ExperimentHarness(station_specs)
                t0 = time.time()
                results = harness.run_scenarios(scenarios, replications=int(reps), sim_time_s=float(sim_override))
                summary = harness.aggregate(results)
                wall = time.time() - t0
                status.update(label=f"Done in {wall:.2f}s", state="complete")

            st.toast(f"Batch complete: {len(scenarios)} scenarios × {int(reps)} reps", icon="✅")
            st.dataframe(summary, use_container_width=True)

            top = summary.sort_values("tput_mean", ascending=False).head(10)
            st.markdown("**Top 10 by mean throughput**")
            st.dataframe(top, use_container_width=True)

            csv_sum = summary.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download batch summary (CSV)", data=csv_sum, file_name="batch_summary.csv", mime="text/csv")

st.divider()
st.caption("Simulators for Industry — ICS Test Bed | Streamlit dashboard for manual assembly line simulator")
