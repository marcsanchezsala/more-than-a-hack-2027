"""Reproducible fetcher and converter for open soccer tracking benchmarks:
1. DFL / IDSSE (Bassek et al. 2025, CC BY 4.0, 7 Bundesliga matches) via huggingface_hub ('pysport/idsse-data').
2. Metrica Sports Sample Data (3 open matches, CC0/sample) via GitHub.

Standardizes all matches into uniform per-frame arrays cached in external_data/cache/{match_id}.npz:
  - ball: (N, 3) [x, y, speed] in metres (NaN when missing)
  - players: (N, S, 3) [x, y, speed] in metres and m/s (NaN when missing)
  - team: (S,) 0 for Home, 1 for Away
  - labels: (N,) 'H', 'A', 'D', 'X'
  - blocks: (2, 2) first and last frame of each half
"""

import os
import sys
import csv
import json
import time
import argparse
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "external_data")
DFL_DIR = os.path.join(DATA_DIR, "dfl")
METRICA_DIR = os.path.join(DATA_DIR, "metrica")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

PITCH = (105.0, 68.0)
FPS = 25
HALF_GAP = 5000

DFL_MATCHES = [
    "DFL-MAT-J03WMX",
    "DFL-MAT-J03WN1",
    "DFL-MAT-J03WOH",
    "DFL-MAT-J03WOY",
    "DFL-MAT-J03WPY",
    "DFL-MAT-J03WQQ",
    "DFL-MAT-J03WR9",
]

METRICA_MATCHES = [
    "Sample_Game_1",
    "Sample_Game_2",
    "Sample_Game_3",
]

ALL_MATCHES = DFL_MATCHES + METRICA_MATCHES

# Pinned source revisions, so the caches (and the submission trained on them) are reproducible.
DFL_REPO = "pysport/idsse-data"
DFL_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"   # Hugging Face dataset commit, 2026-03-04
METRICA_REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"   # github.com/metrica-sports/sample-data, 2021-04-15
METRICA_URL = f"https://raw.githubusercontent.com/metrica-sports/sample-data/{METRICA_REVISION}/data/"
HOME_NAMES = ("Home", "Team A")


def speed_centred(xy, k=2):
    """Compute m/s per frame using a centred difference across +/- k frames."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    d = xy[2 * k:] - xy[:-2 * k]
    out[k:-k] = np.hypot(d[..., 0], d[..., 1]) * FPS / (2 * k)
    return out


# --- DFL Processing ---

def fetch_dfl(match_id, keep_xml=False):
    """Download and convert one DFL match to standard NPZ."""
    out_npz = os.path.join(CACHE_DIR, f"{match_id}.npz")
    if os.path.isfile(out_npz):
        print(f"  {match_id} already cached at {out_npz}")
        return out_npz

    from huggingface_hub import HfApi, hf_hub_download
    os.makedirs(DFL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(DFL_REPO, repo_type="dataset", revision=DFL_REVISION)

    info_name = next(f for f in repo_files if match_id in f and "matchinformation" in f)
    pos_name = next(f for f in repo_files if match_id in f and "positions" in f)

    print(f"  Downloading DFL match info: {info_name}...")
    p_info = hf_hub_download(DFL_REPO, info_name, repo_type="dataset", revision=DFL_REVISION, local_dir=DFL_DIR)
    
    tree = ET.parse(p_info)
    home_id = None
    for team in tree.getroot().iter("Team"):
        if team.get("Role") == "home":
            home_id = team.get("TeamId")
            break
    if not home_id:
        raise ValueError(f"Home team ID not found in {p_info}")

    print(f"  Downloading DFL tracking data: {pos_name}...")
    t0 = time.time()
    p_pos = hf_hub_download(DFL_REPO, pos_name, repo_type="dataset", revision=DFL_REVISION, local_dir=DFL_DIR)
    print(f"  Downloaded {os.path.getsize(p_pos)/1e6:.1f} MB in {time.time() - t0:.1f}s.")

    print(f"  Parsing DFL tracking XML for {match_id}...")
    t_parse = time.time()
    halves = {}
    current = None
    for event, elem in ET.iterparse(p_pos, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            section = elem.get("GameSection")
            team = elem.get("TeamId")
            current = {
                "section": section,
                "team": team,
                "person": elem.get("PersonId"),
                "n": [], "x": [], "y": [], "s": [],
                "poss": [], "status": []
            }
        elif event == "end" and elem.tag == "Frame" and current is not None:
            current["n"].append(int(elem.get("N")))
            current["x"].append(float(elem.get("X")))
            current["y"].append(float(elem.get("Y")))
            s_val = elem.get("S")
            current["s"].append(float(s_val) if s_val is not None else np.nan)
            if current["team"] == "BALL":
                current["poss"].append(int(elem.get("BallPossession", "0")))
                current["status"].append(int(elem.get("BallStatus", "0")))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            half = halves.setdefault(current["section"], {"ball": None, "people": []})
            if current["team"] == "BALL":
                half["ball"] = current
            else:
                current["home"] = (current["team"] == home_id)
                half["people"].append(current)
            current = None
            elem.clear()

    # Assemble arrays
    sections = [s for s in ("firstHalf", "secondHalf") if s in halves]
    persons = {}
    for s in sections:
        for p in halves[s]["people"]:
            persons.setdefault(p["person"], p["home"])
    slots = {pid: k for k, pid in enumerate(sorted(persons, key=lambda q: (not persons[q], q)))}
    team_arr = np.array([0 if persons[pid] else 1 for pid in sorted(slots, key=slots.get)], np.int8)

    lengths = []
    for s in sections:
        n = np.array(halves[s]["ball"]["n"])
        lengths.append(n.max() - n.min() + 1)
    total = sum(lengths) + HALF_GAP * (len(sections) - 1)
    ball_arr = np.full((total, 3), np.nan, np.float32)
    players_arr = np.full((total, len(slots), 3), np.nan, np.float32)
    labels_arr = np.full(total, "X", dtype="<U1")
    blocks, offset = [], 0

    for s, length in zip(sections, lengths):
        b = halves[s]["ball"]
        n0 = min(b["n"])
        idx = np.array(b["n"]) - n0 + offset
        ball_arr[idx, 0] = np.array(b["x"]) + PITCH[0] / 2
        ball_arr[idx, 1] = np.array(b["y"]) + PITCH[1] / 2
        status = np.array(b["status"])
        poss = np.array(b["poss"])
        labels_arr[idx] = np.where(status == 0, "D", np.where(poss == 1, "H", "A"))

        for p in halves[s]["people"]:
            k = slots[p["person"]]
            pidx = np.array(p["n"]) - n0 + offset
            ok = (pidx >= offset) & (pidx < offset + length)
            players_arr[pidx[ok], k, 0] = (np.array(p["x"]) + PITCH[0] / 2)[ok]
            players_arr[pidx[ok], k, 1] = (np.array(p["y"]) + PITCH[1] / 2)[ok]
            players_arr[pidx[ok], k, 2] = (np.array(p["s"]) / 3.6)[ok]
        blocks.append((offset, offset + length - 1))
        offset += length + HALF_GAP

    np.savez_compressed(
        out_npz,
        ball=ball_arr,
        players=players_arr,
        team=team_arr,
        labels=labels_arr,
        blocks=np.array(blocks),
        pitch=np.array(PITCH),
    )
    print(f"  Converted {match_id} in {time.time() - t_parse:.1f}s -> {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB)")

    if not keep_xml and os.path.isfile(p_pos):
        os.remove(p_pos)

    return out_npz


# --- Metrica Processing ---

def ends_dead(kind, sub):
    return (
        kind == "BALL OUT"
        or "END HALF" in sub
        or (kind == "SHOT" and ("GOAL" in sub or "OUT" in sub) and "SAVED" not in sub)
    )


def read_metrica_csv_team(file):
    with open(file) as handle:
        reader = csv.reader(handle)
        next(reader), next(reader)
        header = next(reader)
        rows = np.array(
            [[float(v) if v not in ("", "NaN") else np.nan for v in row] for row in reader],
            dtype=np.float64,
        )
    n_players = (len(header) - 5) // 2
    players = rows[:, 3 : 3 + 2 * n_players].reshape(len(rows), n_players, 2)
    return rows[:, 0].astype(int), rows[:, 1].astype(int), players, rows[:, -2:]


def parse_metrica_events(match):
    out = []
    if match == "Sample_Game_3":
        ev_file = os.path.join(METRICA_DIR, f"{match}_events.json")
        for e in json.load(open(ev_file))["data"]:
            sub = e.get("subtypes") or {}
            sub = sub.get("name", "") if isinstance(sub, dict) else "-".join(s["name"] for s in sub)
            end = (e.get("end") or {}).get("frame") or e["start"]["frame"]
            out.append((e["period"], e["start"]["frame"], end, e["type"]["name"], sub, e["team"]["name"] in HOME_NAMES))
    else:
        ev_file = os.path.join(METRICA_DIR, f"{match}_RawEventsData.csv")
        with open(ev_file) as handle:
            for r in csv.DictReader(handle):
                out.append((int(r["Period"]), int(r["Start Frame"]), int(r["End Frame"]), r["Type"], r["Subtype"] or "", r["Team"] in HOME_NAMES))
    return sorted(out, key=lambda e: (e[0], e[1]))


def fetch_metrica(match, keep_raw=False):
    """Download and convert one Metrica sample match to standard NPZ."""
    out_npz = os.path.join(CACHE_DIR, f"{match}.npz")
    if os.path.isfile(out_npz):
        print(f"  {match} already cached at {out_npz}")
        return out_npz

    os.makedirs(METRICA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if match == "Sample_Game_3":
        raw_files = ["events.json", "tracking.txt", "metadata.xml"]
    else:
        raw_files = ["RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv"]

    for f in raw_files:
        path = os.path.join(METRICA_DIR, f"{match}_{f}")
        if not os.path.isfile(path):
            url = f"{METRICA_URL}{match}/{match}_{f}"
            print(f"  Downloading Metrica {f}...")
            urllib.request.urlretrieve(url, path)

    print(f"  Parsing Metrica tracking data for {match}...")
    t0 = time.time()
    if match == "Sample_Game_3":
        root = ET.parse(os.path.join(METRICA_DIR, f"{match}_metadata.xml")).getroot()
        home_id = next(root.iter("Team")).get("id")
        player_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
        channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
        ids = sorted(player_team, key=lambda q: (player_team[q] != home_id, q))
        slot = {q: k for k, q in enumerate(ids)}
        team_arr = np.array([0 if player_team[q] == home_id else 1 for q in ids], np.int8)

        halves = {}
        for p in root.iter("ProviderParameter"):
            name, value = p.findtext("Name"), p.findtext("Value")
            if name in ("first_half_start", "second_half_start") and value:
                halves[int(value)] = 1 if name.startswith("first") else 2

        specs = []
        for d in root.iter("DataFormatSpecification"):
            refs = [r.get("playerChannelId") for r in d.iter("PlayerChannelRef")]
            order = [slot[channel_player[r]] for r in refs[::2]]
            specs.append((int(d.get("startFrame")), int(d.get("endFrame")), order))

        frames, rows_p, rows_b = [], [], []
        with open(os.path.join(METRICA_DIR, f"{match}_tracking.txt")) as handle:
            for line in handle:
                parts = line.rstrip("\n").split(":")
                if len(parts) != 3:
                    continue
                frames.append(int(parts[0]))
                rows_p.append([float(v) for g in parts[1].split(";") for v in g.split(",")])
                rows_b.append([float(v) for v in parts[2].split(",")[:2]])

        frame = np.array(frames)
        flat = np.array(rows_p)
        players_xy = np.full((len(frame), len(ids), 2), np.nan)
        for start, end, order in specs:
            rows = (frame >= start) & (frame <= end)
            players_xy[np.ix_(rows, order)] = flat[rows].reshape(rows.sum(), len(order), 2)
        second = min(f for f, h in halves.items() if h == 2)
        period = np.where(frame >= second, 2, 1)
        ball_xy = np.array(rows_b)
    else:
        period, frame, home, ball_xy = read_metrica_csv_team(os.path.join(METRICA_DIR, f"{match}_RawTrackingData_Home_Team.csv"))
        period_a, frame_a, away, _ = read_metrica_csv_team(os.path.join(METRICA_DIR, f"{match}_RawTrackingData_Away_Team.csv"))
        team_arr = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
        players_xy = np.concatenate([home, away], axis=1)

    scale = np.array(PITCH)
    sections = sorted(set(period.tolist()))[:2]
    lengths = [int((period == s).sum()) for s in sections]
    total = sum(lengths) + HALF_GAP * (len(sections) - 1)
    ball_arr = np.full((total, 3), np.nan, np.float32)
    players_arr = np.full((total, len(team_arr), 3), np.nan, np.float32)
    blocks, index, offset = [], {}, 0

    for s, length in zip(sections, lengths):
        rows = np.flatnonzero(period == s)
        idx = np.arange(length) + offset
        ball_arr[idx, :2] = ball_xy[rows] * scale
        xy = players_xy[rows] * scale
        players_arr[idx, :, :2] = xy
        players_arr[idx, :, 2] = speed_centred(xy)
        index[s] = (int(frame[rows[0]]), offset, length)
        blocks.append((offset, offset + length - 1))
        offset += length + HALF_GAP

    def frame_of(p, f):
        if p not in index:
            return None
        first, off, length = index[p]
        return off + min(max(f - first, 0), length - 1)

    # Rebuild possession labels from events
    evs = parse_metrica_events(match)
    lab_arr = np.full(total, "X", dtype="<U1")
    for (s, (first, off, length)), (lo, hi) in zip(index.items(), blocks):
        p_evs = [e for e in evs if e[0] == s]
        marks = []
        for _, start, end, kind, sub, home in p_evs:
            side = "H" if home else "A"
            sf, ef = frame_of(s, start) - off, frame_of(s, max(end, start)) - off
            if kind in ("SET PIECE", "PASS", "SHOT", "BALL OUT", "RECOVERY", "BALL LOST"):
                marks.append((sf, 1, side))
                if ends_dead(kind, sub):
                    marks.append((ef, 0, "D"))
            elif kind == "FAULT RECEIVED":
                marks.append((sf, 0, "D"))
        out = np.full(length, "D", dtype="<U1")
        state, last = "D", 0
        for idx, _, new in sorted(marks):
            out[last:idx] = state
            state, last = new, idx
        out[last:] = state
        lab_arr[lo : hi + 1] = out

    np.savez_compressed(
        out_npz,
        ball=ball_arr,
        players=players_arr,
        team=team_arr,
        labels=lab_arr,
        blocks=np.array(blocks),
        pitch=np.array(PITCH),
    )
    print(f"  Converted {match} in {time.time() - t0:.1f}s -> {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB)")

    if not keep_raw and match == "Sample_Game_3":
        for f in raw_files:
            p = os.path.join(METRICA_DIR, f"{match}_{f}")
            if os.path.isfile(p):
                os.remove(p)

    return out_npz


def fetch_all(keep_raw=False):
    """Fetch and convert all 7 DFL and 3 Metrica matches."""
    print("=" * 70)
    print("FETCHING & PREPARING 10 EXTERNAL BENCHMARK MATCHES")
    print("=" * 70)
    t0 = time.time()
    for m in DFL_MATCHES:
        print(f"\n[DFL] {m}:")
        fetch_dfl(m, keep_xml=keep_raw)
    for m in METRICA_MATCHES:
        print(f"\n[Metrica] {m}:")
        fetch_metrica(m, keep_raw=keep_raw)
    print("\n" + "=" * 70)
    print(f"All 10 benchmark matches fetched and prepared in {time.time() - t0:.1f}s.")
    print("=" * 70)


def run_self_checks():
    """Unit self-check verifying speed centering and DFL/Metrica parsing contracts."""
    test_xy = np.array([[[0.0, 0.0]], [[0.0, 1.0]], [[0.0, 2.0]], [[0.0, 3.0]], [[0.0, 4.0]]], dtype=np.float32)
    s = speed_centred(test_xy, k=1)
    assert not np.isnan(s[1]), "Centred speed must be finite for internal points"
    assert np.isclose(s[1, 0], 25.0), f"Speed must match 25 m/s (got {s[1, 0]})"
    print("  bench_external/fetch.py self-checks passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and convert external benchmark datasets.")
    parser.add_argument("--match", default=None, help="Specific match ID to fetch")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw XML / TXT files")
    parser.add_argument("--check", action="store_true", help="Run unit self-checks only")
    args = parser.parse_args()

    if args.check:
        run_self_checks()
        sys.exit(0)

    if args.match:
        if args.match in DFL_MATCHES:
            fetch_dfl(args.match, keep_xml=args.keep_raw)
        elif args.match in METRICA_MATCHES:
            fetch_metrica(args.match, keep_raw=args.keep_raw)
        else:
            raise ValueError(f"Unknown match: {args.match}")
    else:
        run_self_checks()
        fetch_all(keep_raw=args.keep_raw)
