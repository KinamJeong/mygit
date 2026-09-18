"""AC(motec/pkl)와 Putnam 실차 데이터를 DDM 입력 규약으로 정렬하는 전처리 모듈.

규약 근거: AC_Putnam_regulation_mapping.md (게이트 1 확정본) + deep_dynamics csv_parser.py.
DDM 슬립각 공식(models.py):
    alphaf = steering - atan2(lf*yaw_rate + vy, |vx|) + Shf
    alphar = atan2(lr*yaw_rate - vy, |vx|) + Shr
feature 순서(csv_parser.py):
    [VX, VY, YAW_RATE, PEDAL_FB, STEERING_FB, PEDAL_CMD, STEERING_CMD, FUTURE_VX]
    DDM은 앞 7열, DeepPacejka는 3·5열을 제외한 6열을 사용.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DT = 0.04  # 25 Hz, AC(다운샘플 후)와 Putnam 공통
MAX_BRAKE_PRESSURE_KPA = 2757.89990234  # csv_parser.py, = 400 psi
MIN_MOVING_VX = 5.0  # csv_parser.py 속도 게이트
FUTURE_VX_STEPS = 5  # csv_parser.py 8번째 feature

CANONICAL_COLUMNS = ["t", "vx", "vy", "yaw_rate", "steering", "throttle", "brake", "pedal"]
DDM_FEATURES = ["VX", "VY", "YAW_RATE", "PEDAL_FB", "STEERING_FB", "PEDAL_CMD", "STEERING_CMD", "FUTURE_VX"]
DDM_STATE = ["VX", "VY", "YAW_RATE"]
POSE_COLUMNS = ["x", "y", "phi", "vx", "vy", "yaw_rate", "pedal", "steering"]


@dataclass(frozen=True)
class VehicleSpec:
    name: str
    wheelbase: float
    # 부호 포함 조향비 = steerAngle(휠, deg) / 바퀴각(deg). None이면 입력이 이미 바퀴각.
    steer_ratio: float | None
    lf: float | None = None
    lr: float | None = None

    @property
    def lf_(self) -> float:
        return self.wheelbase / 2 if self.lf is None else self.lf

    @property
    def lr_(self) -> float:
        return self.wheelbase / 2 if self.lr is None else self.lr


MIATA = VehicleSpec("ks_mazda_miata", wheelbase=2.27, steer_ratio=-15.07)
# 조향비 미확정 차종: estimate_steer_ratio()로 재역산 후 채울 것
BMW_Z4_GT3 = VehicleSpec("bmw_z4_gt3", wheelbase=2.51, steer_ratio=None)
DALLARA_F317 = VehicleSpec("dallara_f317", wheelbase=2.80, steer_ratio=None)
PUTNAM_AV21 = VehicleSpec("putnam_av21", wheelbase=2.97, steer_ratio=None)

# structures.py 채널명
AC_COLS = {
    "t": "currentTime",  # ms (iCurrentTime); 랩마다 리셋되므로 비단조면 DT로 생성
    "vx": "local_velocity_x",
    "vy": "local_velocity_y",
    "yaw_rate": "angular_velocity_y",
    "steer_wheel_deg": "steerAngle",
    "throttle": "accStatus",
    "brake": "brakeStatus",
    "x": "world_position_x",
    "y": "world_position_y",
    "phi": "yaw",
    "slip_front": ("SlipAngle_fl", "SlipAngle_fr"),
    "slip_rear": ("SlipAngle_rl", "SlipAngle_rr"),
}

# Putnam CSV 헤더명 (단위 괄호 제외; csv_parser.py와 같은 방식으로 매칭)
PUTNAM_COLS = {
    "t": "time",
    "vx": "vx",
    "vy": "vy",
    "yaw_rate": "omega",
    "steering": "delta",
    "steer_rate": "deltadelta",
    "throttle": "throttle_ped_cmd",
    "brake": "brake_ped_cmd",
    "x": "x",
    "y": "y",
    "phi": "phi",
}


# ---------------------------------------------------------------------------
# 입력 정규화
# ---------------------------------------------------------------------------
def as_frame(data) -> pd.DataFrame:
    """DataFrame / list[dict] / dict[str, seq] 어느 형태든 DataFrame으로."""
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True)
    if isinstance(data, Mapping):
        return pd.DataFrame(dict(data))
    if isinstance(data, Sequence) and data and isinstance(data[0], Mapping):
        return pd.DataFrame(list(data))
    raise TypeError(f"지원하지 않는 입력 형태: {type(data)}")


def _time_axis(frame: pd.DataFrame, col: str | None, scale: float, dt: float) -> np.ndarray:
    if col is not None and col in frame.columns:
        t = frame[col].to_numpy(dtype=float) * scale
        if np.all(np.diff(t) > 0):
            return t - t[0]
    return np.arange(len(frame), dtype=float) * dt


def combine_pedals(throttle01: np.ndarray, brake01: np.ndarray) -> np.ndarray:
    """csv_parser.py 규칙: brake>0이면 −brake, 아니면 throttle. 부호 있는 단일 채널(−1~1)."""
    return np.where(brake01 > 0.0, -brake01, throttle01)


# ---------------------------------------------------------------------------
# AC → canonical
# ---------------------------------------------------------------------------
def ac_to_canonical(
    data,
    spec: VehicleSpec = MIATA,
    *,
    vy_sign: float = 1.0,
    yaw_rate_sign: float = 1.0,
    yaw_rate_in_deg: bool = False,
    pedal_scale: float = 1.0,
    dt: float = DT,
) -> pd.DataFrame:
    """loader를 거친 AC pkl/motec 값을 DDM 규약(body frame, m/s, rad, rad/s, 0~1)으로.

    steerAngle만 loader가 건드리지 않으므로 여기서 조향비 나눗셈 + deg→rad를 수행한다.
    vy_sign / yaw_rate_sign은 pkl·motec 부호 규약이 어긋날 때 validate_slip_angles로
    판별한 뒤 -1을 주기 위한 자리. pedal_scale은 원값이 %일 때 0.01.

    주의: AC brakeStatus는 페달 위치(0~1), Putnam brake는 압력/최대압력(0~1)이라
    pedal 채널의 음수 구간은 두 데이터 간 물리적 등가가 아니다.
    """
    if spec.steer_ratio is None:
        raise ValueError(f"{spec.name}: steer_ratio 미확정. estimate_steer_ratio()로 먼저 역산하세요.")
    src = as_frame(data)
    c = AC_COLS

    yaw_rate = src[c["yaw_rate"]].to_numpy(dtype=float) * yaw_rate_sign
    if yaw_rate_in_deg:
        yaw_rate = np.deg2rad(yaw_rate)
    steer_wheel_deg = src[c["steer_wheel_deg"]].to_numpy(dtype=float)
    throttle = src[c["throttle"]].to_numpy(dtype=float) * pedal_scale
    brake = src[c["brake"]].to_numpy(dtype=float) * pedal_scale

    out = pd.DataFrame({
        "t": _time_axis(src, c["t"], 1e-3, dt),
        "vx": src[c["vx"]].to_numpy(dtype=float),
        "vy": src[c["vy"]].to_numpy(dtype=float) * vy_sign,
        "yaw_rate": yaw_rate,
        "steering": np.deg2rad(steer_wheel_deg / spec.steer_ratio),
        "throttle": throttle,
        "brake": brake,
        "pedal": combine_pedals(throttle, brake),
        "steer_wheel_deg": steer_wheel_deg,
    })
    for k in ("x", "y", "phi"):
        if c[k] in src.columns:
            out[k] = src[c[k]].to_numpy(dtype=float)

    # 실측 슬립각(deg, DDM과 같은 부호) → rad, 좌우 평균. 게이트 1 판정선용.
    if all(k in src.columns for k in (*c["slip_front"], *c["slip_rear"])):
        out["alphaf_meas"] = np.deg2rad(src[list(c["slip_front"])].mean(axis=1).to_numpy())
        out["alphar_meas"] = np.deg2rad(src[list(c["slip_rear"])].mean(axis=1).to_numpy())

    out.attrs.update(source="ac", vehicle=spec.name, dt=dt, brake_unit="pedal_position")
    return out


# ---------------------------------------------------------------------------
# Putnam → canonical
# ---------------------------------------------------------------------------
def read_putnam_csv(path: str | Path) -> pd.DataFrame:
    """헤더 '# time(s) ...' 형태, 구분자는 ',' 또는 탭/공백 자동 인식. 컬럼명은 단위 괄호 제거."""
    path = Path(path)
    with path.open() as f:
        header_line = f.readline().lstrip("#").strip()
    sep = "," if "," in header_line else r"\s+"
    names = [h.strip().split("(")[0] for h in (header_line.split(",") if sep == "," else header_line.split())]
    return pd.read_csv(path, sep=sep, skiprows=1, names=names, engine="python")


def putnam_to_canonical(
    data,
    *,
    brake_max_kpa: float = MAX_BRAKE_PRESSURE_KPA,
    dt: float = DT,
) -> pd.DataFrame:
    """Putnam CSV(경로 또는 DataFrame)를 DDM 규약으로.

    delta(rad)는 이미 바퀴각이라 그대로. throttle %→0~1, brake kPa→/brake_max_kpa.
    pedal = combine_pedals(throttle, brake) 가 csv_parser.py의 'throttle' 채널에 해당.
    """
    src = read_putnam_csv(data) if isinstance(data, (str, Path)) else as_frame(data)
    c = PUTNAM_COLS
    throttle = src[c["throttle"]].to_numpy(dtype=float) / 100.0
    brake = src[c["brake"]].to_numpy(dtype=float) / brake_max_kpa

    out = pd.DataFrame({
        "t": _time_axis(src, c["t"], 1.0, dt),
        "vx": src[c["vx"]].to_numpy(dtype=float),
        "vy": src[c["vy"]].to_numpy(dtype=float),
        "yaw_rate": src[c["yaw_rate"]].to_numpy(dtype=float),
        "steering": src[c["steering"]].to_numpy(dtype=float),
        "throttle": throttle,
        "brake": brake,
        "pedal": combine_pedals(throttle, brake),
    })
    for k in ("x", "y", "phi", "steer_rate"):
        if c[k] in src.columns:
            out[k] = src[c[k]].to_numpy(dtype=float)

    out.attrs.update(source="putnam", vehicle=PUTNAM_AV21.name, dt=dt,
                     brake_unit=f"pressure/{brake_max_kpa:.1f}kPa")
    return out


def first_moving_segment(frame: pd.DataFrame, min_vx: float = MIN_MOVING_VX) -> pd.DataFrame:
    """csv_parser.py 게이트: 앞쪽 |vx|<min_vx 구간을 버리고, 주행 후 첫 |vx|<min_vx 직전까지."""
    moving = np.abs(frame["vx"].to_numpy(dtype=float)) >= min_vx
    if not moving.any():
        raise ValueError(f"|vx| >= {min_vx} 인 구간이 없습니다.")
    start = int(np.argmax(moving))
    stops = np.flatnonzero(~moving[start:])
    end = start + int(stops[0]) if len(stops) else len(frame)
    seg = frame.iloc[start:end].reset_index(drop=True)
    seg.attrs.update(frame.attrs)
    return seg


# ---------------------------------------------------------------------------
# 슬립각 · 검증 · 조향비 역산
# ---------------------------------------------------------------------------
def compute_slip_angles(
    frame: pd.DataFrame, spec: VehicleSpec, *, Shf: float = 0.0, Shr: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """models.py 공식 그대로. 반환 (alphaf, alphar) [rad]."""
    vx = np.abs(frame["vx"].to_numpy(dtype=float))
    vy = frame["vy"].to_numpy(dtype=float)
    w = frame["yaw_rate"].to_numpy(dtype=float)
    d = frame["steering"].to_numpy(dtype=float)
    alphaf = d - np.arctan2(spec.lf_ * w + vy, vx) + Shf
    alphar = np.arctan2(spec.lr_ * w - vy, vx) + Shr
    return alphaf, alphar


def _linfit(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(x)),
    }


def validate_slip_angles(
    frame: pd.DataFrame, spec: VehicleSpec, *, min_vx: float = 2.0
) -> dict[str, dict]:
    """게이트 1 판정선: DDM 공식 alphaf/alphar vs AC 실측 SlipAngle.

    부호·단위·조향비가 맞으면 slope≈1, intercept≈0, r2≈1이어야 한다.
    slope≈-1이면 부호 반전, |slope|≠1이면 단위/조향비 오류로 읽는다.
    """
    if "alphaf_meas" not in frame.columns:
        raise ValueError("alphaf_meas/alphar_meas 없음: SlipAngle_* 채널이 있는 AC 데이터가 필요합니다.")
    mask = np.abs(frame["vx"].to_numpy(dtype=float)) >= min_vx
    af, ar = compute_slip_angles(frame, spec)
    report = {}
    for name, calc, meas in (
        ("front", af, frame["alphaf_meas"].to_numpy()),
        ("rear", ar, frame["alphar_meas"].to_numpy()),
    ):
        fit = _linfit(meas[mask], calc[mask])
        fit["rmse_deg"] = float(np.rad2deg(np.sqrt(np.mean((calc[mask] - meas[mask]) ** 2))))
        fit["sign_agreement"] = float(np.mean(np.sign(calc[mask]) == np.sign(meas[mask])))
        report[name] = fit
    return report


def estimate_steer_ratio(
    frame: pd.DataFrame, wheelbase: float, *, min_vx: float = 5.0
) -> dict:
    """바이시클 항등식 δ_wheel = alphaf − alphar + L·ω/vx 로 조향비(부호 포함) 회귀.

    입력은 ac_to_canonical 출력(steer_wheel_deg, alphaf_meas, alphar_meas 필요).
    조향비 미확정 차종은 임시 spec(steer_ratio=1.0)으로 ac_to_canonical을 호출한 뒤 사용.
    반환 steer_ratio가 부호 포함 조향비(Miata 기준 ≈ −15.07).
    소각 근사식이므로 약 0.5% 내외의 편향이 있을 수 있다.
    """
    need = ("steer_wheel_deg", "alphaf_meas", "alphar_meas")
    if not all(k in frame.columns for k in need):
        raise ValueError(f"{need} 컬럼이 필요합니다.")
    vx = frame["vx"].to_numpy(dtype=float)
    mask = np.abs(vx) >= min_vx
    w = frame["yaw_rate"].to_numpy(dtype=float)
    delta_wheel_deg = np.rad2deg(
        frame["alphaf_meas"].to_numpy() - frame["alphar_meas"].to_numpy() + wheelbase * w / vx
    )
    fit = _linfit(delta_wheel_deg[mask], frame["steer_wheel_deg"].to_numpy(dtype=float)[mask])
    fit["steer_ratio"] = fit.pop("slope")
    fit["max_road_wheel_deg"] = float(np.nanmax(np.abs(frame["steer_wheel_deg"])) / abs(fit["steer_ratio"]))
    return fit


# ---------------------------------------------------------------------------
# 규약 위반 조기 탐지
# ---------------------------------------------------------------------------
def check_ranges(frame: pd.DataFrame) -> list[str]:
    """단위·부호 실수를 값 범위로 잡아낸다. 문제 없으면 빈 리스트."""
    issues = []
    st = np.nanmax(np.abs(frame["steering"]))
    if st > math.radians(45):
        issues.append(f"steering 최대 {math.degrees(st):.1f}° — 바퀴각이 아니라 휠각/deg일 가능성")
    yr = np.nanmax(np.abs(frame["yaw_rate"]))
    if yr > 3.0:
        issues.append(f"yaw_rate 최대 {yr:.2f} — deg/s 미변환 가능성")
    for col in ("throttle", "brake"):
        v = frame[col]
        if v.min() < -1e-6 or v.max() > 1.0 + 1e-6:
            issues.append(f"{col} 범위 [{v.min():.2f}, {v.max():.2f}] — 0~1 아님")
    if (frame["vx"] < -0.5).mean() > 0.01:
        issues.append("vx 음수 구간이 1% 초과 — 종축 부호/축 재배열 확인")
    return issues


# ---------------------------------------------------------------------------
# canonical → DDM 학습 윈도우 (csv_parser.py와 동일 규약)
# ---------------------------------------------------------------------------
def to_ddm_windows(
    frame: pd.DataFrame,
    horizon: int,
    *,
    future_vx_steps: int = FUTURE_VX_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """canonical frame → (X[N, horizon, 8], y[N, 3]). csv_parser.py의 features/labels와 동일.

    *_CMD[i] = fb[i+1] − fb[i] (전방 차분). models.py가 steering = FB + CMD로 쓰므로
    예측 시점의 조향/페달이 다음 스텝 값이 된다. 8열은 vx[i+future_vx_steps].
    DDM 입력은 X[:, :, :7]. 끝단의 정의 불가 윈도우는 0으로 채우지 않고 버린다.
    """
    vx = frame["vx"].to_numpy(dtype=float)
    pedal = frame["pedal"].to_numpy(dtype=float)
    steer = frame["steering"].to_numpy(dtype=float)
    n_rows = len(frame)

    future_vx = np.full(n_rows, np.nan)
    if future_vx_steps < n_rows:
        future_vx[: n_rows - future_vx_steps] = vx[future_vx_steps:]

    feats = np.column_stack([
        vx,
        frame["vy"].to_numpy(dtype=float),
        frame["yaw_rate"].to_numpy(dtype=float),
        pedal,
        steer,
        np.diff(pedal, append=np.nan),
        np.diff(steer, append=np.nan),
        future_vx,
    ])
    n = n_rows - horizon
    if n <= 0:
        raise ValueError(f"샘플 {n_rows}개로 horizon {horizon} 윈도우를 만들 수 없습니다.")
    idx = np.arange(horizon)[None, :] + np.arange(n)[:, None]
    X = feats[idx]
    y = feats[horizon:, :3]
    ok = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y).any(axis=1)
    return X[ok], y[ok]


def poses(frame: pd.DataFrame) -> np.ndarray:
    """csv_parser.py의 poses 배열: [x, y, phi, vx, vy, yaw_rate, pedal, steering]."""
    missing = [c for c in POSE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"poses에 필요한 컬럼 없음: {missing}")
    return frame[POSE_COLUMNS].to_numpy(dtype=float)


def write_npz(frame: pd.DataFrame, horizon: int, out_path: str | Path) -> Path:
    """csv_parser.py와 같은 키(features, labels, poses)로 저장."""
    X, y = to_ddm_windows(frame, horizon)
    out_path = Path(out_path)
    np.savez(out_path, features=X, labels=y, poses=poses(frame))
    return out_path


# ---------------------------------------------------------------------------
# 합성 데이터 자기일관성 테스트 (실데이터 없이 규약 코드 검증)
# ---------------------------------------------------------------------------
def _synthetic_ac(spec: VehicleSpec, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    vx = 15 + 10 * np.sin(0.05 * t) + rng.normal(0, 0.05, n)
    vy = 2.0 * np.sin(0.3 * t)
    w = 0.6 * np.sin(0.3 * t + 0.4)
    delta = 0.20 * np.sin(0.3 * t + 0.8)
    af = delta - np.arctan2(spec.lf_ * w + vy, np.abs(vx))
    ar = np.arctan2(spec.lr_ * w - vy, np.abs(vx))
    noise = lambda: rng.normal(0, 0.1, n)  # deg
    return pd.DataFrame({
        "currentTime": t * 1e3,
        "local_velocity_x": vx,
        "local_velocity_y": vy,
        "angular_velocity_y": w,
        "steerAngle": np.rad2deg(delta) * spec.steer_ratio,
        "accStatus": np.clip(0.5 + 0.4 * np.cos(0.2 * t), 0, 1),
        "brakeStatus": np.clip(-0.3 * np.cos(0.2 * t), 0, 1),
        "world_position_x": np.cumsum(vx) * DT,
        "world_position_y": np.cumsum(vy) * DT,
        "yaw": np.cumsum(w) * DT,
        "SlipAngle_fl": np.rad2deg(af) + noise(),
        "SlipAngle_fr": np.rad2deg(af) + noise(),
        "SlipAngle_rl": np.rad2deg(ar) + noise(),
        "SlipAngle_rr": np.rad2deg(ar) + noise(),
    })


def _self_test() -> None:
    ac_raw = _synthetic_ac(MIATA)

    ac = ac_to_canonical(ac_raw, MIATA)
    assert list(ac.columns[: len(CANONICAL_COLUMNS)]) == CANONICAL_COLUMNS
    assert not check_ranges(ac), check_ranges(ac)
    assert np.all(ac["pedal"][ac["brake"] > 0] < 0) and np.all(ac["pedal"][ac["brake"] == 0] >= 0)

    rep = validate_slip_angles(ac, MIATA)
    for side in ("front", "rear"):
        assert abs(rep[side]["slope"] - 1) < 0.02 and rep[side]["r2"] > 0.99, rep

    # 부호가 뒤집힌 yaw_rate/vy는 판정선에서 걸려야 함
    bad = validate_slip_angles(ac_to_canonical(ac_raw, MIATA, vy_sign=-1, yaw_rate_sign=-1), MIATA)
    assert bad["rear"]["slope"] < 0, bad

    est = estimate_steer_ratio(ac_to_canonical(ac_raw, VehicleSpec("tmp", MIATA.wheelbase, 1.0)), MIATA.wheelbase)
    assert abs(est["steer_ratio"] - MIATA.steer_ratio) < 0.3 and est["r2"] > 0.95, est

    X, y = to_ddm_windows(ac, horizon=5)
    assert X.shape[1:] == (5, 8) and y.shape[1] == 3 and len(X) == len(y)
    # FB + CMD == 다음 스텝 값 (models.py의 steering = FB + CMD 전제)
    np.testing.assert_allclose(X[0, -1, 4] + X[0, -1, 6], ac["steering"].iloc[5])
    np.testing.assert_allclose(X[0, 0, 7], ac["vx"].iloc[FUTURE_VX_STEPS])
    np.testing.assert_allclose(y[0], ac[["vx", "vy", "yaw_rate"]].iloc[5].to_numpy())
    assert poses(ac).shape == (len(ac), 8)

    # Putnam: 탭 구분과 쉼표 구분 모두, 속도 게이트 포함
    rows = [
        (1692110152.45, 71.49, -129.36, 2.0, 0.09, -3.07, -0.0041, -0.019, 0.9, 0.0, 35, 35, 35, 35, -0.7, 10.5, 0.0),
        (1692110152.49, 71.10, -129.39, 9.606, 0.0639, -3.0663, -0.00407, -0.01899, 0.89, 0.0, 35, 35, 35, 35, -0.75, 10.43, 0.0),
        (1692110152.53, 70.75, -129.41, 9.580, 0.0699, -3.0671, -0.00465, -0.01890, 0.88, -0.0145, 35, 35, 35, 35, -0.79, 10.41, 120.0),
        (1692110152.57, 70.35, -129.44, 9.619, 0.0844, -3.0680, -0.00465, -0.01939, 0.87, 0.0, 35, 35, 35, 35, -0.79, 10.41, 0.0),
        (1692110152.61, 69.97, -129.47, 3.0, 0.0972, -3.0691, -0.00465, -0.02011, 0.86, 0.0141, 35, 35, 35, 35, -0.78, 10.36, 0.0),
    ]
    header = ("time(s)", "x(m)", "y(m)", "vx(m/s)", "vy(m/s)", "phi(rad)", "delta(rad)", "omega(rad/s)",
              "ax(m/s^2)", "deltadelta(rad/s)", "wheel_fl(kmph)", "wheel_fr(kmph)", "wheel_rl(kmph)",
              "wheel_rr(kmph)", "roll(rad)", "throttle_ped_cmd(%)", "brake_ped_cmd(kPa)")
    import tempfile
    for sep in ("\t", ","):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("# " + sep.join(header) + "\n")
            for r in rows:
                f.write(sep.join(str(v) for v in r) + "\n")
        pt = putnam_to_canonical(f.name)
        assert list(pt.columns[: len(CANONICAL_COLUMNS)]) == CANONICAL_COLUMNS
        assert abs(pt["throttle"].iloc[1] - 0.1043) < 1e-9
        assert abs(pt["pedal"].iloc[2] + 120 / MAX_BRAKE_PRESSURE_KPA) < 1e-9
        assert abs(pt["t"].iloc[1] - DT) < 1e-6 and {"x", "y", "phi"} <= set(pt.columns)
        assert not check_ranges(pt), check_ranges(pt)
        seg = first_moving_segment(pt)
        assert len(seg) == 3 and seg["vx"].iloc[0] == 9.606 and seg.attrs["source"] == "putnam"

    print("self-test OK")
    print("slip validation (synthetic):", {k: round(v["r2"], 4) for k, v in rep.items()})
    print("steer ratio estimate (synthetic):", round(est["steer_ratio"], 3), "r2", round(est["r2"], 4))


if __name__ == "__main__":
    _self_test()
