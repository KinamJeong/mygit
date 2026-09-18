"""AC(motec/pkl)와 Putnam 실차 데이터를 DDM 입력 규약으로 정렬하는 전처리 모듈.

규약 근거: AC_Putnam_regulation_mapping.md (게이트 1 확정본).
DDM 슬립각 공식(models.py):
    alphaf = steering - atan2(lf*yaw_rate + vy, |vx|) + Shf
    alphar = atan2(lr*yaw_rate - vy, |vx|) + Shr
feature 순서(models.py DeepDynamicsDataset / DeepPacejkaDataset에서 역추적):
    [VX, VY, YAW_RATE, THROTTLE_FB, STEERING_FB, THROTTLE_CMD, STEERING_CMD]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DT = 0.04  # 25 Hz, AC(다운샘플 후)와 Putnam 공통

CANONICAL_COLUMNS = ["t", "vx", "vy", "yaw_rate", "steering", "throttle", "brake"]
DDM_FEATURES = ["VX", "VY", "YAW_RATE", "THROTTLE_FB", "STEERING_FB", "THROTTLE_CMD", "STEERING_CMD"]
DDM_STATE = ["VX", "VY", "YAW_RATE"]


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
    "t": "currentTime",  # ms (iCurrentTime); 없으면 DT로 생성
    "vx": "local_velocity_x",
    "vy": "local_velocity_y",
    "yaw_rate": "angular_velocity_y",
    "steer_wheel_deg": "steerAngle",
    "throttle": "accStatus",
    "brake": "brakeStatus",
    "slip_front": ("SlipAngle_fl", "SlipAngle_fr"),
    "slip_rear": ("SlipAngle_rl", "SlipAngle_rr"),
}

# Putnam CSV 헤더명
PUTNAM_COLS = {
    "t": "time(s)",
    "vx": "vx(m/s)",
    "vy": "vy(m/s)",
    "yaw_rate": "omega(rad/s)",
    "steering": "delta(rad)",
    "steer_rate": "deltadelta(rad/s)",
    "throttle": "throttle_ped_cmd(%)",
    "brake": "brake_ped_cmd(kPa)",
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
        # AC currentTime은 랩 타임이라 랩 경계에서 리셋됨 → 그 경우 인덱스 기반으로
        if np.all(np.diff(t) > 0):
            return t - t[0]
    return np.arange(len(frame), dtype=float) * dt


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
    """
    if spec.steer_ratio is None:
        raise ValueError(f"{spec.name}: steer_ratio 미확정. estimate_steer_ratio()로 먼저 역산하세요.")
    src = as_frame(data)
    c = AC_COLS

    yaw_rate = src[c["yaw_rate"]].to_numpy(dtype=float) * yaw_rate_sign
    if yaw_rate_in_deg:
        yaw_rate = np.deg2rad(yaw_rate)
    steer_wheel_deg = src[c["steer_wheel_deg"]].to_numpy(dtype=float)

    out = pd.DataFrame({
        "t": _time_axis(src, c["t"], 1e-3, dt),
        "vx": src[c["vx"]].to_numpy(dtype=float),
        "vy": src[c["vy"]].to_numpy(dtype=float) * vy_sign,
        "yaw_rate": yaw_rate,
        "steering": np.deg2rad(steer_wheel_deg / spec.steer_ratio),
        "throttle": src[c["throttle"]].to_numpy(dtype=float) * pedal_scale,
        "brake": src[c["brake"]].to_numpy(dtype=float) * pedal_scale,
        "steer_wheel_deg": steer_wheel_deg,
    })

    # 실측 슬립각(deg, DDM과 같은 부호) → rad, 좌우 평균. 게이트 1 판정선용.
    if all(k in src.columns for k in (*c["slip_front"], *c["slip_rear"])):
        out["alphaf_meas"] = np.deg2rad(src[list(c["slip_front"])].mean(axis=1).to_numpy())
        out["alphar_meas"] = np.deg2rad(src[list(c["slip_rear"])].mean(axis=1).to_numpy())

    out.attrs.update(source="ac", vehicle=spec.name, dt=dt, brake_unit="normalized")
    return out


# ---------------------------------------------------------------------------
# Putnam → canonical
# ---------------------------------------------------------------------------
def read_putnam_csv(path: str | Path) -> pd.DataFrame:
    """'# ' 로 시작하는 헤더 + 탭/공백 구분 본문."""
    path = Path(path)
    with path.open() as f:
        header = f.readline().lstrip("#").split()
    return pd.read_csv(path, sep=r"\s+", skiprows=1, names=header, engine="python")


def putnam_to_canonical(
    data,
    *,
    brake_max_kpa: float | None = None,
    steering_cmd_from_rate: bool = False,
    dt: float = DT,
) -> pd.DataFrame:
    """Putnam CSV(경로 또는 DataFrame)를 DDM 규약으로.

    delta(rad)는 이미 바퀴각이라 그대로. throttle %→0~1.
    brake는 kPa→0~1 매핑 규칙이 미확정이므로 기본은 원값(kPa) 유지,
    brake_max_kpa를 주면 그 값으로 나눈 0~1을 반환한다.
    """
    src = read_putnam_csv(data) if isinstance(data, (str, Path)) else as_frame(data)
    c = PUTNAM_COLS

    brake = src[c["brake"]].to_numpy(dtype=float)
    brake_unit = "kPa"
    if brake_max_kpa is not None:
        brake = brake / brake_max_kpa
        brake_unit = f"normalized(/{brake_max_kpa} kPa)"

    out = pd.DataFrame({
        "t": _time_axis(src, c["t"], 1.0, dt),
        "vx": src[c["vx"]].to_numpy(dtype=float),
        "vy": src[c["vy"]].to_numpy(dtype=float),
        "yaw_rate": src[c["yaw_rate"]].to_numpy(dtype=float),
        "steering": src[c["steering"]].to_numpy(dtype=float),
        "throttle": src[c["throttle"]].to_numpy(dtype=float) / 100.0,
        "brake": brake,
    })
    if c["steer_rate"] in src.columns:
        out["steer_rate"] = src[c["steer_rate"]].to_numpy(dtype=float)
        if steering_cmd_from_rate:
            out["steering_cmd"] = out["steer_rate"] * dt

    out.attrs.update(source="putnam", vehicle=PUTNAM_AV21.name, dt=dt, brake_unit=brake_unit)
    return out


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
    반환 slope가 부호 포함 조향비(Miata 기준 ≈ −15.07).
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
    thr = frame["throttle"]
    if thr.min() < -1e-6 or thr.max() > 1.0 + 1e-6:
        issues.append(f"throttle 범위 [{thr.min():.2f}, {thr.max():.2f}] — 0~1 아님")
    if frame.attrs.get("brake_unit", "").startswith("normalized"):
        br = frame["brake"]
        if br.min() < -1e-6 or br.max() > 1.0 + 1e-6:
            issues.append(f"brake 범위 [{br.min():.2f}, {br.max():.2f}] — 0~1 아님")
    if (frame["vx"] < -0.5).mean() > 0.01:
        issues.append("vx 음수 구간이 1% 초과 — 종축 부호/축 재배열 확인")
    return issues


# ---------------------------------------------------------------------------
# canonical → DDM 학습 윈도우
# ---------------------------------------------------------------------------
def to_ddm_windows(
    frame: pd.DataFrame,
    horizon: int,
    *,
    throttle_signal: str = "throttle",
) -> tuple[np.ndarray, np.ndarray]:
    """canonical frame → (X[N, horizon, 7], y[N, 3]).

    *_CMD는 다음 스텝과의 차분(fb[i+1] − fb[i]). models.py가 steering = FB + CMD로
    쓰므로 이렇게 두면 예측 시점의 조향/스로틀이 다음 스텝 값이 된다.
    Putnam에서 steering_cmd 컬럼(deltadelta·dt)이 있으면 그것을 우선 사용.
    throttle_signal: "throttle" | "throttle_minus_brake" (brake가 0~1일 때만 의미 있음).
    """
    if throttle_signal == "throttle":
        thr = frame["throttle"].to_numpy(dtype=float)
    elif throttle_signal == "throttle_minus_brake":
        if not frame.attrs.get("brake_unit", "").startswith("normalized"):
            raise ValueError("brake가 0~1 정규화되어 있지 않습니다 (brake_max_kpa 지정 필요).")
        thr = frame["throttle"].to_numpy(dtype=float) - frame["brake"].to_numpy(dtype=float)
    else:
        raise ValueError(f"알 수 없는 throttle_signal: {throttle_signal}")

    steer = frame["steering"].to_numpy(dtype=float)
    thr_cmd = np.diff(thr, append=np.nan)
    steer_cmd = (
        frame["steering_cmd"].to_numpy(dtype=float)
        if "steering_cmd" in frame.columns
        else np.diff(steer, append=np.nan)
    )

    feats = np.column_stack([
        frame["vx"].to_numpy(dtype=float),
        frame["vy"].to_numpy(dtype=float),
        frame["yaw_rate"].to_numpy(dtype=float),
        thr, steer, thr_cmd, steer_cmd,
    ])
    n = len(feats) - horizon
    if n <= 0:
        raise ValueError(f"샘플 {len(feats)}개로 horizon {horizon} 윈도우를 만들 수 없습니다.")
    idx = np.arange(horizon)[None, :] + np.arange(n)[:, None]
    X = feats[idx]
    y = feats[horizon:, :3]
    ok = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y).any(axis=1)
    return X[ok], y[ok]


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
        "SlipAngle_fl": np.rad2deg(af) + noise(),
        "SlipAngle_fr": np.rad2deg(af) + noise(),
        "SlipAngle_rl": np.rad2deg(ar) + noise(),
        "SlipAngle_rr": np.rad2deg(ar) + noise(),
    })


def _self_test() -> None:
    ac_raw = _synthetic_ac(MIATA)

    ac = ac_to_canonical(ac_raw, MIATA)
    assert list(ac.columns[:7]) == CANONICAL_COLUMNS
    assert not check_ranges(ac), check_ranges(ac)

    rep = validate_slip_angles(ac, MIATA)
    for side in ("front", "rear"):
        assert abs(rep[side]["slope"] - 1) < 0.02 and rep[side]["r2"] > 0.99, rep

    # 부호가 뒤집힌 yaw_rate/vy는 판정선에서 걸려야 함
    bad = validate_slip_angles(ac_to_canonical(ac_raw, MIATA, vy_sign=-1, yaw_rate_sign=-1), MIATA)
    assert bad["rear"]["slope"] < 0, bad

    est = estimate_steer_ratio(ac_to_canonical(ac_raw, VehicleSpec("tmp", MIATA.wheelbase, 1.0)), MIATA.wheelbase)
    assert abs(est["steer_ratio"] - MIATA.steer_ratio) < 0.3 and est["r2"] > 0.95, est

    X, y = to_ddm_windows(ac, horizon=5)
    assert X.shape[1:] == (5, 7) and y.shape[1] == 3 and len(X) == len(y)
    # FB + CMD == 다음 스텝 값 (models.py의 steering = FB + CMD 전제)
    np.testing.assert_allclose(X[0, -1, 4] + X[0, -1, 6], ac["steering"].iloc[5])
    np.testing.assert_allclose(y[0], ac[["vx", "vy", "yaw_rate"]].iloc[5].to_numpy())

    sample = (
        "# time(s)\tx(m)\ty(m)\tvx(m/s)\tvy(m/s)\tphi(rad)\tdelta(rad)\tomega(rad/s)\tax(m/s^2)\t"
        "deltadelta(rad/s)\twheel_fl(kmph)\twheel_fr(kmph)\twheel_rl(kmph)\twheel_rr(kmph)\troll(rad)\t"
        "throttle_ped_cmd(%)\tbrake_ped_cmd(kPa)\n"
        "1692110152.45344\t71.49\t-129.36\t9.594\t0.0948\t-3.0656\t-0.00407\t-0.01907\t0.905\t0\t35.2\t35.0\t34.8\t34.7\t-0.707\t10.503\t0\n"
        "1692110152.49344\t71.10\t-129.39\t9.606\t0.0639\t-3.0663\t-0.00407\t-0.01899\t0.890\t0\t35.0\t35.2\t34.8\t34.8\t-0.747\t10.430\t120\n"
        "1692110152.53344\t70.75\t-129.41\t9.580\t0.0699\t-3.0671\t-0.00465\t-0.01890\t0.879\t-0.0145\t35.4\t35.2\t35.0\t34.9\t-0.788\t10.413\t0\n"
    )
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(sample)
    pt = putnam_to_canonical(f.name)
    assert list(pt.columns[:7]) == CANONICAL_COLUMNS
    assert abs(pt["throttle"].iloc[0] - 0.10503) < 1e-6 and pt["brake"].iloc[1] == 120
    assert abs(pt["t"].iloc[1] - DT) < 1e-6
    pt_norm = putnam_to_canonical(f.name, brake_max_kpa=1800, steering_cmd_from_rate=True)
    assert abs(pt_norm["brake"].iloc[1] - 120 / 1800) < 1e-9 and "steering_cmd" in pt_norm.columns
    assert not check_ranges(pt_norm), check_ranges(pt_norm)

    print("self-test OK")
    print("slip validation (synthetic):", {k: round(v["r2"], 4) for k, v in rep.items()})
    print("steer ratio estimate (synthetic):", round(est["steer_ratio"], 3), "r2", round(est["r2"], 4))


if __name__ == "__main__":
    _self_test()
