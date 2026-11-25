import numpy as np
import pytest

from cantinel.checksums import crc8_j1850, sum8, xor8
from cantinel.engine import Engine, Profile
from cantinel.frame import CANFrame
from cantinel.gateway import SecureGateway
from cantinel.io import load, save_csv
from cantinel.metrics import prf
from cantinel.secoc import AuthLayer, KeyStore, SecOCReceiver, SecOCSender
from cantinel.simulator import simulate


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def profile():
    tr = simulate(150, None, seed=11)
    va = simulate(60, None, seed=12)
    return Profile.train(tr, va)


def run(profile, attack, seed=13, duration=40):
    frames = simulate(duration, attack, seed=seed)
    flags, alerts = Engine(profile).run(frames)
    y = np.array([f.label for f in frames])
    return prf(y, flags), alerts, frames


# ---------------------------------------------------------------- checksums
def test_crc8_j1850_check_value():
    assert crc8_j1850(b"123456789") == 0x4B


def test_xor_sum():
    assert xor8(b"\x01\x02\x04") == 0x07
    assert sum8(b"\xff\x02") == 0x01


# ---------------------------------------------------------------- simulator
def test_simulator_is_deterministic():
    a = simulate(5, None, seed=3)
    b = simulate(5, None, seed=3)
    assert [(f.timestamp, f.can_id, f.data) for f in a] == [(f.timestamp, f.can_id, f.data) for f in b]


def test_simulator_labels_dos():
    fr = simulate(20, "dos", seed=4)
    atk = [f for f in fr if f.label]
    assert atk and all(f.can_id == 0 for f in atk)


# ----------------------------------------------------------------------- io
def test_csv_roundtrip(tmp_path):
    fr = simulate(2, "spoof_rpm", seed=5)
    p = tmp_path / "x.csv"
    save_csv(fr, p)
    back = load(p)
    assert len(back) == len(fr)
    assert back[10].data == fr[10].data and back[-1].label == fr[-1].label


def test_hcrl_and_candump_parsers(tmp_path):
    p = tmp_path / "gear_dataset.csv"
    p.write_text("1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
                 "1478198376.389636,018f,8,fe,5b,00,00,00,3c,00,00,R\n"
                 "1478198376.390000,043f,8,01,45,60,ff,6b,00,00,00,T\n"
                 "1478198376.391000,05f0,2,01,00,R\n")
    fr = load(p)
    assert len(fr) == 4 and fr[2].label == 1 and fr[3].dlc == 2 and fr[3].data == b"\x01\x00"
    q = tmp_path / "dump.log"
    q.write_text("(1436509052.249713) vcan0 044#2A366C2BBA\n(1436509052.250000) vcan0 316#0521\n")
    fr = load(q)
    assert fr[0].can_id == 0x044 and fr[0].data == bytes.fromhex("2A366C2BBA")


# ------------------------------------------------------------------- engine
def test_profile_learns_checksums(profile):
    algos = {r["algo"] for p in profile.payload.prof.values() for r in p["bytes"] if r["role"] == "checksum"}
    assert algos == {"xor", "sum", "crc8"}


def test_clean_traffic_has_no_false_positives(profile):
    m, _, _ = run(profile, None)
    assert m["fp"] <= 2


@pytest.mark.parametrize("attack,min_f1", [("dos", 0.99), ("fuzzy", 0.99), ("spoof_rpm", 0.99),
                                           ("spoof_gear", 0.9), ("replay", 0.95), ("masquerade", 0.95)])
def test_attacks_detected(profile, attack, min_f1):
    m, _, _ = run(profile, attack)
    assert m["f1"] >= min_f1, m


def test_alerts_are_explainable(profile):
    _, alerts, _ = run(profile, "masquerade")
    kinds = {a.kind for a in alerts}
    assert "CHECKSUM_FAIL" in kinds
    assert all(a.message for a in alerts)


def test_stealth_masquerade_evades_anomaly_detection(profile):
    """Documents the limitation that motivates SecOC-lite."""
    m, _, _ = run(profile, "stealth_masquerade")
    assert m["recall"] < 0.2


def test_profile_save_load(profile, tmp_path):
    p = tmp_path / "profile.json"
    profile.save(p)
    m1, _, _ = run(profile, "spoof_rpm", duration=20)
    m2, _, _ = run(Profile.load(p), "spoof_rpm", duration=20)
    assert m1 == m2


# -------------------------------------------------------------------- SecOC
def _pair():
    ks = KeyStore([0x4B0], seed=b"test")
    return SecOCSender(ks[0x4B0], 0x4B0), SecOCReceiver(ks[0x4B0], 0x4B0), ks


def test_secoc_accepts_legit_across_freshness_wrap_and_loss():
    tx, rx, _ = _pair()
    ok = 0
    for i in range(1000):
        pdu = tx.protect(bytes([i & 0xFF]) * 8)
        if i % 7 == 3:
            continue  # lost frame
        ok += rx.verify(pdu)[0]
    assert ok == 1000 - len([i for i in range(1000) if i % 7 == 3])


def test_secoc_rejects_replay_and_tampering():
    tx, rx, _ = _pair()
    pdus = [tx.protect(bytes(8)) for _ in range(5)]
    assert all(rx.verify(p)[0] for p in pdus)
    assert not rx.verify(pdus[2])[0]                       # replay
    p = tx.protect(bytes(8))
    bad = bytes([p[0] ^ 1]) + p[1:]
    assert not rx.verify(bad)[0]                           # tampered payload
    assert rx.verify(p)[0]                                 # genuine one still fine


def test_auth_layer_flags_forgeries():
    tx, _, ks = _pair()
    auth = AuthLayer(ks)
    good = tx.protect(bytes(8))
    assert auth.check(CANFrame(0.0, 0x4B0, 12, good)) == []
    a = auth.check(CANFrame(0.01, 0x4B0, 12, bytes(12)))
    assert a and a[0].kind == "AUTH_FAIL"


# ------------------------------------------------------------------ gateway
def test_gateway_blocks_flood_and_keeps_legit(profile):
    frames = simulate(30, "dos", seed=21)
    _, st = SecureGateway(Engine(profile)).run(frames)
    d = st.as_dict()
    assert d["attack_block_rate"] == 1.0 and d["legit_pass_rate"] > 0.999
