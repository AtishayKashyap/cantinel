"""Minimal end-to-end example: learn a car, detect an attack, block it, authenticate it.

    python examples/quickstart.py
"""
from cantinel import Engine, Profile, simulate
from cantinel.gateway import SecureGateway
from cantinel.metrics import prf
from cantinel.secoc import KeyStore, SecOCReceiver, SecOCSender

# 1. learn the vehicle from attack-free traffic (+ a second drive for self-tuning)
profile = Profile.train(simulate(180, seed=1), simulate(60, seed=2))
print(profile.summary())

# 2. detect: a capture with RPM-spoofing bursts (HCRL Car-Hacking style)
capture = simulate(60, "spoof_rpm", seed=3)
flags, alerts = Engine(profile).run(capture)
m = prf([f.label for f in capture], flags)
print(f"\nspoof_rpm -> precision {m['precision']:.3f}  recall {m['recall']:.3f}")
print("first alert:", next(a for a in alerts if a.flags).to_dict())

# 3. prevent: the secure gateway drops what the engine flags
_, stats = SecureGateway(Engine(profile)).run(capture)
print("\ngateway:", {k: v for k, v in stats.as_dict().items() if k.endswith("rate")})

# 4. authenticate: SecOC-lite stops even a perfect (stealth) masquerade
keys = KeyStore([0x4B0])
tx, rx = SecOCSender(keys[0x4B0], 0x4B0), SecOCReceiver(keys[0x4B0], 0x4B0)
genuine = tx.protect(bytes.fromhex("1027102700000A5C"))
forged = bytes.fromhex("A00FA00F00000B33") + bytes([2]) + genuine[-3:]  # plausible payload, guessed FV, stolen MAC
print("\nSecOC genuine:", rx.verify(genuine)[2], "| forged:", rx.verify(forged)[2],
      "| replayed:", rx.verify(genuine)[2])
