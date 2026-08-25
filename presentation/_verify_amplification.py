import numpy as np
from pathlib import Path

npz = Path("cache/microsoft_wavlm-large/predict_artifacts_multiK.npz")
print("bundle exists:", npz.exists())
b = np.load(npz, allow_pickle=True)
seeds = b["seeds"].tolist()
bk1 = b["betas_k1"]
bk2 = b["betas_k2"]
print("seeds:", seeds)
print("betas_k1:", bk1.tolist())
print("betas_k2:", bk2.tolist())
print()

hdr = "{:>6} {:>4} {:>9} {:>9} {:>6} {:>6} {:>13} {:>13}".format(
    "seed", "grp", "z_sigma", "z_mu", "b_k1", "b_k2", "amp_k1", "amp_k2/grp")
print(hdr)
for i, s in enumerate(seeds):
    for g in ("g4", "g5"):
        sig = float(b["s{}_{}_z_sigma".format(s, g)][0])
        mu = float(b["s{}_{}_z_mu".format(s, g)][0])
        b1 = float(bk1[i])
        b2 = float(bk2[i])
        amp1 = (b1 / sig) if g == "g4" else float("nan")
        amp2 = b2 / (2 * sig)
        print("{:>6} {:>4} {:>9.4f} {:>9.4f} {:>6.1f} {:>6.1f} {:>13.2f} {:>13.2f}".format(
            s, g, sig, mu, b1, b2, amp1, amp2))
print()

g4_sig = np.array([float(b["s{}_g4_z_sigma".format(s)][0]) for s in seeds])
g5_sig = np.array([float(b["s{}_g5_z_sigma".format(s)][0]) for s in seeds])
print("G4 z_sigma range: {:.3f} - {:.3f}".format(g4_sig.min(), g4_sig.max()))
print("G5 z_sigma range: {:.3f} - {:.3f}".format(g5_sig.min(), g5_sig.max()))
amp_k1 = bk1 / g4_sig
print("K1 raw-logit amp (b1/g4_sig): {:.2f} - {:.2f}x".format(amp_k1.min(), amp_k1.max()))
print("K1 one-train-sigma contribution (b1): {:.1f} - {:.1f} final-logit units".format(bk1.min(), bk1.max()))
print("K2 per-group one-train-sigma contribution (b2/2): {:.1f} - {:.1f} final-logit units".format((bk2 / 2).min(), (bk2 / 2).max()))
