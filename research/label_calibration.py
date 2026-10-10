"""How should report labels be turned into training targets that mean what the experts mean?

The 58 expert-labelled studies show that a report label and an expert label are different things:
a hedged or mild report finding (pilkwang grade 0.70) is expert-positive in 0% (ACL, MCL) to 50%
(Effusion) of studies, yet the consensus table trains it as ~0.9. This script compares ways of mapping
the report labels to P(expert positive), each fitted on 57 gold studies and scored on the one left out:

  raw        the consensus as it is (what the reader is trained on now)
  pooled     one logistic curve on logit(consensus) for all findings
  finding    one curve per finding, shrunk toward the pooled curve
  grades     per finding, the expert-positive share of each pilkwang grade level, shrunk toward the
             all-findings share of that level and made monotone
  sources    per finding, a logistic model on the four label tables, shrunk toward equal weights

Reported: macro AUC and log-loss of the left-out predictions (a calibration cannot beat its input's
ranking by much; the point is the log-loss, i.e. targets that are right on average).

Usage (from the RSNA project folder): python research/label_calibration.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LEVELS = [0.10, 0.30, 0.70, 0.80, 0.95]          # pilkwang: NO, UNK, YES hedged / plain / definite
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
logit = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))
sig = lambda z: 1 / (1 + np.exp(-z))


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_logistic(X, y, prior, lam):
    """Logistic regression with a ridge penalty toward `prior` (X has a leading column of ones)."""
    def f(th):
        z = X @ th
        return np.logaddexp(0, z).sum() - (y * z).sum() + lam * ((th - prior) ** 2).sum()
    return minimize(f, prior, jac=lambda th: X.T @ (sig(X @ th) - y) + 2 * lam * (th - prior), method="L-BFGS-B").x


def level_of(score):
    return np.abs(np.asarray(score)[:, None] - np.array(LEVELS)[None, :]).argmin(1)


def monotone(p, w):
    """Weighted pool-adjacent-violators: non-decreasing fit to p."""
    p, w = list(map(float, p)), list(map(float, w))
    blocks = [[v, wt, 1] for v, wt in zip(p, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            blocks[i:i + 2] = [[(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, _, n in blocks])


class Calibrator:
    """Fitted on gold rows (cons, lev, src, y: arrays [n, 12]); `predict` gives targets for any rows."""

    def __init__(self, kind, lam=2.0, m=4.0):
        self.kind, self.lam, self.m = kind, lam, m

    def fit(self, cons, lev, src, y):
        z = logit(cons)
        if self.kind in ("pooled", "finding"):
            X = np.stack([np.ones(z.size), z.ravel()], 1)
            self.pool = fit_logistic(X, y.ravel(), np.array([0.0, 1.0]), 1e-3)
            self.th = [self.pool if self.kind == "pooled" else
                       fit_logistic(np.stack([np.ones(len(z)), z[:, j]], 1), y[:, j], self.pool, self.lam) for j in range(len(T))]
        elif self.kind == "grades":
            # all-findings share of each level, then per finding shrunk toward it (m pseudo-studies) and made monotone
            # from the hedged level upward; NO and UNK are kept as they are (a silent report may hide more than a denial)
            pool = np.array([(y[lev == L].sum() + 0.5) / ((lev == L).sum() + 1.0) for L in range(len(LEVELS))])
            self.tab = []
            for j in range(len(T)):
                n = np.array([(lev[:, j] == L).sum() for L in range(len(LEVELS))], float)
                k = np.array([y[lev[:, j] == L, j].sum() for L in range(len(LEVELS))], float)
                p = (k + self.m * pool) / (n + self.m)
                p[2:] = monotone(p[2:], n[2:] + self.m)
                self.tab.append(np.clip(p, 0.01, 0.99))
        elif self.kind == "sources":
            self.th = []
            for j in range(len(T)):
                X = np.concatenate([np.ones((len(y), 1)), logit(src[:, j, :])], 1)
                prior = np.concatenate([[0.0], np.full(src.shape[2], 1.0 / src.shape[2])])
                self.th.append(fit_logistic(X, y[:, j], prior, self.lam))
        return self

    def predict(self, cons, lev, src):
        if self.kind == "raw":
            return np.clip(cons, 0.01, 0.99)
        if self.kind in ("pooled", "finding"):
            return np.stack([sig(self.th[j][0] + self.th[j][1] * logit(cons[:, j])) for j in range(len(T))], 1)
        if self.kind == "grades":
            return np.stack([self.tab[j][lev[:, j]] for j in range(len(T))], 1)
        return np.stack([sig(np.concatenate([np.ones((len(cons), 1)), logit(src[:, j, :])], 1) @ self.th[j]) for j in range(len(T))], 1)


def load():
    lab = ld("runs/solo/gpjw_labels_v0.csv")
    tabs = {"steven": ld("research/public_notebooks/knee_g2/lab2/llm_labels_v4_blend.csv"),
            "pilkwang": ld("research/public_notebooks/knee_g2/lab1/report_labels_v2.csv"),
            "lixin": ld("data/labels/labels_llm_gpt56sol.csv")}
    q = ld("data/labels/qwen_knee_weak_labels.csv")
    tabs["qwen"] = q[[f"{c}__label" for c in T]].set_axis(T, axis=1)
    cons = lab[T].values
    # a source that lacks a study gets the consensus there (as the consensus itself is the mean of those present)
    src = np.stack([tabs[k][T].astype(float).reindex(lab.index).fillna(lab[T]).values for k in tabs], 2)
    pilk = tabs["pilkwang"][T].astype(float).reindex(lab.index)
    lev = np.stack([np.where(pilk[c].isna(), 1, level_of(pilk[c].fillna(0.3).values)) for c in T], 1)   # no pilkwang row: treated as UNK
    return lab, cons, lev, src


if __name__ == "__main__":
    lab, cons, lev, src = load()
    g = (lab.is_gold == 1).values
    y = lab.loc[g, [f"gold_{c}" for c in T]].values.astype(float)
    C, L, S = cons[g], lev[g], src[g]
    kinds = [("raw", {}), ("pooled", {}), ("finding", {"lam": 2.0}), ("finding", {"lam": 0.5}), ("grades", {"m": 4.0}), ("grades", {"m": 2.0}),
             ("sources", {"lam": 2.0}), ("sources", {"lam": 0.5})]
    rows, keep = [], {}
    for kind, kw in kinds:
        P = np.zeros_like(y)
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            P[i] = Calibrator(kind, **kw).fit(C[tr], L[tr], S[tr], y[tr]).predict(C[i:i + 1], L[i:i + 1], S[i:i + 1])[0] if kind != "raw" else np.clip(C[i], 0.01, 0.99)
        name = kind + "".join(f" {k}={v}" for k, v in kw.items())
        keep[name] = P
        ll = -(y * np.log(P) + (1 - y) * np.log(1 - P)).mean(0)
        a = np.array([auc(y[:, j], P[:, j]) for j in range(len(T))])
        rows.append({"mapping": name, "macro AUC": a.mean(), "log-loss": ll.mean(), **{f"ll {t[:10]}": ll[j] for j, t in enumerate(T)}})
    print("left-one-study-out on gold-58 (each study predicted by a mapping fitted on the other 57)")
    print(pd.DataFrame(rows).set_index("mapping").round(3).to_string())
    a = pd.DataFrame({name: [auc(y[:, j], P[:, j]) for j in range(len(T))] for name, P in keep.items()}, index=T)
    print("\nAUC per finding of the left-out predictions")
    print(a.round(3).to_string())
    full = Calibrator("grades").fit(C, L, S, y)
    print("\ngrades mapping fitted on all 58 (target per pilkwang level; NO, UNK, hedged, plain, definite):")
    print(pd.DataFrame(np.stack(full.tab), index=T, columns=[f"{v:.2f}" for v in LEVELS]).round(2).to_string())
    fin = Calibrator("finding").fit(C, L, S, y)
    grid = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    print("\nfinding mapping fitted on all 58 (target at a given consensus value):")
    print(pd.DataFrame(np.stack([sig(fin.th[j][0] + fin.th[j][1] * logit(grid)) for j in range(len(T))]), index=T, columns=[f"{v:.2f}" for v in grid]).round(2).to_string())
    weak = ~g
    for name, cal in (("grades", full), ("finding", fin)):
        P = cal.predict(cons[weak], lev[weak], src[weak])
        print(f"\n{name}: mean target on the 4,349 report-labelled studies vs the consensus mean and the expert share on gold-58")
        print(pd.DataFrame({"calibrated": P.mean(0), "consensus": cons[weak].mean(0), "experts, gold-58": y.mean(0)}, index=T).round(2).T.to_string())
