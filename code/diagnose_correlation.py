"""
v12 추가 분석: SNR과 separability_score가 서로 독립적인 정보인지 편상관/다중회귀로 확인.
diagnose_subjects.py 결과(results/v12/diagnose_subjects.json)와
v9의 피험자별 base_acc(results/v9/finetune_per_subject.json)를 사용.
"""
import json
import numpy as np
import numpy.linalg as la
from scipy.stats import pearsonr, t as tdist


def partial_corr(r_xy, r_xz, r_yz):
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))


def main():
    diag = {r["subject"]: r for r in json.load(open("../results/v12/diagnose_subjects.json"))}
    acc = {r["subject"]: r["base_acc"] for r in json.load(open("../results/v9/finetune_per_subject.json"))}
    subs = sorted(diag.keys())
    snr = np.array([diag[s]["snr_db"] for s in subs])
    sep = np.array([diag[s]["separability_score"] for s in subs])
    a = np.array([acc[s] for s in subs])
    n = len(subs)

    r_snr_sep, p_snr_sep = pearsonr(snr, sep)
    r_snr_a, _ = pearsonr(snr, a)
    r_sep_a, _ = pearsonr(sep, a)

    r_snr_a_given_sep = partial_corr(r_snr_a, r_snr_sep, r_sep_a)
    r_sep_a_given_snr = partial_corr(r_sep_a, r_snr_sep, r_snr_a)

    def z(x):
        return (x - x.mean()) / x.std()

    X = np.column_stack([np.ones(n), z(snr), z(sep)])
    beta, _, _, _ = np.linalg.lstsq(X, a, rcond=None)
    pred = X @ beta
    ss_res = np.sum((a - pred) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2_combined = 1 - ss_res / ss_tot

    dof = n - X.shape[1]
    sigma2 = ss_res / dof
    se = np.sqrt(np.diag(la.inv(X.T @ X)) * sigma2)
    tvals = beta / se
    pvals = 2 * (1 - tdist.cdf(np.abs(tvals), dof))

    result = {
        "r_snr_separability": r_snr_sep, "p_snr_separability": p_snr_sep,
        "r_snr_acc": r_snr_a, "r_separability_acc": r_sep_a,
        "partial_r_snr_acc_given_sep": r_snr_a_given_sep,
        "partial_r_sep_acc_given_snr": r_sep_a_given_snr,
        "r2_snr_only": r_snr_a ** 2, "r2_sep_only": r_sep_a ** 2, "r2_combined": r2_combined,
        "coef_intercept": beta[0], "coef_snr_z": beta[1], "coef_sep_z": beta[2],
        "p_snr_z": pvals[1], "p_sep_z": pvals[2],
    }
    print(json.dumps(result, indent=2))
    with open("../results/v12/correlation_analysis.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
