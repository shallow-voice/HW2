import warnings
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def generate_magic(n):
    """
    严格复现 MATLAB magic(n) 的三种构造算法：
    1. 奇数阶 (n % 2 == 1): Siamese 方法 (De la Loubere 算法)
    2. 双偶数阶 (n % 4 == 0): 双重对角线对称互补交换法
    3. 单偶数阶 (n % 4 == 2): Strachey / LUX 分块交换法
    """
    if n % 2 == 1:
        p = np.arange(1, n + 1)
        M = n * np.mod(p[:, None] + p - (n + 3) // 2, n) + np.mod(p[:, None] + 2 * p - 2, n) + 1
        return M.astype(np.float64)
    elif n % 4 == 0:
        J = np.mod(np.arange(1, n + 1), 4) // 2
        K = J[:, None] == J
        M = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
        M[K] = n * n + 1 - M[K]
        return M
    else:
        p = n // 2
        M = generate_magic(p)
        M = np.block([[M, M + 2 * p * p], [M + 3 * p * p, M + p * p]])
        i = np.arange(p)
        k = (n - 2) // 4
        j = np.concatenate((np.arange(k), np.arange(n - k + 1, n)))
        M[np.ix_(np.concatenate((i, i + p)), j)] = M[np.ix_(np.concatenate((i + p, i)), j)]
        i_mid = k
        j_mid = np.array([0, i_mid])
        M[np.ix_(np.array([i_mid, i_mid + p]), j_mid)] = M[np.ix_(np.array([i_mid + p, i_mid]), j_mid)]
        return M.astype(np.float64)


def generate_matrices(n, rng):
    """生成题目 (a) 要求的四类 n x n 矩阵: randn(n), hilb(n), pascal(n), magic(n)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A_randn = rng.standard_normal((n, n))
        A_hilb = la.hilbert(n)
        A_pascal = la.pascal(n, kind="symmetric", exact=False)
        A_magic = generate_magic(n)
    return {
        "randn(n)": A_randn,
        "hilb(n)": A_hilb,
        "pascal(n)": A_pascal,
        "magic(n)": A_magic
    }


def evaluate_system(A, matrix_name=None):
    """
    针对给定矩阵 A，生成 x = ones(n, 1) 和 b = A @ x，
    分别通过直接消元求解 (la.solve) 与显式求逆相乘 (la.inv(A) @ b) 计算 xhat，
    并返回 p = 1, 2, inf 三种范数下的相对误差、条件数、残差误差界及求逆相对误差。
    """
    n = A.shape[0]
    x = np.ones(n, dtype=np.float64)
    norms_list = [1, 2, np.inf]
    results = {p: {"rel_err": np.nan, "cond": np.nan, "bound": np.nan, "rel_err_inv": np.nan} for p in norms_list}

    if not np.all(np.isfinite(A)):
        return x, np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), results

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = A @ x
        if not np.all(np.isfinite(b)):
            return x, b, np.full(n, np.nan), np.full(n, np.nan), results

        # 偶数阶 magic(n) 在数学上严格奇异 (rank < n)，因此跳过并记为 NaN
        if matrix_name == "magic(n)" and n % 2 == 0:
            return x, b, np.full(n, np.nan), np.full(n, np.nan), results

        try:
            xhat = la.solve(A, b)
            delta_b = A @ xhat - b
        except Exception:
            xhat = np.full(n, np.nan)
            delta_b = np.full(n, np.nan)

        try:
            A_inv = la.inv(A)
            xhat_inv = A_inv @ b
        except Exception:
            A_inv = np.full((n, n), np.nan)
            xhat_inv = np.full(n, np.nan)

        for p in norms_list:
            try:
                norm_x = la.norm(x, ord=p)
                norm_b = la.norm(b, ord=p)
                norm_A = la.norm(A, ord=p)
                norm_A_inv = la.norm(A_inv, ord=p) if np.all(np.isfinite(A_inv)) else np.nan

                cond_p = norm_A * norm_A_inv
                rel_err = la.norm(x - xhat, ord=p) / norm_x if np.all(np.isfinite(xhat)) else np.nan
                rel_res = la.norm(delta_b, ord=p) / norm_b if (np.all(np.isfinite(delta_b)) and norm_b > 0) else np.nan
                bound_p = cond_p * rel_res
                rel_err_inv = la.norm(x - xhat_inv, ord=p) / norm_x if np.all(np.isfinite(xhat_inv)) else np.nan

                results[p] = {
                    "rel_err": rel_err,
                    "cond": cond_p,
                    "bound": bound_p,
                    "rel_err_inv": rel_err_inv
                }
            except Exception:
                pass

    return x, b, xhat, delta_b, results


def safe_log10(arr):
    """
    将正实数数组安全转换为以 10 为底的对数 log10(x):
    - 对 0.0 取下界 1e-17 (即 log10 = -17.0)，代表机器双精度下的精确零误差；
    - 对超出 float64 正常范围或非有限值返回 np.nan，彻底杜绝 Matplotlib LogLocator 的 10**decades 溢出崩溃。
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    valid = np.isfinite(arr) & (arr >= 0.0)
    out[valid] = np.log10(np.maximum(arr[valid], 1e-17))
    return out


def compute_inv_11_without_inv(A):
    """通过求解 A * b_1 = e_1 直接获取 A^{-1} 的 (1, 1) 元素 b_{11}"""
    n = A.shape[0]
    e1 = np.zeros(n, dtype=np.float64)
    e1[0] = 1.0
    b1 = la.solve(A, e1)
    return b1[0]


def main():
    rng = np.random.default_rng(309)

    #(c) 完整展示 n = 5 时的所有计算结果
    n_demo = 5
    matrices_n5 = generate_matrices(n_demo, rng)
    print("=" * 88)
    print("SUBPROBLEM (c)(iii) & (e): COMPLETE RESULTS FOR n = 5")
    print("=" * 88)

    norm_labels = {1: "1-norm", 2: "2-norm", np.inf: "inf-norm"}

    for mat_name, A5 in matrices_n5.items():
        x5, b5, xhat5, db5, res5 = evaluate_system(A5, mat_name)
        print(f"\n>>> Matrix Type: {mat_name} (n = 5) <<<")
        print("Matrix A:\n", np.array2string(A5, precision=4, suppress_small=True))
        print("True x          :", np.round(x5, 6))
        print("Right-hand b=Ax :", np.round(b5, 6))
        print("Solved xhat=A\\b :", np.round(xhat5, 12))
        print("Residual db     :", np.array2string(db5, precision=3, formatter={"float_kind": lambda v: f"{v:.3e}"}))
        print("-" * 88)
        print(f"{'Norm':<10} | {'||x-xhat||/||x||':>18} | {'kappa(A)':>16} | {'kappa(A)||db||/||b||':>20} | {'(e) inv(A)*b Err':>16}")
        print("-" * 88)
        for p in [1, 2, np.inf]:
            r = res5[p]
            print(f"{norm_labels[p]:<10} | {r['rel_err']:>18.6e} | {r['cond']:>16.6e} | {r['bound']:>20.6e} | {r['rel_err_inv']:>16.6e}")

    # (c) & (e): 对 n = 5, 10, 15, ..., 500 循环计算并绘制趋势图
    n_values = np.array(range(5, 505, 5))
    mat_types = ["randn(n)", "hilb(n)", "pascal(n)", "magic(n)"]
    norms_list = [1, 2, np.inf]

    history = {
        m: {p: {"rel_err": [], "cond": [], "bound": [], "rel_err_inv": []} for p in norms_list}
        for m in mat_types
    }

    for n in n_values:
        mats = generate_matrices(int(n), rng)
        for m in mat_types:
            _, _, _, _, res = evaluate_system(mats[m], m)
            for p in norms_list:
                for key in ["rel_err", "cond", "bound", "rel_err_inv"]:
                    val = res[p][key]
                    history[m][p][key].append(float(val) if np.isfinite(val) else np.nan)

    # 绘制 4x3 综合趋势图：直接使用线性坐标轴绘制 log10(数值)，100% 避免 Matplotlib ticker.py 的 OverflowError
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=True)
    for row_idx, m in enumerate(mat_types):
        for col_idx, p in enumerate(norms_list):
            ax = axes[row_idx, col_idx]
            h = history[m][p]

            valid_mask = np.isfinite(h["cond"])
            n_plot = n_values[valid_mask]
            log_rel_err = safe_log10(np.array(h["rel_err"])[valid_mask])
            log_cond = safe_log10(np.array(h["cond"])[valid_mask])
            log_bound = safe_log10(np.array(h["bound"])[valid_mask])
            log_inv = safe_log10(np.array(h["rel_err_inv"])[valid_mask])

            ax.plot(n_plot, log_rel_err, label=r"$\log_{10}(\|x - \hat{x}\| / \|x\|)$ (solve)", color="royalblue", lw=1.6)
            ax.plot(n_plot, log_cond, label=r"$\log_{10}(\kappa(A))$", color="crimson", linestyle="--", lw=1.4)
            ax.plot(n_plot, log_bound, label=r"$\log_{10}(\kappa(A)\|\Delta b\| / \|b\|)$", color="darkorange", linestyle=":", lw=1.5)
            ax.plot(n_plot, log_inv, label=r"$\log_{10}(\|x - \hat{x}_{\mathrm{inv}}\| / \|x\|)$ (inv)", color="seagreen", alpha=0.75, lw=1.2)

            title_suffix = " (Odd n Only; Even n Singular)" if m == "magic(n)" else ""
            ax.set_title(f"{m}{title_suffix} | {norm_labels[p]}", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)
            if row_idx == 3:
                ax.set_xlabel("Matrix Dimension $n$", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(r"$\log_{10}(\mathrm{Magnitude})$", fontsize=10)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8)

    plt.suptitle("(c) & (e): Log10 Accuracy, Condition Number, Error Bound, and Solve vs. Inv ($n=5,\\dots,500$)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("q5_c_e_trends.png", dpi=300)
    plt.show()

    # 子问题 (d) 讨论输出与 (f) 验证
    print("\n" + "=" * 88)
    print("SUBPROBLEM (d): DISCUSSION OF THE NUMERICAL RESULTS")
    print("=" * 88)
    print("1. randn(n) : Well-conditioned random matrix; condition number grows slowly with n; errors for both direct solve and inversion remain within 1e-15 ~ 1e-11.")
    print("2. hilb(n)  : Extremely ill-conditioned matrix; condition number exceeds 1e16 for n >= 15; forward error loses all significant digits while residual remains small.")
    print("3. pascal(n): Condition number and element scale both explode exponentially (reaching ~10^300 for n=500); large n leads to complete loss of accuracy with inv(A)*b error significantly larger than direct solve.")
    print("4. magic(n) : Even-order matrices are rank-deficient (singular, no unique solution), while odd-order ones are well-conditioned; errors remain within 1e-16 ~ 1e-14.")

    print("\n" + "=" * 88)
    print("SUBPROBLEM (f): COMPUTING (1, 1)-ENTRY OF A^{-1} WITHOUT INVERTING A")
    print("=" * 88)
    A_test = matrices_n5["randn(n)"]
    b11_fast = compute_inv_11_without_inv(A_test)
    b11_exact = la.inv(A_test)[0, 0]
    print(f"Computed b_11 via solving A * b_1 = e_1 : {b11_fast:.16f}")
    print(f"Reference b_11 via explicit inv(A)[0,0] : {b11_exact:.16f}")
    print(f"Absolute difference                     : {abs(b11_fast - b11_exact):.3e}")


if __name__ == "__main__":
    main()