import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def main():

    # 1. 读取本地数据文件
    with open("row.txt", "r", encoding="utf-8") as f:
        countries = [line.strip() for line in f if line.strip()]

    with open("column.txt", "r", encoding="utf-8") as f:
        variables = [line.strip() for line in f if line.strip()]

    # 读取 49x7 的逗号分隔矩阵 processed.txt
    A = np.loadtxt("processed.txt", delimiter=",")

    if A.shape != (49, 7):
        raise ValueError(f"processed.txt 应为 49x7，实际读取到 {A.shape}")
    if len(countries) != A.shape[0] or len(variables) != A.shape[1]:
        raise ValueError("row.txt、column.txt 的标签数量与 processed.txt 不一致")


    #(a): 说明为什么二维投影坐标为 (sigma_j*u_ij, sigma_k*u_ik)
    print("=" * 72)
    print("(a) If A = U Sigma V^T, then the coordinate of the i-th row vector a_i along v_j is:")
    print("    <a_i, v_j> = e_i^T A v_j = e_i^T U Sigma e_j = sigma_j u_ij.")
    print("    Therefore, in span{v_j, v_k}, it suffices to plot (sigma_j u_ij, sigma_k u_ik).")


    #(b): 计算 A 的 SVD，获取前两个右奇异向量 v1, v2 并绘制国家散点图

    # scipy.linalg.svd 返回 U (49x49), 奇异值一维数组 s (7,), 以及 V^T (7x7)
    U, s, Vt = la.svd(A, full_matrices=True)
    V = Vt.T

    # 提取前两个右奇异向量 v1, v2 in R^7
    v1 = V[:, 0]
    v2 = V[:, 1]

    print("=" * 72)
    print("(b) The first two right singular vectors v1, v2 (corresponding to singular values sigma_1 = {:.4f}, sigma_2 = {:.4f}):".format(s[0], s[1]))
    print("-" * 72)
    print(f"{'Demographic Variable':<28} {'v1':>18} {'v2':>18}")
    print("-" * 72)
    for var_name, val1, val2 in zip(variables, v1, v2):
        print(f"{var_name:<28} {val1:>18.6f} {val2:>18.6f}")

    # 根据 (a) 的结论，行向量在 span{v1, v2} 上的投影坐标 A @ [v1, v2] 等于 (sigma_1 * u_{i1}, sigma_2 * u_{i2})
    row_proj_x = s[0] * U[:, 0]
    row_proj_y = s[1] * U[:, 1]

    # 自动识别离原点最远的两个明显离群点，而不是写死国家名称。
    row_radii = np.hypot(row_proj_x, row_proj_y)
    outlier_indices = np.argsort(row_radii)[-2:]
    outlier_names = [countries[i] for i in outlier_indices]
    print("\n(b) The two distinct outliers farthest from the origin:", ", ".join(outlier_names))

    # 绘制 (b) 49 个国家的二维投影散点图
    plt.figure(figsize=(11, 8))
    plt.scatter(row_proj_x, row_proj_y, c="royalblue", edgecolors="k", s=45, alpha=0.85)
    for i, country in enumerate(countries):
        plt.annotate(
            country,
            (row_proj_x[i], row_proj_y[i]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5
        )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel(r"$v_1$ (First Right Singular Vector)", fontsize=11)
    plt.ylabel(r"$v_2$ (Second Right Singular Vector)", fontsize=11)
    plt.title(r"(b) Projection of 49 Countries onto $\mathrm{span}\{v_1, v_2\}$", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("q1_b_countries.png", dpi=300)
    plt.show()

    # (c): 将 7 个列向量（人口统计指标）投影到左奇异向量子空间 span{u1, u2}

    # 列向量 alpha_j 在 u1, u2 上的投影坐标 U[:, :2].T @ A 等于 (sigma_1 * v_{j1}, sigma_2 * v_{j2})
    col_proj_x = s[0] * V[:, 0]
    col_proj_y = s[1] * V[:, 1]

    print("\n" + "=" * 72)
    print("(c) Projected coordinates of the 7 demographic variables onto span{u1, u2}:")
    print("-" * 72)
    for var_name, cx, cy in zip(variables, col_proj_x, col_proj_y):
        print(f"{var_name:<28} : (u1 = {cx:>9.4f}, u2 = {cy:>9.4f})")

    plt.figure(figsize=(9, 7))
    plt.scatter(col_proj_x, col_proj_y, c="crimson", marker="D", edgecolors="k", s=65)
    for i, var_name in enumerate(variables):
        plt.annotate(
            var_name,
            (col_proj_x[i], col_proj_y[i]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="darkred"
        )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel(r"$u_1$ (First Left Singular Vector)", fontsize=11)
    plt.ylabel(r"$u_2$ (Second Left Singular Vector)", fontsize=11)
    plt.title(r"(c) Projection of 7 Demographic Variables onto $\mathrm{span}\{u_1, u_2\}$", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("q1_c_variables.png", dpi=300)
    plt.show()

    # (d): 叠加 (b) 与 (c) 绘制双标图 (Biplot)
    plt.figure(figsize=(12, 8.5))
    plt.scatter(row_proj_x, row_proj_y, c="royalblue", edgecolors="k", s=45, alpha=0.75, label="Countries (Rows)")
    for i, country in enumerate(countries):
        plt.annotate(
            country,
            (row_proj_x[i], row_proj_y[i]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            color="navy"
        )

    plt.scatter(col_proj_x, col_proj_y, c="crimson", marker="D", edgecolors="k", s=70, label="Demographic Variables (Columns)")
    for i, var_name in enumerate(variables):
        plt.arrow(
            0, 0, col_proj_x[i], col_proj_y[i],
            color="crimson", alpha=0.45, width=0.015, head_width=0.12, length_includes_head=True
        )
        plt.annotate(
            var_name,
            (col_proj_x[i], col_proj_y[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="darkred"
        )

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel(r"First Singular Vector Direction ($v_1$ / $u_1$)", fontsize=11)
    plt.ylabel(r"Second Singular Vector Direction ($v_2$ / $u_2$)", fontsize=11)
    plt.title("(d) Biplot: Overlay of Countries and Demographic Variables", fontsize=13)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("q1_d_biplot.png", dpi=300)
    plt.show()

    # 找到与两个离群国家组成的点集距离最近的两个变量。
    outlier_coords = np.column_stack((row_proj_x[outlier_indices], row_proj_y[outlier_indices]))
    variable_coords = np.column_stack((col_proj_x, col_proj_y))
    distance_to_outliers = np.min(
        np.linalg.norm(variable_coords[:, None, :] - outlier_coords[None, :, :], axis=2),
        axis=1
    )
    nearest_variable_indices = np.argsort(distance_to_outliers)[:2]
    nearest_variables = [variables[i] for i in nearest_variable_indices]
    print("\n(d) The two demographic variables closest to the two outlier countries:")
    for i in nearest_variable_indices:
        print(f"    {variables[i]:<28} Minimum 2D distance = {distance_to_outliers[i]:.4f}")


    # 5.(e): 剔除两个显著离群点 (Hong Kong, Singapore) 后对 47x7 矩阵重做 (b)
    outliers = set(outlier_names)
    keep_indices = [i for i, c in enumerate(countries) if c not in outliers]

    A_reduced = A[keep_indices, :]
    countries_reduced = [countries[i] for i in keep_indices]

    # 对 47x7 矩阵重新执行 SVD
    U_red, s_red, Vt_red = la.svd(A_reduced, full_matrices=True)
    proj_red_x = s_red[0] * U_red[:, 0]
    proj_red_y = s_red[1] * U_red[:, 1]

    plt.figure(figsize=(11, 8))
    plt.scatter(proj_red_x, proj_red_y, c="seagreen", edgecolors="k", s=45, alpha=0.85)
    for i, country in enumerate(countries_reduced):
        plt.annotate(
            country,
            (proj_red_x[i], proj_red_y[i]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if country in {"Japan", "USSR", "Austria", "Finland"} else "normal",
            color="darkred" if country == "Japan" else "black"
        )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel(r"$v_1$ (Reduced $47 \times 7$ Matrix)", fontsize=11)
    plt.ylabel(r"$v_2$ (Reduced $47 \times 7$ Matrix)", fontsize=11)
    plt.title(r"(e) Projection of 47 Countries onto $\mathrm{span}\{v_1, v_2\}$ (Hong Kong & Singapore Removed)", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("q1_e_reduced_countries.png", dpi=300)
    plt.show()

    # 计算新 2D 投影平面上所有国家到 Japan 的欧氏距离
    japan_idx = countries_reduced.index("Japan")
    dists = np.hypot(proj_red_x - proj_red_x[japan_idx], proj_red_y - proj_red_y[japan_idx])
    sorted_order = np.argsort(dists)

    print("\n" + "=" * 72)
    print("(e) After removing outliers, the 6 countries closest to Japan in the 2D projection space:")
    print("-" * 72)
    for rank, idx in enumerate(sorted_order[1:7], start=1):
        print(f"{rank}. {countries_reduced[idx]:<16} : 2D Euclidean distance = {dists[idx]:.4f}, Coordinates = ({proj_red_x[idx]:>7.4f}, {proj_red_y[idx]:>7.4f})")

    # 题目问的是“最像 Japan 的两个欧洲国家”，因此必须先限定欧洲国家集合。
    european_countries = {
        "Austria", "Belgium", "Bulgaria", "Cyprus", "Czechoslovakia", "Denmark",
        "Finland", "France", "Hungary", "Iceland", "Ireland", "Italy", "Luxembourg",
        "Malta", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Spain",
        "Sweden", "Switzerland", "United Kingdom", "USSR", "West Germany", "Yugoslavia"
    }
    european_indices = [
        i for i, country in enumerate(countries_reduced)
        if country in european_countries
    ]
    nearest_european = sorted(european_indices, key=lambda i: dists[i])[:2]

    print("\n(e) The two European countries closest to Japan in the new 2D projection space:")
    for rank, idx in enumerate(nearest_european, start=1):
        print(f"{rank}. {countries_reduced[idx]:<16} : 2D Euclidean distance = {dists[idx]:.6f}")


if __name__ == "__main__":
    main()
