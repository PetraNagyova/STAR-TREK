from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from setup.common import (
    load_episode_df,
    load_relevant_characters_df,
    build_meta_X_y_groups,
)

OUT_DIR = Path("../output_models")
FIG_DIR = OUT_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_logo_cv(estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    g_np = groups.to_numpy()

    preds = np.full(len(y_np), np.nan, dtype=float)

    uniq = pd.Series(g_np).dropna().unique()
    if len(uniq) >= 2:
        splitter = LeaveOneGroupOut()
        splits = splitter.split(X_np, y_np, groups=g_np)
    else:
        n_splits = min(5, len(y_np))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = splitter.split(X_np, y_np)

    for tr, te in splits:
        est = clone(estimator)
        est.fit(X_np[tr], y_np[tr])
        preds[te] = est.predict(X_np[te])

    ok = ~np.isnan(preds)
    rmse = float(np.sqrt(mean_squared_error(y_np[ok], preds[ok])))
    mae = float(mean_absolute_error(y_np[ok], preds[ok]))
    r2 = float(r2_score(y_np[ok], preds[ok])) if ok.sum() >= 2 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_pred_oof": preds,
    }


def pick_best_alpha_ridge(bundle, alphas):
    best_a, best_rmse = None, float("inf")
    for a in alphas:
        r = run_logo_cv(Ridge(alpha=a), bundle.X, bundle.y, bundle.groups)
        if r["rmse"] < best_rmse:
            best_rmse = r["rmse"]
            best_a = a
    return float(best_a), float(best_rmse)


def pick_best_alpha_lasso(bundle, alphas):
    best_a, best_rmse = None, float("inf")
    for a in alphas:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lasso", Lasso(alpha=a, max_iter=40000, random_state=0)),
            ]
        )
        r = run_logo_cv(model, bundle.X, bundle.y, bundle.groups)
        if r["rmse"] < best_rmse:
            best_rmse = r["rmse"]
            best_a = a
    return float(best_a), float(best_rmse)


def fit_full(estimator, bundle):
    est = clone(estimator)
    est.fit(bundle.X.to_numpy(dtype=float), bundle.y.to_numpy(dtype=float))
    return est


def get_coefs(estimator, feature_names) -> pd.Series:
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
    elif hasattr(estimator, "named_steps") and "lasso" in estimator.named_steps:
        coef = np.asarray(estimator.named_steps["lasso"].coef_, dtype=float)
    else:
        raise ValueError("Estimator nema koeficienty")

    coef = coef.ravel()
    return pd.Series(coef, index=list(feature_names))


def plot_top_signed(series: pd.Series, title: str, save_path: Path, top_k: int = 25) -> None:
    s = series.copy()
    if len(s) == 0:
        return

    idx = np.argsort(np.abs(s.to_numpy()))[::-1][:top_k]
    s = s.iloc[idx]

    fig = plt.figure()
    plt.bar(s.index.astype(str), s.to_numpy())
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("koeficient")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def save_pred_vs_true(y_true, y_pred_oof, save_path: Path, title: str) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_oof, dtype=float)
    ok = ~np.isnan(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]

    fig = plt.figure()
    plt.scatter(y_true, y_pred, s=10)

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])

    plt.xlabel("y_true (popularity)")
    plt.ylabel("y_pred (OOF)")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def save_residual_hist(y_true, y_pred_oof, save_path: Path, title: str, bins: int = 40) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_oof, dtype=float)
    ok = ~np.isnan(y_pred)
    resid = y_true[ok] - y_pred[ok]

    fig = plt.figure()
    plt.hist(resid, bins=bins)
    plt.xlabel("reziduum (y_true - y_pred)")
    plt.ylabel("počet epizód")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def main():
    ep = load_episode_df()
    series_list = sorted(ep["series"].astype(str).dropna().unique().tolist())

    rel = load_relevant_characters_df()
    rows = []

    for s in series_list:
        series_dir = FIG_DIR / str(s)
        series_dir.mkdir(parents=True, exist_ok=True)

        models = [
            ("B0_dummy0",
             DummyRegressor(strategy="constant", constant=0.0),
             dict(use_meta=False, use_characters=False, use_director=False)),

            ("B1_linear_meta",
             LinearRegression(),
             dict(use_meta=True, use_characters=False, use_director=False)),

            ("B2_ridge_meta_a1",
             Ridge(alpha=1.0),
             dict(use_meta=True, use_characters=False, use_director=False)),

            ("M1_ridge_meta+chars_a1",
             Ridge(alpha=1.0),
             dict(use_meta=True, use_characters=True, use_director=False)),

            ("M3_ridge_meta+chars+dir_a1",
             Ridge(alpha=1.0),
             dict(use_meta=True, use_characters=True, use_director=True)),
        ]

        for tag, est, cfg in models:
            bundle = build_meta_X_y_groups(
                series_filter=s,
                group_col="season",
                merge_aliases=True,
                main_cast_only=True,
                **cfg,
            )

            r = run_logo_cv(est, bundle.X, bundle.y, bundle.groups)

            rows.append({
                "series": s,
                "model": tag,
                "rmse": r["rmse"],
                "mae": r["mae"],
                "r2": r["r2"],
                "n": len(bundle.y),
                "n_groups": int(pd.Series(bundle.groups).nunique()),
            })

            t = safe_name(tag)
            save_pred_vs_true(
                bundle.y, r["y_pred_oof"],
                series_dir / f"{t}_pred_vs_true.png",
                title=f"{s} / {tag}",
            )
            save_residual_hist(
                bundle.y, r["y_pred_oof"],
                series_dir / f"{t}_residual_hist.png",
                title=f"{s} / {tag}",
            )

        char_set = set(
            rel[rel["series"].astype(str) == str(s)]["character_name_norm"].astype(str).tolist()
        )

        b3_bundle = build_meta_X_y_groups(
            use_meta=True, use_characters=False, use_director=False,
            series_filter=s, group_col="season", merge_aliases=True
        )
        a_b3, _ = pick_best_alpha_ridge(b3_bundle, alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        tag_b3 = f"B3_ridge_meta_best_a{a_b3:g}"
        r = run_logo_cv(Ridge(alpha=a_b3), b3_bundle.X, b3_bundle.y, b3_bundle.groups)

        rows.append({
            "series": s,
            "model": tag_b3,
            "rmse": r["rmse"],
            "mae": r["mae"],
            "r2": r["r2"],
            "n": len(b3_bundle.y),
            "n_groups": int(pd.Series(b3_bundle.groups).nunique()),
        })

        t = safe_name(tag_b3)
        save_pred_vs_true(b3_bundle.y, r["y_pred_oof"], series_dir / f"{t}_pred_vs_true.png", f"{s} / {tag_b3}")
        save_residual_hist(b3_bundle.y, r["y_pred_oof"], series_dir / f"{t}_residual_hist.png", f"{s} / {tag_b3}")

        m1_bundle = build_meta_X_y_groups(
            use_meta=True, use_characters=True, use_director=False,
            series_filter=s, group_col="season", merge_aliases=True
        )
        est_m1 = fit_full(Ridge(alpha=1.0), m1_bundle)
        coef_m1 = get_coefs(est_m1, m1_bundle.X.columns)
        coef_m1_chars = coef_m1[coef_m1.index.isin(char_set)]

        plot_top_signed(
            coef_m1_chars,
            title=f"{s} / M1: najvplyvnejšie postavy (Ridge a=1)",
            save_path=series_dir / f"{safe_name('M1_ridge_meta+chars_a1')}_top_characters.png",
            top_k=25,
        )
        coef_m1_chars.reindex(coef_m1_chars.abs().sort_values(ascending=False).index).head(50).to_csv(
            series_dir / f"{safe_name('M1_ridge_meta+chars_a1')}_top_characters.csv",
            header=["coef"],
        )

        m3_bundle = build_meta_X_y_groups(
            use_meta=True, use_characters=True, use_director=True,
            series_filter=s, group_col="season", merge_aliases=True
        )
        est_m3 = fit_full(Ridge(alpha=1.0), m3_bundle)
        coef_m3 = get_coefs(est_m3, m3_bundle.X.columns)
        coef_m3_chars = coef_m3[coef_m3.index.isin(char_set)]

        plot_top_signed(
            coef_m3_chars,
            title=f"{s} / M3: najvplyvnejšie postavy (Ridge a=1 + director)",
            save_path=series_dir / f"{safe_name('M3_ridge_meta+chars+dir_a1')}_top_characters.png",
            top_k=25,
        )
        coef_m3_chars.reindex(coef_m3_chars.abs().sort_values(ascending=False).index).head(50).to_csv(
            series_dir / f"{safe_name('M3_ridge_meta+chars+dir_a1')}_top_characters.csv",
            header=["coef"],
        )

        m2_bundle = build_meta_X_y_groups(
            use_meta=True, use_characters=True, use_director=False,
            series_filter=s, group_col="season", merge_aliases=True
        )
        a_m2, _ = pick_best_alpha_ridge(m2_bundle, alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])

        tag_m2 = f"M2_ridge_meta+chars_best_a{a_m2:g}"
        r = run_logo_cv(Ridge(alpha=a_m2), m2_bundle.X, m2_bundle.y, m2_bundle.groups)

        rows.append({
            "series": s,
            "model": tag_m2,
            "rmse": r["rmse"],
            "mae": r["mae"],
            "r2": r["r2"],
            "n": len(m2_bundle.y),
            "n_groups": int(pd.Series(m2_bundle.groups).nunique()),
        })

        t = safe_name(tag_m2)
        save_pred_vs_true(m2_bundle.y, r["y_pred_oof"], series_dir / f"{t}_pred_vs_true.png", f"{s} / {tag_m2}")
        save_residual_hist(m2_bundle.y, r["y_pred_oof"], series_dir / f"{t}_residual_hist.png", f"{s} / {tag_m2}")

        est_m2 = fit_full(Ridge(alpha=a_m2), m2_bundle)
        coef_m2 = get_coefs(est_m2, m2_bundle.X.columns)
        coef_m2_chars = coef_m2[coef_m2.index.isin(char_set)]

        plot_top_signed(
            coef_m2_chars,
            title=f"{s} / M2: najvplyvnejšie postavy (Ridge)",
            save_path=series_dir / f"{safe_name(tag_m2)}_top_characters.png",
            top_k=25,
        )

        coef_m2_chars.reindex(coef_m2_chars.abs().sort_values(ascending=False).index).head(50).to_csv(
            series_dir / f"{safe_name(tag_m2)}_top_characters.csv",
            header=["coef"],
        )

        m4_bundle = build_meta_X_y_groups(
            use_meta=True, use_characters=True, use_director=True,
            series_filter=s, group_col="season", merge_aliases=True
        )
        a_m4, _ = pick_best_alpha_ridge(m4_bundle, alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])

        tag_m4 = f"M4_ridge_meta+chars+dir_best_a{a_m4:g}"
        r = run_logo_cv(Ridge(alpha=a_m4), m4_bundle.X, m4_bundle.y, m4_bundle.groups)

        rows.append({
            "series": s,
            "model": tag_m4,
            "rmse": r["rmse"],
            "mae": r["mae"],
            "r2": r["r2"],
            "n": len(m4_bundle.y),
            "n_groups": int(pd.Series(m4_bundle.groups).nunique()),
        })

        t = safe_name(tag_m4)
        save_pred_vs_true(m4_bundle.y, r["y_pred_oof"], series_dir / f"{t}_pred_vs_true.png", f"{s} / {tag_m4}")
        save_residual_hist(m4_bundle.y, r["y_pred_oof"], series_dir / f"{t}_residual_hist.png", f"{s} / {tag_m4}")

        est_m4 = fit_full(Ridge(alpha=a_m4), m4_bundle)
        coef_m4 = get_coefs(est_m4, m4_bundle.X.columns)
        coef_m4_chars = coef_m4[coef_m4.index.isin(char_set)]

        plot_top_signed(
            coef_m4_chars,
            title=f"{s} / M4: najvplyvnejšie postavy (Ridge + director)",
            save_path=series_dir / f"{safe_name(tag_m4)}_top_characters.png",
            top_k=25,
        )
        coef_m4_chars.reindex(coef_m4_chars.abs().sort_values(ascending=False).index).head(50).to_csv(
            series_dir / f"{safe_name(tag_m4)}_top_characters.csv",
            header=["coef"],
        )

        m5_bundle = m4_bundle
        a_m5, _ = pick_best_alpha_lasso(m5_bundle, alphas=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

        tag_m5 = f"M5_lasso_meta+chars+dir_best_a{a_m5:g}"
        lasso = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lasso", Lasso(alpha=a_m5, max_iter=40000, random_state=0)),
            ]
        )
        r = run_logo_cv(lasso, m5_bundle.X, m5_bundle.y, m5_bundle.groups)

        rows.append({
            "series": s,
            "model": tag_m5,
            "rmse": r["rmse"],
            "mae": r["mae"],
            "r2": r["r2"],
            "n": len(m5_bundle.y),
            "n_groups": int(pd.Series(m5_bundle.groups).nunique()),
        })

        t = safe_name(tag_m5)
        save_pred_vs_true(m5_bundle.y, r["y_pred_oof"], series_dir / f"{t}_pred_vs_true.png", f"{s} / {tag_m5}")
        save_residual_hist(m5_bundle.y, r["y_pred_oof"], series_dir / f"{t}_residual_hist.png", f"{s} / {tag_m5}")

        est_m5 = fit_full(lasso, m5_bundle)
        coef_m5 = get_coefs(est_m5, m5_bundle.X.columns)
        coef_m5_chars = coef_m5[coef_m5.index.isin(char_set)]

        plot_top_signed(
            coef_m5_chars,
            title=f"{s} / M5: najvplyvnejšie postavy (Lasso)",
            save_path=series_dir / f"{safe_name(tag_m5)}_top_characters.png",
            top_k=25,
        )
        coef_m5_chars.reindex(coef_m5_chars.abs().sort_values(ascending=False).index).head(50).to_csv(
            series_dir / f"{safe_name(tag_m5)}_top_characters.csv",
            header=["coef"],
        )

    res = pd.DataFrame(rows).sort_values(["series", "rmse", "model"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "series_cv_results.csv", index=False)
    print(res.to_string(index=False))

    for s in sorted(res["series"].unique().tolist()):
        sub = res[res["series"] == s].sort_values("rmse")
        fig = plt.figure()
        plt.barh(sub["model"].astype(str), sub["rmse"].to_numpy())
        plt.xlabel("RMSE (OOF, leave-one-season-out)")
        plt.title(f"{s}: porovnanie modelov")
        fig.tight_layout()
        fig.savefig((FIG_DIR / str(s) / f"{s}_rmse_comparison.png"), dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    main()
