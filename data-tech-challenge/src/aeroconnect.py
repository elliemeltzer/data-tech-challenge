#Ellie Meltzer -- Tech Challenege

#imports
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

#File Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_raw" / "TechChallenge_Data.xlsx"
OUT = ROOT / "aero_out"
(OUT/ "data").mkdir(parents = True, exist_ok= True)
(OUT / "figs").mkdir(parents = True, exist_ok= True)

#Loadd and Clean Data
def load_clean(path: Path) -> pd.DataFrame:
    #load excel
    df = pd.read_excel(path)
    #clean
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month"]).sort_values("Month")

    df["Passengers_Total"] = pd.to_numeric(df["Passengers_Total"], errors="coerce").fillna(0)
    df["Route"] = df["AustralianPort"].astype(str) + "-" + df["ForeignPort"].astype(str)
    return df

#Top/Bottom Routes
def top_bottom_routes(df: pd.DataFrame, n = 10, min_total = 0):
    #Top 10 Routes
    totals = df.groupby("Route", as_index=False)["Passengers_Total"].sum()
    topn = totals.sort_values("Passengers_Total", ascending=False).head(n)
    #Bottom 10 Routes---exluding 0
    if min_total == 0:
        botn = totals[totals["Passengers_Total"] > 0].sort_values("Passengers_Total").head(n)
    else:
        botn = totals[totals["Passengers_Total"] > min_total].sort_values("Passengers_Total").head(n)
    return topn, botn

#Growth Patterns
def avg_yoy_last12(df: pd.DataFrame, min_last12_total =0):
    #Avg monthly totals by country
    g = (df.groupby(["Month", "Country"], as_index=False)["Passengers_Total"].sum()
            .sort_values("Month"))
    #compute YoY percentage change
    g["YoY_pct"] = g.groupby("Country")["Passengers_Total"].pct_change(12) * 100

    #keep only last 12 months
    last_month = g["Month"].max()
    start = last_month - pd.DateOffset(months=11)
    last12 = g[g["Month"] >= start]
    #filter out small totals
    if min_last12_total >0:
        size = last12.groupby("Country")["Passengers_Total"].sum()
        keep = size[size >= min_last12_total].index
        last12 = last12[last12["Country"].isin(keep)]
    out = (last12.groupby("Country", as_index=False)["YoY_pct"].mean()
                .dropna()
                .sort_values("YoY_pct", ascending=False))
    return out


def monthly_series(df: pd.DataFrame, route: str) -> pd.Series:
    ts = (df[df["Route"] == route]
          .groupby("Month")["Passengers_Total"].sum()
          .sort_index())
    if ts.empty:
        raise ValueError(f"No rows for route: {route}")

    start = pd.Timestamp(ts.index.min()).to_period("M").to_timestamp()
    end = pd.Timestamp(ts.index.max()).to_period("M").to_timestamp()
    idx = pd.date_range(start=start, end=end, freq="MS")
    ts = ts.reindex(idx, fill_value=0.0)
    ts.index.name = "Month"
    return ts

def forecast_route_holtwinters(df: pd.DataFrame, route: str, horizon: int = 12):
    ts = monthly_series(df, route)
    # months used for better modeling
    if len(ts) < 36:
        raise ValueError("Need at least 36 months of history for model to be reliable")

    test = ts.iloc[-12:]
    train = ts.iloc[:-12]
    #canidate model configurations
    candidates = [
        ("add", None),
        ("add", "add"),
        ("add", "mul"),
        (None, "add"),
        (None, "mul"),
    ]

    best = None
    for trend, seasonal in candidates:
        try:
            fit = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=12
            ).fit(optimized=True, use_brute=True)

            preds = fit.forecast(len(test))
            #compute Mean Absolute Percentage Error (MAPE)
            denom = test.replace(0, np.nan)
            mape = float((np.abs(preds - test) / denom).dropna().mean() * 100)

            #Keep best model
            if np.isfinite(mape) and (best is None or mape < best["mape"]):
                best = {"trend": trend, "seasonal": seasonal, "fit": fit, "preds": preds, "mape": mape}
        except Exception:
            # skip configs that fail to converge
            continue

    if best is None:
        raise RuntimeError("Holt–Winters could not find a convergent configuration.")

    # Final forecast
    future = best["fit"].forecast(horizon)
    rmse = float(np.sqrt(np.mean((best["preds"] - test) ** 2)))
    mae  = float(np.mean(np.abs(best["preds"] - test)))

    forecast = {idx.strftime("%Y-%m-%d"): float(val) for idx, val in future.items()}

    return {
        "route": route,
        "model": "holt_winters",
        "trend": best["trend"],
        "seasonal": best["seasonal"],
        "history_months": int(len(ts)),
        "rmse": rmse,
        "mae": mae,
        "mape": float(best["mape"]),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "forecast": forecast
    }

#Forcast for models
def forecast_route(df: pd.DataFrame, route: str, horizon=12):
    ts = monthly_series(df, route)
    if len(ts) < 24:
        raise ValueError("Need at least 24 months for baseline")
    test = ts.iloc[-12:]
    train = ts.iloc[:-12]

    preds_test = train.iloc[-12:].values
    mae = float(np.mean(np.abs(preds_test - test.values)))
    rmse = float(np.sqrt(np.mean((preds_test - test.values)**2)))
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mape = float(np.nanmean(np.abs((preds_test - test.values) / np.where(test.values==0, np.nan, test.values))) * 100)
    #future forecast
    future_vals = ts.iloc[-12:].values
    future_idx = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    reps = int(np.ceil(horizon/12))
    forecast_values = np.tile(future_vals, reps)[:horizon]
    forecast = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(future_idx, forecast_values)}

    return {
        "route" : route,
        "model" : "seasonal model",
        "history_months" : int(len(ts)),
        "rmse" : rmse,
        "mae" : mae,
        "mape" : mape,
        "test_start" : str(test.index.min().date()),
        "test_end" : str(test.index.max().date()),
        "forecast" : forecast
    }

def plot_top_routes_bar(df: pd.DataFrame, outpath: Path, n: int = 10) -> None:
    totals = df.groupby("Route", as_index=False)["Passengers_Total"].sum()
    tops = totals.sort_values("Passengers_Total", ascending=False).head(n)
    plt.figure()
    plt.barh(tops["Route"][::-1], tops["Passengers_Total"][::-1])
    plt.title(f"Top {n} Routes by Total Passengers")
    plt.xlabel("Passengers")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_top_routes_timeseries(df: pd.DataFrame, outpath: Path, n: int = 5) -> None:
    totals = df.groupby("Route", as_index=False)["Passengers_Total"].sum()
    top_routes = totals.sort_values("Passengers_Total", ascending=False)["Route"].head(n).tolist()
    ts = (df[df["Route"].isin(top_routes)]
            .groupby(["Month", "Route"], as_index=False)["Passengers_Total"].sum())
    plt.figure()
    for r in top_routes:
        s = ts[ts["Route"] == r].sort_values("Month")
        plt.plot(s["Month"], s["Passengers_Total"], label=r)
    plt.title(f"Top {n} Routes — Monthly Passengers")
    plt.xlabel("Month"); plt.ylabel("Passengers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

##MAIN
def main():
    #Load and clean data
    df = load_clean(DATA_PATH)
    df.to_csv(OUT / "data" / "cleaned.csv", index=False)

    #Save top 10 and bottom 10 routes
    top10, bottom10 = top_bottom_routes(df, n=10, min_total=0)
    top10.to_csv(OUT / "data" / "top10_routes.csv", index=False)
    bottom10.to_csv(OUT / "data" / "bottom10_routes_nonzero.csv", index=False)

    #Average YoY growth by country
    growth = avg_yoy_last12(df, min_last12_total=50_000)
    growth.to_csv(OUT / "data" / "avg_yoy_last12_by_country.csv", index=False)

    #Select Busiest route for forecasting
    busiest = top10.iloc[0]["Route"]

    #Baseline model--more simple
    summary_naive = forecast_route(df, busiest, horizon=12)
    with open(OUT / "data" / "model_summary.json", "w") as f:
        json.dump(summary_naive, f, indent=2)

    #Learned model: Holt–Winters
    summary_hw = forecast_route_holtwinters(df, busiest, horizon=12)
    with open(OUT / "data" / "model_summary_holtwinters.json", "w") as f:
        json.dump(summary_hw, f, indent=2)

    #comparison of model metrics
    cmp = pd.DataFrame([
        {"model": summary_naive["model"], "rmse": summary_naive["rmse"], "mae": summary_naive["mae"],
         "mape": summary_naive["mape"]},
        {"model": summary_hw["model"], "rmse": summary_hw["rmse"], "mae": summary_hw["mae"],
         "mape": summary_hw["mape"]},
    ])
    cmp.to_csv(OUT / "data" / "model_comparison.csv", index=False)

    plot_top_routes_bar(df, OUT / "figs" / "top_routes.png", n=10)
    plot_top_routes_timeseries(df, OUT / "figs" / "time_series_top5.png", n=5)

if __name__ == "__main__":
    main()
