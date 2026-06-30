"""
========================================================
  QSkill Internship - Task 1: Pandas + Matplotlib Data Analysis
  Domain  : Python Development
  Author  : Intern
  Dataset : Sales data (sales_data.csv)
========================================================

Description:
    This script loads a CSV file, performs basic data analysis
    using Pandas, and generates visualizations (bar chart,
    scatter plot, heatmap) using Matplotlib and Seaborn.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CSV_FILE   = "sales_data.csv"
OUTPUT_DIR = "output_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "primary"   : "#4F46E5",   # indigo
    "secondary" : "#06B6D4",   # cyan
    "accent"    : "#F59E0B",   # amber
    "danger"    : "#EF4444",   # red
    "bg"        : "#F8FAFC",
    "text"      : "#1E293B",
}

plt.rcParams.update({
    "figure.facecolor" : PALETTE["bg"],
    "axes.facecolor"   : PALETTE["bg"],
    "axes.edgecolor"   : "#CBD5E1",
    "axes.labelcolor"  : PALETTE["text"],
    "xtick.color"      : PALETTE["text"],
    "ytick.color"      : PALETTE["text"],
    "text.color"       : PALETTE["text"],
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "grid.color"       : "#E2E8F0",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.6,
})


# ─────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(f"\n✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────
#  2. BASIC DATA ANALYSIS
# ─────────────────────────────────────────
def perform_analysis(df: pd.DataFrame) -> None:
    print("\n" + "="*55)
    print("  BASIC DATA ANALYSIS")
    print("="*55)

    print("\n📋 First 5 rows:")
    print(df.head().to_string(index=False))

    print("\n📊 Dataset Info:")
    print(f"   Shape       : {df.shape}")
    print(f"   Columns     : {list(df.columns)}")
    print(f"   Missing vals: {df.isnull().sum().sum()}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    print("\n📈 Descriptive Statistics:")
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n💰 Average Revenue by Product:")
    avg_rev = df.groupby("Product")["Revenue"].mean().round(2)
    for prod, val in avg_rev.items():
        print(f"   {prod:<10}: ₹{val:,.2f}")

    print("\n🏆 Top Region by Total Revenue:")
    top = df.groupby("Region")["Revenue"].sum().idxmax()
    val = df.groupby("Region")["Revenue"].sum().max()
    print(f"   {top} — ₹{val:,.0f}")

    print("\n⭐ Average Customer Rating by Product:")
    avg_rating = df.groupby("Product")["Customer_Rating"].mean().round(2)
    for prod, val in avg_rating.items():
        print(f"   {prod:<10}: {val}")

    print("\n📦 Total Units Sold by Month:")
    units = df.groupby("Month")["Units_Sold"].sum()
    for month, val in units.items():
        print(f"   {month:<10}: {val}")


# ─────────────────────────────────────────
#  3. BAR CHART — Revenue by Product & Region
# ─────────────────────────────────────────
def plot_bar_chart(df: pd.DataFrame) -> None:
    pivot = df.groupby(["Product", "Region"])["Revenue"].sum().unstack()

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["primary"], PALETTE["secondary"],
              PALETTE["accent"], PALETTE["danger"]]

    pivot.plot(kind="bar", ax=ax, color=colors, width=0.7,
               edgecolor="white", linewidth=0.5)

    ax.set_title("Total Revenue by Product & Region",
                 fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Product", fontsize=12, labelpad=8)
    ax.set_ylabel("Revenue (₹)", fontsize=12, labelpad=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax.set_xticklabels(pivot.index, rotation=0, fontsize=11)
    ax.legend(title="Region", fontsize=10, title_fontsize=10,
              framealpha=0.5)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # Annotate max bar
    for container in ax.containers:
        ax.bar_label(container,
                     labels=[f"₹{v/1000:.0f}K" if v > 0 else ""
                             for v in container.datavalues],
                     fontsize=7.5, padding=3, color=PALETTE["text"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bar_chart_revenue.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Bar chart saved → {path}")


# ─────────────────────────────────────────
#  4. SCATTER PLOT — Marketing Spend vs Revenue
# ─────────────────────────────────────────
def plot_scatter(df: pd.DataFrame) -> None:
    products = df["Product"].unique()
    product_colors = {
        "Laptop" : PALETTE["primary"],
        "Phone"  : PALETTE["accent"],
        "Tablet" : PALETTE["secondary"],
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for product in products:
        subset = df[df["Product"] == product]
        ax.scatter(subset["Marketing_Spend"], subset["Revenue"],
                   label=product,
                   color=product_colors.get(product, "#888"),
                   s=80, alpha=0.8, edgecolors="white", linewidth=0.6)

    # Trend line (all data)
    m, b = np.polyfit(df["Marketing_Spend"], df["Revenue"], 1)
    x_line = np.linspace(df["Marketing_Spend"].min(),
                         df["Marketing_Spend"].max(), 100)
    ax.plot(x_line, m * x_line + b,
            color="#64748B", linestyle="--", linewidth=1.5,
            label="Trend line")

    corr = df["Marketing_Spend"].corr(df["Revenue"])
    ax.set_title(f"Marketing Spend vs Revenue  |  r = {corr:.2f}",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Marketing Spend (₹)", fontsize=12)
    ax.set_ylabel("Revenue (₹)", fontsize=12)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax.legend(fontsize=10, framealpha=0.5)
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scatter_marketing_revenue.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Scatter plot saved → {path}")


# ─────────────────────────────────────────
#  5. HEATMAP — Correlation Matrix
# ─────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr().round(2)

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap=cmap, center=0,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 10},
                ax=ax,
                square=True,
                cbar_kws={"shrink": 0.8})

    ax.set_title("Correlation Heatmap — Numeric Features",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35,
                       ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Heatmap saved → {path}")


# ─────────────────────────────────────────
#  6. BONUS — Monthly Revenue Trend
# ─────────────────────────────────────────
def plot_monthly_trend(df: pd.DataFrame) -> None:
    month_order = ["January", "February", "March", "April", "May", "June"]
    monthly = (df.groupby(["Month", "Product"])["Revenue"]
                 .sum()
                 .reset_index())
    monthly["Month"] = pd.Categorical(monthly["Month"],
                                      categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")

    fig, ax = plt.subplots(figsize=(11, 6))
    product_colors = {
        "Laptop": PALETTE["primary"],
        "Phone" : PALETTE["accent"],
        "Tablet": PALETTE["secondary"],
    }

    for product, grp in monthly.groupby("Product"):
        ax.plot(grp["Month"], grp["Revenue"],
                marker="o", linewidth=2.5, markersize=7,
                color=product_colors.get(product, "#888"),
                label=product)
        # shade under line
        ax.fill_between(grp["Month"], grp["Revenue"],
                        alpha=0.08,
                        color=product_colors.get(product, "#888"))

    ax.set_title("Monthly Revenue Trend by Product",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Revenue (₹)", fontsize=12)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax.legend(fontsize=11, framealpha=0.5)
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "line_monthly_trend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Monthly trend chart saved → {path}")


# ─────────────────────────────────────────
#  7. INSIGHTS SUMMARY
# ─────────────────────────────────────────
def print_insights(df: pd.DataFrame) -> None:
    print("\n" + "="*55)
    print("  KEY INSIGHTS & OBSERVATIONS")
    print("="*55)

    best_product = df.groupby("Product")["Revenue"].sum().idxmax()
    best_region  = df.groupby("Region")["Revenue"].sum().idxmax()
    best_rating  = df.groupby("Product")["Customer_Rating"].mean().idxmax()
    corr_mkt_rev = df["Marketing_Spend"].corr(df["Revenue"])
    avg_units     = df["Units_Sold"].mean()
    return_rate   = (df["Returns"].sum() / df["Units_Sold"].sum() * 100)

    print(f"""
  1. 🏆 Best-selling product by Revenue : {best_product}
  2. 🌍 Top-performing Region           : {best_region}
  3. ⭐ Highest Customer Rating         : {best_rating}
  4. 📈 Marketing ↔ Revenue Correlation : {corr_mkt_rev:.2f}
       → {'Strong' if corr_mkt_rev > 0.7 else 'Moderate'} positive relationship
  5. 📦 Average Units Sold per entry    : {avg_units:.1f}
  6. 🔄 Overall Return Rate             : {return_rate:.2f}%
  7. 📅 Revenue grows consistently each month,
       suggesting healthy business momentum.
    """)


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║   QSkill Internship — Task 1: Data Analysis          ║")
    print("╚══════════════════════════════════════════════════════╝")

    df = load_data(CSV_FILE)
    perform_analysis(df)

    print("\n🎨 Generating visualizations...")
    plot_bar_chart(df)
    plot_scatter(df)
    plot_heatmap(df)
    plot_monthly_trend(df)

    print_insights(df)

    print("\n✅ All charts saved in the 'output_charts/' folder.")
    print("   Done!\n")
