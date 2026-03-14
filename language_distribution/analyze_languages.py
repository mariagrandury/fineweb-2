import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("fineweb2-language-distribution.csv")

train = df[(df["split"] == "train") & (df["words"] != "-")].copy()
train["words"] = train["words"].astype(int)

families = (
    train.groupby("family")
    .agg(
        num_languages=("name", "nunique"),
        total_words=("words", "sum"),
        languages=("name", lambda x: sorted(x.unique().tolist())),
    )
    .sort_values("total_words", ascending=False)
    .reset_index()
)

families.to_csv("families.csv", index=False)
print(f"Saved families.csv with {len(families)} families")

# --- Circle (pie) plot ---
TOP_N = 20
top = families.head(TOP_N)
other_words = families.iloc[TOP_N:]["total_words"].sum()
if other_words > 0:
    other_row = pd.DataFrame([{"family": "Other", "total_words": other_words}])
    plot_data = pd.concat([top[["family", "total_words"]], other_row], ignore_index=True)
else:
    plot_data = top[["family", "total_words"]]

fig, ax = plt.subplots(figsize=(10, 10))
wedges, texts, autotexts = ax.pie(
    plot_data["total_words"],
    labels=plot_data["family"],
    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    startangle=90,
    pctdistance=0.75,
    wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
)

for t in texts:
    t.set_fontsize(9)
for at in autotexts:
    at.set_fontsize(8)

total = families["total_words"].sum()
ax.set_title(
    f"FineWeb-2 Train Words by Language Family\nTotal: {total:,.0f} words",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

plt.tight_layout()
plt.savefig("families_words_distribution.png", dpi=150, bbox_inches="tight")
print("Saved families_words_distribution.png")

# --- Circle (pie) plot: number of languages per family ---
by_langs = families.sort_values("num_languages", ascending=False)
TOP_N = 20
top = by_langs.head(TOP_N)
other_langs = by_langs.iloc[TOP_N:]["num_languages"].sum()
if other_langs > 0:
    other_row = pd.DataFrame([{"family": "Other", "num_languages": other_langs}])
    plot_data2 = pd.concat([top[["family", "num_languages"]], other_row], ignore_index=True)
else:
    plot_data2 = top[["family", "num_languages"]]

fig, ax = plt.subplots(figsize=(10, 10))
wedges, texts, autotexts = ax.pie(
    plot_data2["num_languages"],
    labels=plot_data2["family"],
    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    startangle=90,
    pctdistance=0.75,
    wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
)

for t in texts:
    t.set_fontsize(9)
for at in autotexts:
    at.set_fontsize(8)

total_langs = families["num_languages"].sum()
ax.set_title(
    f"FineWeb-2 Languages per Family\nTotal: {total_langs:,} languages",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

plt.tight_layout()
plt.savefig("families_languages_distribution.png", dpi=150, bbox_inches="tight")
print("Saved families_languages_distribution.png")
