import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── configuration ──────────────────────────────────────────────────────────────
N = 100
TARGET = 1_500_000_000_000   # 1.5 trillion words
IE_SUBFAMILIES = True


# ── data processing ────────────────────────────────────────────────────────────

def load_training_data(path: str = "fineweb2-language-distribution.csv") -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(path)
    train = df[(df["split"] == "train") & (df["words"] != "-")].copy()
    train["words"] = train["words"].astype(int)
    # Disambiguate languages that appear in multiple scripts (e.g. Serbian Cyrillic
    # and Latin), so duplicate names don't appear silently in the table.
    duplicated_names = train["name"][train["name"].duplicated(keep=False)].unique()
    train["name"] = train.apply(
        lambda r: f"{r['name']} ({r['script']})" if r["name"] in duplicated_names else r["name"],
        axis=1,
    )
    return train, train["words"].sum()


def build_ie_lang_map(path: str = "ie_subfamilies.json") -> dict[str, str]:
    with open(path) as fh:
        subfamilies: dict[str, list[str]] = json.load(fh)
    return {lang: f"{subfamily} (IE)" for subfamily, langs in subfamilies.items() for lang in langs}


def greedy_allocate(items: list[tuple[str, float]], budget: float) -> dict[str, float]:
    """Equal-share greedy allocation, processing items smallest-first.

    Each item gets min(remaining_budget / remaining_items, available).
    Unspent budget from small items is redistributed to the remaining ones.
    """
    allocation: dict[str, float] = {}
    remaining = budget
    for i, (key, available) in enumerate(sorted(items, key=lambda x: x[1])):
        per_item = remaining / (len(items) - i)
        alloc = min(per_item, float(available))
        allocation[key] = alloc
        remaining -= alloc
    return allocation


def compute_mixture(top: pd.DataFrame, target: float) -> pd.DataFrame:
    family_totals = top.groupby("display_family")["words"].sum().to_dict()

    family_allocation = greedy_allocate(list(family_totals.items()), target)

    rows = []
    for family, group in top.groupby("display_family"):
        fam_alloc = family_allocation[family]
        # Use DataFrame index as key to avoid collisions from languages that
        # appear in multiple scripts (e.g. Serbian Cyrillic + Latin).
        lang_items = [(idx, float(row["words"])) for idx, row in group.iterrows()]
        lang_allocation = greedy_allocate(lang_items, fam_alloc)

        for idx, lang_row in group.iterrows():
            rows.append({
                "name": lang_row["name"],
                "display_family": family,
                "available_words": int(lang_row["words"]),
                "allocated_words": lang_allocation[idx],
                "family_allocated": fam_alloc,
            })

    return pd.DataFrame(rows)


def prepare_mixture_data(
    train: pd.DataFrame,
    n: int,
    ie_subfamilies: bool,
    target: float,
) -> pd.DataFrame:
    top = train.nlargest(n, "words").copy()

    if ie_subfamilies:
        lang_to_family = build_ie_lang_map()
        top["display_family"] = top.apply(
            lambda r: lang_to_family.get(r["name"], r["family"]), axis=1
        )
    else:
        top["display_family"] = top["family"]

    df = compute_mixture(top, target)

    df["pct_of_mixture"]     = df["allocated_words"] / target * 100
    df["pct_available_used"] = df["allocated_words"] / df["available_words"] * 100

    family_rank = {
        f: i for i, f in enumerate(
            df.groupby("display_family")["allocated_words"]
            .sum().sort_values(ascending=False).index
        )
    }
    df["_family_rank"] = df["display_family"].map(family_rank)
    df = df.sort_values(
        ["_family_rank", "allocated_words"], ascending=[True, False]
    ).reset_index(drop=True)

    return df


# ── table rendering ────────────────────────────────────────────────────────────

PALETTE = [
    "#EBF5FB", "#E8F8F5", "#FEF9E7", "#F9EBEA", "#F4ECF7",
    "#EAFAF1", "#FDF2E9", "#EAF2F8", "#FDEDEC", "#E9F7EF",
    "#FEF5E7", "#F0F3F4", "#F5EEF8", "#E8F5E9", "#FFF3E0",
    "#E3F2FD", "#FCE4EC", "#F9FBE7", "#EDE7F6", "#E0F7FA",
]

COLS = [
    (0.010, 0.250, "Language Family (alloc. words, % of 1.5T)", "left"),
    (0.255, 0.455, "Language",                                   "left"),
    (0.460, 0.615, "Available Words",                            "right"),
    (0.620, 0.775, "Allocated Words",                            "right"),
    (0.780, 0.885, "% of Mixture",                               "right"),
    (0.890, 0.990, "% Available Used",                           "right"),
]

ROW_H    = 0.30
HEADER_H = 0.45
TITLE_H  = 0.80
MARGIN_B = 0.20
FIG_W    = 15.0
TEXT_PAD = 0.010
HEADER_BG = "#2C3E50"


def make_figure(n_rows: int) -> tuple[plt.Figure, plt.Axes]:
    fig_h = TITLE_H + HEADER_H + n_rows * ROW_H + MARGIN_B
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    return fig, ax


def row_bottom(fig_h: float, i: int) -> float:
    return fig_h - TITLE_H - HEADER_H - i * ROW_H


def row_mid(fig_h: float, i: int) -> float:
    h = HEADER_H if i == 0 else ROW_H
    return row_bottom(fig_h, i) + h / 2


def draw_title(ax: plt.Axes, fig_h: float, n: int, target: float, total_allocated: float) -> None:
    ax.text(
        0.5, fig_h - 0.28,
        f"FineWeb-2 · Proposed Data Mixture  (Top {n} Languages, Training Split)",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A252F",
    )
    ax.text(
        0.5, fig_h - 0.58,
        f"Target: {target / 1e12:.1f}T words  ·  Total allocated: {total_allocated / 1e12:.3f}T words",
        ha="center", va="center", fontsize=9.5, color="#555555",
    )


def draw_header(ax: plt.Axes, fig_h: float) -> None:
    for x0, x1, label, align in COLS:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, row_bottom(fig_h, 0)), x1 - x0, HEADER_H,
            boxstyle="square,pad=0", linewidth=0, facecolor=HEADER_BG,
        ))
        tx = x0 + TEXT_PAD if align == "left" else x1 - TEXT_PAD
        ax.text(tx, row_mid(fig_h, 0), label,
                ha=align, va="center", fontsize=8.5, fontweight="bold", color="white")


def draw_rows(ax: plt.Axes, fig_h: float, df: pd.DataFrame, target: float) -> None:
    families_ordered = list(dict.fromkeys(df["display_family"]))
    family_bg = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(families_ordered)}

    seen: set[str] = set()
    family_first_idx: dict[str, int] = {}
    for idx, row in df.iterrows():
        if row["display_family"] not in seen:
            family_first_idx[row["display_family"]] = int(idx)
            seen.add(row["display_family"])

    prev_family = None
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        bg = family_bg[row["display_family"]]
        yb = row_bottom(fig_h, i)
        ym = row_mid(fig_h, i)

        for x0, x1, _, _ in COLS:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, yb), x1 - x0, ROW_H,
                boxstyle="square,pad=0", linewidth=0.3,
                edgecolor="#CCCCCC", facecolor=bg,
            ))

        if (i - 1) == family_first_idx[row["display_family"]]:
            fam_pct = row["family_allocated"] / target * 100
            ax.text(COLS[0][0] + TEXT_PAD, ym + ROW_H * 0.13,
                    row["display_family"],
                    ha="left", va="center", fontsize=7.5, fontweight="bold", color="#1A252F")
            ax.text(COLS[0][0] + TEXT_PAD, ym - ROW_H * 0.18,
                    f"{row['family_allocated']:,.0f} words · {fam_pct:.1f}%",
                    ha="left", va="center", fontsize=6.5, color="#555555")

        ax.text(COLS[1][0] + TEXT_PAD, ym, row["name"],
                ha="left", va="center", fontsize=8, color="#1A252F")
        ax.text(COLS[2][1] - TEXT_PAD, ym, f"{row['available_words']:,.0f}",
                ha="right", va="center", fontsize=8, color="#1A252F")
        ax.text(COLS[3][1] - TEXT_PAD, ym, f"{row['allocated_words']:,.0f}",
                ha="right", va="center", fontsize=8, color="#1A252F")
        ax.text(COLS[4][1] - TEXT_PAD, ym, f"{row['pct_of_mixture']:.2f}%",
                ha="right", va="center", fontsize=8, color="#1A252F")
        ax.text(COLS[5][1] - TEXT_PAD, ym, f"{row['pct_available_used']:.1f}%",
                ha="right", va="center", fontsize=8, color="#1A252F")

        if prev_family is not None and row["display_family"] != prev_family:
            ax.plot([COLS[0][0], COLS[-1][1]], [yb + ROW_H, yb + ROW_H],
                    color="#555555", linewidth=0.9, zorder=5)

        prev_family = row["display_family"]


def render_table(df: pd.DataFrame, target: float, n: int) -> plt.Figure:
    total_allocated = df["allocated_words"].sum()
    fig, ax = make_figure(len(df))
    fig_h = fig.get_figheight()
    draw_title(ax, fig_h, n, target, total_allocated)
    draw_header(ax, fig_h)
    draw_rows(ax, fig_h, df, target)
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

train, _ = load_training_data()
df = prepare_mixture_data(train, N, IE_SUBFAMILIES, TARGET)

total_allocated = df["allocated_words"].sum()
print(f"Families: {df['display_family'].nunique()}")
print(f"Target:   {TARGET / 1e12:.1f}T words")
print(f"Allocated: {total_allocated / 1e12:.4f}T words  ({total_allocated / TARGET * 100:.2f}% of target)")

fig = render_table(df, TARGET, N)
fig.savefig("data_mixture.png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
print("Saved data_mixture.png")
