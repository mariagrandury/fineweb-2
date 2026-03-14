import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ── configuration ──────────────────────────────────────────────────────────────
N = 100
IE_SUBFAMILIES = False


# ── data processing ────────────────────────────────────────────────────────────


def load_training_data(
    path: str = "fineweb2-language-distribution.csv",
) -> tuple[pd.DataFrame, int]:
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
    return {
        lang: f"{subfamily} (IE)"
        for subfamily, langs in subfamilies.items()
        for lang in langs
    }


def prepare_table_data(
    train: pd.DataFrame,
    train_total: int,
    n: int,
    ie_subfamilies: bool,
) -> tuple[pd.DataFrame, int]:
    top = train.nlargest(n, "words").copy()
    top_total = top["words"].sum()

    if ie_subfamilies:
        lang_to_family = build_ie_lang_map()
        top["display_family"] = top.apply(
            lambda r: lang_to_family.get(r["name"], r["family"]), axis=1
        )
    else:
        top["display_family"] = top["family"]

    family_agg = (
        top.groupby("display_family")["words"]
        .sum()
        .reset_index()
        .rename(columns={"words": "family_total"})
        .sort_values("family_total", ascending=False)
    )
    family_rank = {f: i for i, f in enumerate(family_agg["display_family"])}

    top = top.merge(family_agg, on="display_family")
    top["_family_rank"] = top["display_family"].map(family_rank)
    top = top.sort_values(
        ["_family_rank", "words"], ascending=[True, False]
    ).reset_index(drop=True)

    top["pct_of_family"] = top["words"] / top["family_total"] * 100
    top["pct_of_top_n"] = top["words"] / top_total * 100
    top["pct_of_total"] = top["words"] / train_total * 100

    return top, top_total


# ── table rendering ────────────────────────────────────────────────────────────

PALETTE = [
    "#EBF5FB",
    "#E8F8F5",
    "#FEF9E7",
    "#F9EBEA",
    "#F4ECF7",
    "#EAFAF1",
    "#FDF2E9",
    "#EAF2F8",
    "#FDEDEC",
    "#E9F7EF",
    "#FEF5E7",
    "#F0F3F4",
    "#F5EEF8",
    "#E8F5E9",
    "#FFF3E0",
    "#E3F2FD",
    "#FCE4EC",
    "#F9FBE7",
    "#EDE7F6",
    "#E0F7FA",
]

COLS = [
    (0.010, 0.265, "Language Family (words, % of top N)", "left"),
    (0.270, 0.490, "Language", "left"),
    (0.495, 0.690, "Words", "right"),
    (0.695, 0.790, "% of Family", "right"),
    (0.795, 0.892, f"% of Top {N}", "right"),
    (0.897, 0.990, "% of Total", "right"),
]

ROW_H = 0.30
HEADER_H = 0.45
TITLE_H = 0.80
MARGIN_B = 0.20
FIG_W = 15.0
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


def draw_title(
    ax: plt.Axes, fig_h: float, n: int, top_total: int, train_total: int
) -> None:
    ax.text(
        0.5,
        fig_h - 0.28,
        f"FineWeb-2 · Top {n} Languages by Word Count  (Training Split)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#1A252F",
    )
    ax.text(
        0.5,
        fig_h - 0.58,
        f"Top-{n} total: {top_total:,.0f} words  ·  Full training split total: {train_total:,.0f} words",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
    )


def draw_header(ax: plt.Axes, fig_h: float) -> None:
    for x0, x1, label, align in COLS:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, row_bottom(fig_h, 0)),
                x1 - x0,
                HEADER_H,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor=HEADER_BG,
            )
        )
        tx = x0 + TEXT_PAD if align == "left" else x1 - TEXT_PAD
        ax.text(
            tx,
            row_mid(fig_h, 0),
            label,
            ha=align,
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )


def draw_rows(ax: plt.Axes, fig_h: float, top: pd.DataFrame, top_total: int) -> None:
    families_ordered = list(dict.fromkeys(top["display_family"]))
    family_bg = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(families_ordered)}

    seen: set[str] = set()
    family_first_idx: dict[str, int] = {}
    for idx, row in top.iterrows():
        if row["display_family"] not in seen:
            family_first_idx[row["display_family"]] = int(idx)
            seen.add(row["display_family"])

    prev_family = None
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        bg = family_bg[row["display_family"]]
        yb = row_bottom(fig_h, i)
        ym = row_mid(fig_h, i)

        for x0, x1, _, _ in COLS:
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x0, yb),
                    x1 - x0,
                    ROW_H,
                    boxstyle="square,pad=0",
                    linewidth=0.3,
                    edgecolor="#CCCCCC",
                    facecolor=bg,
                )
            )

        if (i - 1) == family_first_idx[row["display_family"]]:
            fpct = row["family_total"] / top_total * 100
            ax.text(
                COLS[0][0] + TEXT_PAD,
                ym + ROW_H * 0.13,
                row["display_family"],
                ha="left",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#1A252F",
            )
            ax.text(
                COLS[0][0] + TEXT_PAD,
                ym - ROW_H * 0.18,
                f"{row['family_total']:,.0f} words · {fpct:.1f}%",
                ha="left",
                va="center",
                fontsize=6.5,
                color="#555555",
            )

        ax.text(
            COLS[1][0] + TEXT_PAD,
            ym,
            row["name"],
            ha="left",
            va="center",
            fontsize=8,
            color="#1A252F",
        )
        ax.text(
            COLS[2][1] - TEXT_PAD,
            ym,
            f"{row['words']:,.0f}",
            ha="right",
            va="center",
            fontsize=8,
            color="#1A252F",
        )
        ax.text(
            COLS[3][1] - TEXT_PAD,
            ym,
            f"{row['pct_of_family']:.1f}%",
            ha="right",
            va="center",
            fontsize=8,
            color="#1A252F",
        )
        ax.text(
            COLS[4][1] - TEXT_PAD,
            ym,
            f"{row['pct_of_top_n']:.2f}%",
            ha="right",
            va="center",
            fontsize=8,
            color="#1A252F",
        )
        ax.text(
            COLS[5][1] - TEXT_PAD,
            ym,
            f"{row['pct_of_total']:.2f}%",
            ha="right",
            va="center",
            fontsize=8,
            color="#1A252F",
        )

        if prev_family is not None and row["display_family"] != prev_family:
            ax.plot(
                [COLS[0][0], COLS[-1][1]],
                [yb + ROW_H, yb + ROW_H],
                color="#555555",
                linewidth=0.9,
                zorder=5,
            )

        prev_family = row["display_family"]


def render_table(
    top: pd.DataFrame, top_total: int, train_total: int, n: int
) -> plt.Figure:
    fig, ax = make_figure(len(top))
    fig_h = fig.get_figheight()
    draw_title(ax, fig_h, n, top_total, train_total)
    draw_header(ax, fig_h)
    draw_rows(ax, fig_h, top, top_total)
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

train, train_total = load_training_data()
top, top_total = prepare_table_data(train, train_total, N, IE_SUBFAMILIES)

fig = render_table(top, top_total, train_total, N)

output = f"top{N}_languages_table{'_ie_subfamilies' if IE_SUBFAMILIES else ''}.png"
fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved {output}  ({len(top)} rows, {top['display_family'].nunique()} groups)")
