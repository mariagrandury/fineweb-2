# Language Distribution

Analysis scripts and visualizations for the FineWeb-2 language distribution.

## Scripts

- `analyze_languages.py`: Groups training languages by family, saves `families.csv`, and generates the two family-level pie charts.
- `topn_languages_table.py`: Renders a styled table of the top-N languages by word count, grouped by language family, with per-family and per-total percentages.
- `data_mixture.py`: Computes a greedy equal-share allocation of a 1.5T-word training budget across the top-N languages and families, and renders the result as a styled table.

## Data

- `families.csv`: Aggregated word counts and language counts per language family, output of `analyze_languages.py`.
- `ie_subfamilies.json`: Mapping of Indo-European subfamilies (e.g. Romance, Germanic) to their member languages, used to split the IE family into finer-grained groups.

## Graphs

- `families_words_distribution.png`: Pie chart of total training words broken down by language family (top 20 + Other).
- `families_languages_distribution.png`: Pie chart of the number of languages per family (top 20 + Other).
- `top80_languages_table.png`: Styled table of the top 80 languages by word count, grouped and color-coded by language family.
- `top100_languages_table.png`: Same as above for the top 100 languages.
- `top100_languages_table_ie_subfamilies.png`: Top 100 languages table with the Indo-European family further split into subfamilies (Romance, Germanic, etc.).
- `data_mixture.png`: Proposed training data mixture for 1.5T words across the top 100 languages, showing allocated vs. available words per language and family.
