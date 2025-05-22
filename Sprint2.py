# --- Hypothesis Testing: Does Danceability Affect Popularity? ---

# Research Question: Can specific audio features — like danceability — predict whether a song becomes popular?
# Null Hypothesis (H0): Danceability has no statistically significant effect on a song's popularity.
# Alternative Hypothesis (H1): Higher danceability positively influences a song's popularity.

# Pearson correlation between danceability and popularity
from scipy.stats import pearsonr
correlation, p_value = pearsonr(df['danceability'], df['popularity'])
print(f"Pearson correlation between danceability and popularity: {correlation:.2f}")
print(f"P-value: {p_value:.4f}")

# Interpretation:
# - The correlation is weak (-0.05), meaning danceability alone is not a strong predictor.
# - However, p-value = 0.0000 means the result is statistically significant.
# - Conclusion: Reject H0, but effect size is minimal. Danceability shouldn't be the only focus.
