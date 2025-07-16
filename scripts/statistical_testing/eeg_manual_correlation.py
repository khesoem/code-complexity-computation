import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr

df = pd.read_csv("../../results/complexities_and_CLs.csv")
df = df.dropna(subset=["subject", "ID", "manual_cl", "eeg_cl_fzpz34"]).copy()

grouped = df.groupby('ID')[['eeg_cl_fzpz34', 'manual_cl']].mean().reset_index()
rho, p_val = spearmanr(grouped["eeg_cl_fzpz34"], grouped["manual_cl"])
print(f"Spearman ρ = {rho:.3f},  p = {p_val:.4g}")