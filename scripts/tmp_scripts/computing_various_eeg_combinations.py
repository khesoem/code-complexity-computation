import pandas as pd

eeg_df = pd.read_csv("../../results/backup_result_files/eegdata_all_combinations.csv")

eeg_df['eeg_cl_fzpz34']= eeg_df['Theta+AF8-Fz'] / eeg_df['Alpha+AF8-P3PzP4']
eeg_df['eeg_cl_fz34pz34']= eeg_df['Theta+AF8-F3F4Fz'] / eeg_df['Alpha+AF8-P3PzP4']
eeg_df['eeg_cl_fzpz']= eeg_df['Theta+AF8-Fz'] / eeg_df['Alpha+AF8-Pz']

complexities_df = pd.read_csv(
    "../../results/backup_result_files/revised_complexities_and_manual_cl_CC_LOC_are_wrong.csv")
complexities_df = complexities_df.rename(columns={"rating": "manual_cl"})

id_mapping_df = pd.read_csv("../../results/backup_result_files/program_id_to_snippet_id.csv")
id_mapping = dict(zip(id_mapping_df.iloc[:, 0], id_mapping_df.iloc[:, 1]))

# --- your custom key-building logic -----------------------------------
def make_key_from_complexities(row):
    """
    Build the key *for df1* using its first & second columns
    Replace this stub with your real rule.
    Example below:  take col 0, strip spaces, uppercase,
                    plus col 1 converted to int and zero-padded.
    """
    return f"{int(row.iloc[0])}_{id_mapping[int(row.iloc[1])]}"

def make_key_from_eeg(row):
    """
    Build the key *for df2* in whatever way produces the same
    string for the matching row.
    """
    participant_id = int(row['Participant'].strip().replace('P',''))
    participant_id = participant_id if participant_id < 12 else participant_id - 1
    return f"{participant_id}_{row['Snippet'].strip().replace('snippet','')}"

complexities_df["match_key"] = complexities_df.apply(make_key_from_complexities,  axis=1)
eeg_df["match_key"] = eeg_df.apply(make_key_from_eeg, axis=1)
df = complexities_df.merge(eeg_df, on="match_key", how="left", suffixes=("", "_y"))
df = df[complexities_df.columns.drop('match_key').tolist() + ["eeg_cl_fzpz34", "eeg_cl_fz34pz34", "eeg_cl_fzpz"]]

df.loc[df["ID"] == 6, "LOC"] = 11 # Correcting LOC for ID 6
df.loc[df["ID"] == 6, "Cyclomatic"] = 5 # Correcting LOC for ID 6

df.to_csv("../../results/complexities_and_CLs.csv", index=False)