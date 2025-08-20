import pandas as pd

def compare_repeats_overlap_only(
    original_csv: str,
    repeat_csvs: list,
    key_col: str,
    label_col: str,
    out_all_csv: str = "agreements_allrows.csv",
    out_overlap_csv: str = "agreements_overlap_only.csv",
):

    merged = pd.read_csv(original_csv)

    def _norm(x):
        if pd.isna(x):
            return x
        return str(x).strip().lower()

    rep_label_cols = []
    for i, rep_csv in enumerate(repeat_csvs, start=1):
        rep = pd.read_csv(rep_csv)
        rep_collapsed = (
            rep.groupby(key_col)[label_col]
              .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
              .reset_index()
              .rename(columns={label_col: f"{label_col}_rep{i}"})
        )
        merged = merged.merge(rep_collapsed, on=key_col, how="left")
        rep_label_cols.append(f"{label_col}_rep{i}")

        merged[f"match_rep{i}"] = (
            merged[label_col].map(_norm) == merged[f"{label_col}_rep{i}"].map(_norm)
        )
        merged.loc[merged[f"{label_col}_rep{i}"].isna(), f"match_rep{i}"] = pd.NA

    merged.to_csv(out_all_csv, index=False)

    overlap_mask = merged[rep_label_cols].notna().any(axis=1)
    overlap = merged.loc[overlap_mask].copy()
    overlap.to_csv(out_overlap_csv, index=False)

    print("=== OVERLAP-ONLY SUMMARY (rows with at least one repeat) ===")
    print(f"Rows in overlap: {len(overlap):,}")

    for i in range(1, len(repeat_csvs) + 1):
        have_rep = overlap[f"{label_col}_rep{i}"].notna()
        total_with_rep = int(have_rep.sum())
        s = overlap.loc[have_rep, f"match_rep{i}"]
        matches = int(s.sum(skipna=True))
        mismatches = int((~s).sum(skipna=True))
        rate = (matches / total_with_rep * 100) if total_with_rep else 0.0

        print(f"\n--- Rep {i} ---")
        print(f"Docs with this rep: {total_with_rep:,}")
        print(f"Matches:            {matches:,}")
        print(f"Mismatches:         {mismatches:,}")
        print(f"Agreement rate:     {rate:.2f}%")

    total_pairs = total_matches = total_mismatches = 0
    for i in range(1, len(repeat_csvs) + 1):
        s = overlap[f"match_rep{i}"].dropna()
        total_pairs += len(s)
        total_matches += int(s.sum())
        total_mismatches += int((~s).sum())

    print("\n=== OVERALL (all reps combined, pair-level) ===")
    print(f"Total pairs compared: {total_pairs:,}")
    print(f"Total matches:        {total_matches:,}")
    print(f"Total mismatches:     {total_mismatches:,}")
    if total_pairs:
        print(f"Overall agreement:    {total_matches/total_pairs:.2%}")

    print(f"\nSaved (all rows):      {out_all_csv}")
    print(f"Saved (overlap only):  {out_overlap_csv}")

    return overlap

compare_repeats_overlap_only('/Users/jamesshortland/Desktop/masters_final_annotations/labels/complete_annotations.csv',
                             ['/Users/jamesshortland/Desktop/masters_final_annotations/labels/re-annotation_070511.csv',
                              '/Users/jamesshortland/Desktop/masters_final_annotations/labels/re-annotation_070512.csv',
                              '/Users/jamesshortland/Desktop/masters_final_annotations/labels/re-annotation_070513.csv',
                              '/Users/jamesshortland/Desktop/masters_final_annotations/labels/re-annotation_070514.csv',
                              '/Users/jamesshortland/Desktop/masters_final_annotations/labels/re-annotation_070515.csv',
                              '/Users/jamesshortland/Desktop/masters_final_annotations/labels/re-annotation_070516.csv'],
                             'filename', 'label')