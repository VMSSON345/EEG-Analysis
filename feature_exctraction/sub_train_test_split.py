import pandas as pd
import numpy as np
import random
import os

def sync_split_from_two_files(
    csv_old_path,
    csv_new_path,
    output_dir,
    exclude_old,
    exclude_new,
    n_test=8,
    seed=42
):
    if not os.path.exists(csv_old_path) or not os.path.exists(csv_new_path):
        print("Do not find any CSV files.")
        return

    df_old = pd.read_csv(csv_old_path)
    df_new = pd.read_csv(csv_new_path)

    def normalize_ids(df):
        return df['subject_id'].astype(str).str.lower().str.replace('_', '-')

    df_old['norm_id'] = normalize_ids(df_old)
    df_new['norm_id'] = normalize_ids(df_new)

    subs_old = set(df_old['norm_id'].unique())
    subs_new = set(df_new['norm_id'].unique())

    bad_old = set(x.lower().replace('_', '-') for x in exclude_old)
    bad_new = set(x.lower().replace('_', '-') for x in exclude_new)

    common_subs = subs_old.intersection(subs_new)
    all_bad_subs = bad_old.union(bad_new)

    safe_pool = sorted(list(common_subs - all_bad_subs))

    print(f"Total subjects in old file: {len(subs_old)} (excluding {len(bad_old)})")
    print(f"Total subjects in new file: {len(subs_new)} (excluding {len(bad_new)})")
    print(f"Clean common subjects (Safe Pool): {len(safe_pool)}")

    random.seed(seed)
    fixed_test_subs = sorted(random.sample(safe_pool, n_test))

    print(f"8 test subjects: {fixed_test_subs}")

    def process_and_save(df, version, current_bad_set):
        mask_test = df['norm_id'].isin(fixed_test_subs)
        df_test = df[mask_test].drop(columns=['norm_id'])

        mask_bad = df['norm_id'].isin(current_bad_set)
        mask_train = (~mask_test) & (~mask_bad)
        df_train = df[mask_train].drop(columns=['norm_id'])

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        p_train = os.path.join(output_dir, f"{version}_TRAIN.csv")
        p_test = os.path.join(output_dir, f"{version}_TEST.csv")

        df_train.to_csv(p_train, index=False)
        df_test.to_csv(p_test, index=False)

        print(f"\n> {version}:")
        print(f"  Train: {df_train['subject_id'].nunique()} subjects | Shape: {df_train.shape}")
        print(f"  Test : {df_test['subject_id'].nunique()} subjects  | Shape: {df_test.shape}")

    process_and_save(df_old, "Old_0.5-30Hz", bad_old)
    process_and_save(df_new, "New_0.1-40Hz", bad_new)

    print(f"Finish: {output_dir}")
if __name__ == "__main__":
    FILE_OLD = './features/0.5-30/features_2.csv'
    FILE_NEW = './features/0.1-40/features.csv'
    sync_split_from_two_files(
        csv_old_path=FILE_OLD,
        csv_new_path=FILE_NEW,
        output_dir='./FINAL_SYNC_DATA',
        exclude_old=['sub-02', 'sub-07', 'sub-08', 'sub-10', 'sub-16', 'sub-23', 'sub-25', 'sub-35'],
        exclude_new= ['sub-25', 'sub-16', 'sub-23', 'sub-35'],
        n_test=8
    )