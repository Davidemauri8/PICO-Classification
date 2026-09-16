"""Shared data loading and sampling utilities."""

import pandas as pd


def load_dataset(path, text_col="Text", label_col="Category"):
    """Load the corpus and check the expected schema."""
    df = pd.read_excel(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected column(s): {sorted(missing)}")
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    return df


def underSample2Min(df, labelName, random_state=None):
    """Undersample every label group down to the size of the minority group.

    The parameter labelName is the DataFrame column hosting the labels.
    Pass random_state to make the subset reproducible across runs and models.
    """
    vc = df.loc[:, labelName].value_counts()
    lab2freq = dict(zip(vc.index.tolist(), vc.values.tolist()))
    minfreq = min(lab2freq.values())

    idxSample = []
    for selectedLabel in lab2freq:
        selIndexes = (
            df.loc[df.loc[:, labelName] == selectedLabel, :]
            .sample(n=minfreq, random_state=random_state)
            .index.tolist()
        )
        idxSample += selIndexes
    idxSample.sort()

    return df.loc[idxSample, :].reset_index(drop=True)
