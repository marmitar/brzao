import numpy as np
from numpy.typing import NDArray
import pandas as pd


def sort_negatives_after(data: NDArray[np.int_]) -> np.ndarray[np.int_]:
    negs = np.count_nonzero(data < 0)
    order = np.argsort(data)

    neg, pos = order[:negs], order[negs:]
    out = np.concat([pos, neg])
    return np.argsort(out)


def read_dataset(*, usecols: list[str] = ['year', 'position', 'points']) -> pd.DataFrame:
    df1 = pd.read_csv('dataset-2003-2018.csv', sep=';', usecols=usecols)
    df2 = pd.read_csv('dataset-2021-2023.csv', sep=';', usecols=usecols)
    return pd.concat([df1, df2])


def sorted_table(
    df: pd.DataFrame,
    /, *,
    key: str | list[str] = ['year', 'position'],
    ignore_index: bool = True,
) -> pd.DataFrame:
    df = df.sort_values(key, key=sort_negatives_after, ignore_index=ignore_index)
    if ignore_index:
        df.reset_index(inplace=True)

    return df


def gen_negatives(df: pd.DataFrame, /, *, limit: int = 5, key: str = 'year', order: str = 'position') -> pd.DataFrame:
    if limit <= 0:
        assert limit == 0, f"invalid limit: {limit}"
        return df

    dfs = [df]
    for _, table in df.groupby(key):
        selected = table.iloc[-limit:].copy()
        selected.sort_values(order)

        selected[order] = range(-limit, 0)
        dfs.append(selected)

    return sorted_table(pd.concat(dfs))


def starting_from(df: pd.DataFrame, /, *, start: int, key: str = 'year') -> pd.DataFrame:
    return sorted_table(df[df[key] >= start])


def position_points(df: pd.DataFrame, /, *, key: str = 'position', value: str = 'points') -> pd.DataFrame:
    df = df[[key, value]]

    out = df.groupby(key).aggregate(['min', 'mean', 'median', 'std', 'max'])
    return sorted_table(out, key=key, ignore_index=False)


def main() -> None:
    df = read_dataset()
    df = gen_negatives(df, limit=0)
    df = starting_from(df, start=2006)

    pp = position_points(df)
    print(pp.to_string())


if __name__ == '__main__':
    main()
