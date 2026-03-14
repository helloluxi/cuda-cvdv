"""Compare committed vs current ncu results. Called by run.sh."""
import sys, os, math, shutil
import pandas as pd

PROFILING_DIR = os.path.dirname(os.path.abspath(__file__))
COMMITTED_CSV = os.path.join(PROFILING_DIR, 'committed', 'results.csv')
CURRENT_CSV   = os.path.join(PROFILING_DIR, 'current',   'results.csv')

TIME_COL = 'gpu__time_duration.sum'


def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[pd.to_numeric(df.get('ID', df.iloc[:, 0]), errors='coerce').notna()]
    df = df[['Kernel Name', 'Metric Name', 'Metric Value']].copy()
    df['Metric Value'] = pd.to_numeric(
        df['Metric Value'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df = df.pivot_table(index='Kernel Name', columns='Metric Name',
                        values='Metric Value', aggfunc='sum').reset_index()
    df.columns.name = None
    return df


def short_name(name):
    return name.split('<')[0].split('(')[0].strip()[:36]


def main():
    if not os.path.exists(CURRENT_CSV):
        print('No current results. Run: sudo env PATH="$PATH" ./profiling/run.sh')
        sys.exit(1)

    current = load_csv(CURRENT_CSV)

    if not os.path.exists(COMMITTED_CSV):
        os.makedirs(os.path.dirname(COMMITTED_CSV), exist_ok=True)
        shutil.copy(CURRENT_CSV, COMMITTED_CSV)
        print('No committed baseline — saved current as first baseline.')
        return

    committed = load_csv(COMMITTED_CSV)
    merged = committed.merge(current, on='Kernel Name', suffixes=('_c', '_n'))

    print(f'\n{"Kernel":<36}  {"Before(us)":>10}  {"After(us)":>10}  {"Speedup":>8}')
    print('-' * 70)
    for _, row in merged.sort_values(f'{TIME_COL}_c', ascending=False).iterrows():
        t0 = row[f'{TIME_COL}_c']
        t1 = row[f'{TIME_COL}_n']
        sp = t0 / t1 if t1 else float('nan')
        print(f'{short_name(row["Kernel Name"]):<36}  {t0:>10.1f}  {t1:>10.1f}  {sp:>8.2f}x')
    print()


if __name__ == '__main__':
    main()
