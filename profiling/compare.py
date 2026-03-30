"""Compare committed vs current ncu results. Called by run.sh."""
import sys, os, math, shutil
import pandas as pd

PROFILING_DIR = os.path.dirname(os.path.abspath(__file__))
COMMITTED_CSV = os.path.join(PROFILING_DIR, 'committed', 'results.csv')
CURRENT_CSV   = os.path.join(PROFILING_DIR, 'current',   'results.csv')

# ── Set to your GPU's peak memory bandwidth in GB/s to get BW% column.
# Examples: A100 SXM = 2000, H100 SXM = 3350, RTX 4090 = 1008, RTX 3090 = 936
PEAK_BW_GBS = 256

TIME_COL         = 'gpu__time_duration.sum'
# Use DRAM metrics for accurate bandwidth (industry standard)
DRAM_READ_COL    = 'dram__sectors_read.sum'
DRAM_WRITE_COL   = 'dram__sectors_write.sum'
DRAM_SECTOR_BYTES = 32  # DRAM sectors are 32 bytes
OCC_COL          = 'sm__warps_active.avg.pct_of_peak_sustained_active'
FLOP_COLS = [
    'sm__sass_thread_inst_executed_op_fadd_pred_on.sum',
    'sm__sass_thread_inst_executed_op_fmul_pred_on.sum',
    'sm__sass_thread_inst_executed_op_ffma_pred_on.sum',
]


def total_bytes(row, suffix=''):
    """DRAM sector traffic in bytes: sectors × 32 bytes/sector."""
    r = f'{DRAM_READ_COL}{suffix}'
    w = f'{DRAM_WRITE_COL}{suffix}'
    sectors = (row[r] if r in row.index else 0.0) + \
              (row[w] if w in row.index else 0.0)
    return sectors * DRAM_SECTOR_BYTES


def arithmetic_intensity(row, suffix=''):
    cols  = [f'{c}{suffix}' for c in FLOP_COLS]
    if not all(c in row.index for c in cols):
        return float('nan')
    flops  = sum(row[c] for c in cols[:2]) + 2 * row[cols[2]]  # ffma = 2 FLOP
    bytes_ = total_bytes(row, suffix)
    return flops / bytes_ if bytes_ else float('nan')


def achieved_bw_gbs(row, suffix=''):
    """Achieved DRAM bandwidth in GB/s (industry standard metric)."""
    t_us = row.get(f'{TIME_COL}{suffix}', float('nan'))
    if t_us == 0 or math.isnan(t_us):
        return float('nan')
    return total_bytes(row, suffix) / (t_us * 1e-6) / 1e9


def occupancy(row, suffix=''):
    col = f'{OCC_COL}{suffix}'
    return row[col] if col in row.index else float('nan')



def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[pd.to_numeric(df.get('ID', df.iloc[:, 0]), errors='coerce').notna()]
    df = df[['Kernel Name', 'Metric Name', 'Metric Value']].copy()
    df['Metric Value'] = pd.to_numeric(
        df['Metric Value'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df = df.pivot_table(index='Kernel Name', columns='Metric Name',
                        values='Metric Value', aggfunc='mean').reset_index()
    df.columns.name = None
    return df


def short_name(name):
    return name.split('<')[0].split('(')[0].strip()[:36]


def fmt_metrics(row, suffix):
    ai   = arithmetic_intensity(row, suffix)
    bw   = achieved_bw_gbs(row, suffix)
    occ  = occupancy(row, suffix)
    data = total_bytes(row, suffix) / 1e6

    data_str = f'{data:.1f}'   if data > 0           else 'n/a'
    ai_str   = f'{ai:.2f}'    if not math.isnan(ai)  else 'n/a'
    occ_str  = f'{occ:.0f}'   if not math.isnan(occ) else 'n/a'
    if PEAK_BW_GBS and not math.isnan(bw):
        bw_str = f'{100*bw/PEAK_BW_GBS:.0f}%'
    else:
        bw_str = f'{bw:.1f}'  if not math.isnan(bw)  else 'n/a'

    metrics = (f'  {row[f"{TIME_COL}{suffix}"]:>10.1f}'
               f'  {data_str:>9}  {ai_str:>8}'
               f'  {bw_str:>8}  {occ_str:>5}')
    return metrics


def print_table(rows, has_baseline):
    bw_hdr = 'BW%' if PEAK_BW_GBS else 'BW(GB/s)'
    hdr = (f'{"Kernel":<36}  {"":>8}  {"Time(us)":>10}'
           f'  {"Data(MB)":>9}  {"AI(F/B)":>8}'
           f'  {bw_hdr:>8}  {"Occ%":>5}  {"Speedup":>8}')
    print(f'\n{hdr}')
    print('-' * len(hdr))

    for _, row in rows:
        name = short_name(row['Kernel Name'])

        if has_baseline:
            after = fmt_metrics(row, '_n')
            t1 = row[f'{TIME_COL}_n']
            has_before = f'{TIME_COL}_c' in row.index and not pd.isna(row[f'{TIME_COL}_c'])
            if has_before:
                before = fmt_metrics(row, '_c')
                t0 = row[f'{TIME_COL}_c']
                sp_str = f'{t0/t1:.2f}x' if t1 else 'n/a'
                print(f'{name:<36}  {"Before":>8}{before}')
                print(f'{"":36}  {"After":>8}{after}  {sp_str:>8}')
            else:
                print(f'{name:<36}  {"(new)":>8}{after}  {"n/a":>8}')
        else:
            metrics = fmt_metrics(row, '')
            print(f'{name:<36}  {"":>8}{metrics}')

        print()
    print()


def main():
    if not os.path.exists(CURRENT_CSV):
        print('No current results. Run: sudo env PATH="$PATH" ./profiling/run.sh')
        sys.exit(1)

    current = load_csv(CURRENT_CSV)

    if not os.path.exists(COMMITTED_CSV):
        os.makedirs(os.path.dirname(COMMITTED_CSV), exist_ok=True)
        shutil.copy(CURRENT_CSV, COMMITTED_CSV)
        print('No committed baseline — saved current as first baseline.')
        print_table(current.sort_values(TIME_COL, ascending=False).iterrows(), has_baseline=False)
        return

    committed = load_csv(COMMITTED_CSV)
    merged = committed.merge(current, on='Kernel Name', suffixes=('_c', '_n'), how='right')
    merged = merged.sort_values(f'{TIME_COL}_n', ascending=False)
    print_table(merged.iterrows(), has_baseline=True)


if __name__ == '__main__':
    main()
