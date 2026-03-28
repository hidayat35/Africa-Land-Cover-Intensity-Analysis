"""
Supplementary Material Generator
Addresses:
  - R2 Comment 6: Per-region interval, category, transition-level figures
  - R3 Comment 15: Temporal trajectory plots (gradual curve plots of LC area)

Outputs per region (EAF, MED, SAF, SAH, WAF):
  - S_L1_Interval_<REGION>.png + .csv
  - S_L2_Category_<REGION>.png + .csv
  - S_L3_Stationarity_<REGION>.png
  - S_L3_Heatmaps_<REGION>.png
  - S_L3_Transition_<REGION>.csv

Temporal trajectories (AFRICA + each region):
  - S_Trajectory_AFRICA.png
  - S_Trajectory_<REGION>.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import seaborn as sns

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
matrix_folder = r"D:\africa\5 regions\africa_5r_my_matrix"
output_folder = r"D:\AFRICA IA\review\supplementary_regions"
os.makedirs(output_folder, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

LC_CLASSES = {
    1: 'CRP', 2: 'FST', 3: 'SHR', 4: 'GRS', 5: 'TUD',
    6: 'WET', 7: 'IMP', 8: 'BAL', 9: 'WTR', 10: 'PSI'
}

LC_COLORS = {
    'CRP': '#E6B800', 'FST': '#228B22', 'SHR': '#808000', 'GRS': '#32CD32',
    'TUD': '#A9A9A9', 'WET': '#00CED1', 'IMP': '#DC143C', 'BAL': '#8B4513',
    'WTR': '#4169E1', 'PSI': '#1E90FF'
}

LC_FULLNAMES = {
    'CRP': 'Cropland', 'FST': 'Forest', 'SHR': 'Shrubland', 'GRS': 'Grassland',
    'TUD': 'Tundra', 'WET': 'Wetland', 'IMP': 'Impervious', 'BAL': 'Bare Area',
    'WTR': 'Water Body', 'PSI': 'Snow/Ice'
}

REGION_ORDER = ['EAF', 'MED', 'SAF', 'SAH', 'WAF']
REGION_FULLNAMES = {
    'EAF': 'East Africa', 'MED': 'Mediterranean',
    'SAF': 'Southern Africa', 'SAH': 'Sahara-Sahel', 'WAF': 'West Africa'
}

lc_ids = sorted(LC_CLASSES.keys())
YEARS = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2022]

def save_table(df, folder, name):
    if df.empty: return
    df.to_csv(os.path.join(folder, f"{name}.csv"), index=False)

# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------
csv_files = [f for f in os.listdir(matrix_folder) if f.lower().endswith(".csv")]
dfs = []
for fname in csv_files:
    df = pd.read_csv(os.path.join(matrix_folder, fname))
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    needed = ["region", "year_initial", "year_final", "interval_duration_yrs",
              "from_class", "to_class", "area_km2"]
    if all(c in df.columns for c in needed):
        dfs.append(df[needed])

all_df = pd.concat(dfs, ignore_index=True)

# Create AFRICA aggregate
africa_agg = all_df.groupby(['year_initial', 'year_final', 'interval_duration_yrs',
                             'from_class', 'to_class'])['area_km2'].sum().reset_index()
africa_agg['region'] = 'AFRICA'
all_df_with_africa = pd.concat([all_df, africa_agg], ignore_index=True)

# -------------------------------------------------------------------
# 2. MATRIX BUILDER
# -------------------------------------------------------------------

def build_matrices(df_site):
    intervals = df_site[["year_initial", "year_final", "interval_duration_yrs"]].drop_duplicates().sort_values("year_initial")
    matrices = {}
    for _, r in intervals.iterrows():
        yi, yf, dur = int(r["year_initial"]), int(r["year_final"]), float(r["interval_duration_yrs"])
        sub = df_site[(df_site["year_initial"] == yi) & (df_site["year_final"] == yf)]
        mat = sub.pivot_table(index="from_class", columns="to_class",
                              values="area_km2", aggfunc="sum", fill_value=0.0)
        mat = mat.reindex(index=lc_ids, columns=lc_ids, fill_value=0.0)
        matrices[(yi, yf)] = {"matrix": mat, "duration": dur}
    return matrices

# -------------------------------------------------------------------
# 3. INTERVAL LEVEL
# -------------------------------------------------------------------

def compute_interval_metrics(matrices):
    rows = []
    for (yi, yf), info in matrices.items():
        mat, dur = info["matrix"], info["duration"]
        total = mat.values.sum()
        change = total - np.trace(mat.values)
        intensity = (change / total) / dur * 100 if total > 0 else 0
        change_pct = (change / total) * 100 if total > 0 else 0
        rows.append({
            "interval": f"{yi}-{yf}", "year_initial": yi, "year_final": yf,
            "duration": dur, "total_area": total, "change_area": change,
            "intensity": intensity, "change_pct": change_pct
        })
    df = pd.DataFrame(rows).sort_values("year_initial", ascending=False)
    U = np.nan
    if not df.empty and df["duration"].sum() > 0:
        U = (df["change_area"].sum() / df["total_area"].mean()) / df["duration"].sum() * 100
    return df, U


def plot_interval_level(df_int, U, site_name):
    if df_int.empty: return
    df_int = df_int.sort_values('year_initial', ascending=True)
    intervals = df_int["interval"].tolist()
    y_pos = np.arange(len(intervals))

    fig, (ax_size, ax_speed) = plt.subplots(1, 2, figsize=(12, 7), sharey=True,
                                            gridspec_kw={'wspace': 0})

    ax_size.barh(y_pos, df_int["change_pct"], color='#d9d9d9', edgecolor='black', height=0.7)
    ax_size.invert_xaxis()
    ax_size.set_xlabel("Interval Change Area\n(% of map)", fontweight='bold', fontsize=11)
    ax_size.set_yticks(y_pos)
    ax_size.set_yticklabels(intervals, fontweight='bold', fontsize=11)
    ax_size.grid(axis='x', linestyle='--', alpha=0.5)

    ax_speed.barh(y_pos, df_int["intensity"], color='#696969', edgecolor='black', height=0.7)
    if not np.isnan(U):
        ax_speed.axvline(U, color='red', linestyle='--', linewidth=2)
        ax_speed.text(U, len(intervals) - 0.3, f'U = {U:.2f}%', ha='center', va='bottom',
                      fontsize=10, color='red', fontweight='bold')
    ax_speed.set_xlabel("Annual Change Intensity\n(% per year)", fontweight='bold', fontsize=11)
    ax_speed.grid(axis='x', linestyle='--', alpha=0.5)

    ax_size.spines['right'].set_linewidth(1.5)
    ax_speed.spines['left'].set_visible(False)

    full = REGION_FULLNAMES.get(site_name, site_name)
    plt.suptitle(f"Interval Level: {site_name} ({full})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"S_L1_Interval_{site_name}.png"), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# 4. CATEGORY LEVEL
# -------------------------------------------------------------------

def human_format_abs(x, pos):
    if x == 0: return ''
    x = abs(x)
    if x >= 1e6: return f'{x/1e6:.1f}M'
    if x >= 1e3: return f'{x/1e3:.0f}k'
    return f'{x:.0f}'


def analyze_category_level(matrices, site_name):
    classes = [LC_CLASSES[i] for i in lc_ids]
    cat_data = []

    for (yi, yf), info in matrices.items():
        mat, dur = info["matrix"], info["duration"]
        if dur <= 0: continue
        row_sums = mat.sum(axis=1)
        col_sums = mat.sum(axis=0)
        diag = pd.Series(np.diag(mat.values), index=mat.index)
        gross_gain = col_sums - diag
        gross_loss = row_sums - diag
        total_change = mat.values.sum() - diag.sum()
        S_t = (total_change / mat.values.sum()) / dur * 100 if mat.values.sum() > 0 else 0

        for c in lc_ids:
            G_int = (gross_gain[c] / dur) / col_sums[c] * 100 if col_sums[c] > 0 else 0
            L_int = (gross_loss[c] / dur) / row_sums[c] * 100 if row_sums[c] > 0 else 0
            cat_data.append({
                'interval': f"{yi}-{yf}", 'year_initial': yi, 'year_final': yf,
                'class_id': c, 'class_name': LC_CLASSES[c],
                'gain_annual_km2_yr': round(gross_gain[c] / dur, 2),
                'loss_annual_km2_yr': round(gross_loss[c] / dur, 2),
                'gain_intensity_pct_yr': round(G_int, 4),
                'loss_intensity_pct_yr': round(L_int, 4),
                'uniform_intensity': round(S_t, 4)
            })

    df_cat = pd.DataFrame(cat_data).sort_values(['year_initial', 'class_id'])
    if df_cat.empty: return
    save_table(df_cat, output_folder, f"S_L2_Category_{site_name}")

    intervals = df_cat['interval'].unique()
    n_int = len(intervals)
    fig_h = n_int * 4.0 + 2.0

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, fig_h))
    gs = GridSpec(n_int, 2, figure=fig, wspace=0, hspace=0.45, width_ratios=[1, 1])

    y = np.arange(len(classes))
    bh = 0.40

    for i, interval in enumerate(intervals):
        data = df_cat[df_cat['interval'] == interval].set_index('class_name').reindex(classes)
        cols = [LC_COLORS[c] for c in classes]
        U = data['uniform_intensity'].iloc[0]

        ax_area = fig.add_subplot(gs[i, 0])
        ax_int = fig.add_subplot(gs[i, 1])

        ax_area.barh(y + bh/2, data['gain_annual_km2_yr'], height=bh,
                     color=cols, edgecolor='black', lw=0.5)
        ax_area.barh(y - bh/2, data['loss_annual_km2_yr'], height=bh,
                     color='white', edgecolor=cols, hatch='////', lw=0.5)
        ax_area.invert_xaxis()
        max_area = max(data['gain_annual_km2_yr'].max(), data['loss_annual_km2_yr'].max()) * 1.15
        if max_area > 0:
            ax_area.set_xlim(max_area, 0)

        ax_int.barh(y + bh/2, data['gain_intensity_pct_yr'], height=bh,
                    color=cols, edgecolor='black', lw=0.5)
        ax_int.barh(y - bh/2, data['loss_intensity_pct_yr'], height=bh,
                    color='white', edgecolor=cols, hatch='////', lw=0.5)
        ax_int.axvline(U, color='red', linestyle='--', lw=1.5, zorder=10)

        max_int = max(data['gain_intensity_pct_yr'].max(), data['loss_intensity_pct_yr'].max(), U) * 1.2
        if max_int > 0:
            ax_int.set_xlim(0, max_int)

        y_min, y_max = -0.5, len(classes) - 0.5
        ax_area.set_ylim(y_min, y_max)
        ax_int.set_ylim(y_min, y_max)

        ax_int.text(U, y_max + 0.3, f'U={U:.2f}', ha='center', va='bottom',
                    fontsize=9, color='red', fontweight='bold', clip_on=False)

        ax_area.spines['right'].set_linewidth(2)
        ax_int.spines['left'].set_visible(False)
        ax_area.spines['top'].set_visible(False)
        ax_int.spines['top'].set_visible(False)

        ax_area.set_yticks(y)
        ax_area.set_yticklabels(classes, fontsize=10, fontweight='bold')
        ax_area.set_ylabel(interval, fontweight='bold', fontsize=11)
        ax_int.set_yticks([])

        ax_area.xaxis.set_major_formatter(ticker.FuncFormatter(human_format_abs))
        ax_area.grid(axis='x', linestyle='--', alpha=0.3)
        ax_int.grid(axis='x', linestyle='--', alpha=0.3)

    full = REGION_FULLNAMES.get(site_name, site_name)
    fig.suptitle(f"{site_name} ({full}) - Category Level", fontsize=14, fontweight='bold', y=0.99)

    legend_items = [
        patches.Patch(facecolor='gray', edgecolor='black', label='Gain'),
        patches.Patch(facecolor='white', edgecolor='gray', hatch='////', label='Loss'),
        mlines.Line2D([0], [0], color='red', linestyle='--', label='Uniform')
    ]
    fig.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, 0.01),
              ncol=3, frameon=False, fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95)
    plt.savefig(os.path.join(output_folder, f"S_L2_Category_{site_name}.png"), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# 5. TRANSITION LEVEL
# -------------------------------------------------------------------

def analyze_transition_level(matrices, site_name):
    classes = [LC_CLASSES[i] for i in lc_ids]
    N = len(classes)
    results = []
    intervals_list = []

    for (yi, yf), info in matrices.items():
        interval = f"{yi}-{yf}"
        if interval not in intervals_list: intervals_list.append(interval)
        mat, dur = info["matrix"], info["duration"]
        if dur <= 0: continue

        initial = mat.sum(axis=1)
        for n in lc_ids:
            gross_gain_n = mat[n].sum() - mat.at[n, n]
            other_area = initial.sum() - initial[n]
            W_tn = (gross_gain_n / dur) / other_area * 100 if other_area > 0 else 0
            for m in lc_ids:
                if m == n: continue
                trans_area = mat.at[m, n]
                R_tin = (trans_area / dur) / initial[m] * 100 if initial[m] > 0 else 0
                results.append({
                    'interval': interval, 'from_class': LC_CLASSES[m], 'to_class': LC_CLASSES[n],
                    'from_id': m, 'to_id': n,
                    'area_km2': round(trans_area, 2),
                    'R_tin': round(R_tin, 4), 'W_tn': round(W_tn, 4),
                    'is_targeted': R_tin > W_tn,
                })

    df_trans = pd.DataFrame(results)
    if df_trans.empty: return
    save_table(df_trans, output_folder, f"S_L3_Transition_{site_name}")

    # --- Stationarity Matrix ---
    n_int = len(intervals_list)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)

    for r_idx, from_cls in enumerate(classes):
        for c_idx, to_cls in enumerate(classes):
            y_base = N - 1 - r_idx
            x_base = c_idx

            if from_cls == to_cls:
                ax.add_patch(patches.Rectangle((x_base, y_base), 1, 1,
                            facecolor='#dddddd', hatch='///'))
                continue

            strip_w = 1.0 / n_int
            cell = df_trans[(df_trans['from_class'] == from_cls) & (df_trans['to_class'] == to_cls)]
            info_map = {}
            for _, r in cell.iterrows():
                info_map[r['interval']] = r['is_targeted']

            for k, intv in enumerate(intervals_list):
                targeted = info_map.get(intv, False)
                rect_x = x_base + k * strip_w
                color = "#d62728" if targeted else "#f0f0f0"
                ax.add_patch(patches.Rectangle((rect_x, y_base), strip_w, 1,
                            facecolor=color, edgecolor='none', lw=0))
                if k < n_int - 1:
                    ax.plot([rect_x + strip_w, rect_x + strip_w], [y_base, y_base + 1],
                           color='black', lw=0.3, zorder=10)

            ax.add_patch(patches.Rectangle((x_base, y_base), 1, 1,
                        fill=False, edgecolor='black', lw=2.0))

    for i in range(N + 1):
        ax.axhline(i, color='black', lw=2.0)
        ax.axvline(i, color='black', lw=2.0)

    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_xticklabels(classes, rotation=90, fontsize=12, fontweight='bold')
    ax.set_yticklabels(classes[::-1], fontsize=12, fontweight='bold')
    ax.set_xlabel("To Category (Gaining)", fontsize=14, fontweight='bold')
    ax.set_ylabel("From Category (Losing)", fontsize=14, fontweight='bold')

    full = REGION_FULLNAMES.get(site_name, site_name)
    ax.set_title(f"{site_name} ({full}): Transition Stationarity\n"
                 f"(Strips: {intervals_list[0]} → {intervals_list[-1]})",
                 fontsize=14, fontweight='bold')

    legend = [
        patches.Patch(facecolor='#d62728', edgecolor='black', label='Targeted'),
        patches.Patch(facecolor='#f0f0f0', edgecolor='black', label='Avoided/Random'),
        patches.Patch(facecolor='#dddddd', hatch='///', edgecolor='black', label='Persistence')
    ]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.08),
             ncol=3, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_folder, f"S_L3_Stationarity_{site_name}.png"), bbox_inches='tight')
    plt.close()

    # --- Heatmaps ---
    intervals_keys = list(matrices.keys())
    n = len(intervals_keys)
    if n == 0: return
    cols_h = 2
    rows_h = math.ceil(n / cols_h)
    fig, axes = plt.subplots(rows_h, cols_h, figsize=(11*cols_h, 9*rows_h), constrained_layout=True)
    axes = np.array(axes).flatten() if n > 1 else [axes]
    labels = [LC_CLASSES[i] for i in lc_ids]

    for i, (yi, yf) in enumerate(intervals_keys):
        ax = axes[i]
        mat = matrices[(yi, yf)]["matrix"]
        pct = mat.div(mat.sum(axis=1), axis=0).fillna(0) * 100
        annot = pd.DataFrame(index=mat.index, columns=mat.columns)
        for r in mat.index:
            for c in mat.columns:
                annot.at[r, c] = f"{pct.at[r,c]:.1f}%\n({mat.at[r,c]:.0f})"

        sns.heatmap(pct, ax=ax, annot=annot, fmt="", cmap="OrRd", vmin=0, vmax=100,
                   cbar=False, xticklabels=labels, yticklabels=labels,
                   annot_kws={"size": 9, "weight": "bold"}, linewidths=1, linecolor='gray')
        ax.set_title(f"{yi}-{yf}", fontsize=14, fontweight='bold')
        ax.set_ylabel("From", fontsize=12, fontweight='bold')
        ax.set_xlabel("To", fontsize=12, fontweight='bold')

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    norm = plt.Normalize(0, 100)
    sm = plt.cm.ScalarMappable(cmap="OrRd", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.02, aspect=50,
                label='Row Fraction (%)')

    full = REGION_FULLNAMES.get(site_name, site_name)
    fig.suptitle(f"{site_name} ({full}): Transition Matrices", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_folder, f"S_L3_Heatmaps_{site_name}.png"), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# 6. TEMPORAL TRAJECTORY PLOTS (R3 Comment 15)
# -------------------------------------------------------------------

def compute_class_areas_at_years(df_site):
    """
    Compute the area of each LC class at each time point.
    Uses row sums (initial area) for all years except the last,
    and column sums (final area) for the last year of each interval.
    """
    intervals = [(YEARS[i], YEARS[i+1]) for i in range(len(YEARS)-1)]
    year_areas = {yr: {c: 0.0 for c in lc_ids} for yr in YEARS}

    for yi, yf in intervals:
        sub = df_site[(df_site['year_initial'] == yi) & (df_site['year_final'] == yf)]
        if sub.empty:
            continue
        for c in lc_ids:
            # Initial area (row sum)
            year_areas[yi][c] = sub[sub['from_class'] == c]['area_km2'].sum()
            # Final area (column sum) — overwritten for intermediate years, correct for last
            year_areas[yf][c] = sub[sub['to_class'] == c]['area_km2'].sum()

    return year_areas


def plot_trajectory_single(df_site, site_name):
    """Single plot: all classes on one figure."""
    year_areas = compute_class_areas_at_years(df_site)

    fig, ax = plt.subplots(figsize=(13, 7))

    for c in lc_ids:
        cls = LC_CLASSES[c]
        areas = [year_areas[yr].get(c, 0) for yr in YEARS]
        # Skip classes with negligible area (< 0.1% of total)
        total = sum(year_areas[YEARS[0]].values())
        if max(areas) < total * 0.001:
            continue
        ax.plot(YEARS, [a / 1000 for a in areas], '-o', color=LC_COLORS[cls],
                linewidth=2.5, markersize=6, label=f"{cls} ({LC_FULLNAMES[cls]})",
                zorder=5)

    ax.set_xlabel("Year", fontsize=13, fontweight='bold')
    ax.set_ylabel("Area (×10³ km²)", fontsize=13, fontweight='bold')

    full = REGION_FULLNAMES.get(site_name, site_name)
    ax.set_title(f"{site_name} ({full}): LC Class Area Trajectories (1985–2022)",
                 fontsize=15, fontweight='bold')

    ax.legend(ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.10),
             frameon=True, edgecolor='gray', fancybox=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"S_Trajectory_{site_name}.png"), bbox_inches='tight')
    plt.close()


def plot_trajectory_subplots(df_site, site_name):
    """
    Two-panel version: large classes (top) and small classes (bottom)
    for better visibility when class areas differ by orders of magnitude.
    """
    year_areas = compute_class_areas_at_years(df_site)
    total = sum(year_areas[YEARS[0]].values())

    # Split into large and small classes by mean area
    large_classes = []
    small_classes = []
    for c in lc_ids:
        cls = LC_CLASSES[c]
        mean_area = np.mean([year_areas[yr].get(c, 0) for yr in YEARS])
        if mean_area < total * 0.001:
            continue  # skip negligible
        if mean_area > total * 0.02:
            large_classes.append(c)
        else:
            small_classes.append(c)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [1, 1]})

    # Top: large classes
    for c in large_classes:
        cls = LC_CLASSES[c]
        areas = [year_areas[yr].get(c, 0) / 1000 for yr in YEARS]
        ax1.plot(YEARS, areas, '-o', color=LC_COLORS[cls], linewidth=2.5, markersize=6,
                label=f"{cls} ({LC_FULLNAMES[cls]})")

    ax1.set_ylabel("Area (×10³ km²)", fontsize=12, fontweight='bold')
    ax1.set_title("Major Classes", fontsize=12, fontweight='bold')
    ax1.legend(ncol=3, fontsize=9, loc='best', frameon=True, edgecolor='gray')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xticks(YEARS)
    ax1.set_xticklabels(YEARS, fontsize=10, fontweight='bold')

    # Bottom: small classes
    for c in small_classes:
        cls = LC_CLASSES[c]
        areas = [year_areas[yr].get(c, 0) / 1000 for yr in YEARS]
        ax2.plot(YEARS, areas, '-o', color=LC_COLORS[cls], linewidth=2.5, markersize=6,
                label=f"{cls} ({LC_FULLNAMES[cls]})")

    ax2.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Area (×10³ km²)", fontsize=12, fontweight='bold')
    ax2.set_title("Minor Classes", fontsize=12, fontweight='bold')
    ax2.legend(ncol=3, fontsize=9, loc='best', frameon=True, edgecolor='gray')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks(YEARS)
    ax2.set_xticklabels(YEARS, fontsize=10, fontweight='bold')

    full = REGION_FULLNAMES.get(site_name, site_name)
    fig.suptitle(f"{site_name} ({full}): LC Class Area Trajectories (1985–2022)",
                 fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"S_Trajectory_split_{site_name}.png"), bbox_inches='tight')
    plt.close()


def plot_trajectory_all_regions_combined():
    """
    Single summary figure: one subplot per region + AFRICA, showing
    the top 4-5 classes that dominate dynamics.
    """
    sites = ['AFRICA'] + REGION_ORDER
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    # Top 5 most dynamic classes (by total gross change across AFRICA)
    highlight_classes = [2, 3, 4, 1, 8]  # FST, SHR, GRS, CRP, BAL

    for idx, site_name in enumerate(sites):
        ax = axes[idx]

        if site_name == 'AFRICA':
            df_site = all_df_with_africa[all_df_with_africa['region'] == 'AFRICA']
        else:
            df_site = all_df[all_df['region'] == site_name]

        year_areas = compute_class_areas_at_years(df_site)

        for c in highlight_classes:
            cls = LC_CLASSES[c]
            areas = [year_areas[yr].get(c, 0) / 1000 for yr in YEARS]
            if max(areas) < 1:  # skip if < 1k km²
                continue
            ax.plot(YEARS, areas, '-o', color=LC_COLORS[cls], linewidth=2, markersize=4,
                    label=cls)

        full = REGION_FULLNAMES.get(site_name, site_name)
        ax.set_title(f"{site_name} ({full})", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y)[-2:] for y in YEARS], fontsize=8)
        ax.tick_params(axis='y', labelsize=9)

        if idx >= 3:
            ax.set_xlabel("Year", fontsize=10, fontweight='bold')
        if idx % 3 == 0:
            ax.set_ylabel("Area (×10³ km²)", fontsize=10, fontweight='bold')

        ax.legend(fontsize=8, loc='best', frameon=True)

    # Hide extra subplot if any
    for j in range(len(sites), len(axes)):
        axes[j].axis('off')

    fig.suptitle("LC Class Area Trajectories by Region (1985–2022)\nTop 5 dynamic classes",
                 fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_folder, "S_Trajectory_AllRegions_Combined.png"), bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------
# 7. TRAJECTORY CSV EXPORT
# -------------------------------------------------------------------

def export_trajectory_csv(df_site, site_name):
    """Export class areas at each year as a CSV."""
    year_areas = compute_class_areas_at_years(df_site)
    rows = []
    for yr in YEARS:
        for c in lc_ids:
            rows.append({
                'region': site_name,
                'year': yr,
                'class_id': c,
                'class_name': LC_CLASSES[c],
                'area_km2': round(year_areas[yr].get(c, 0), 2)
            })
    df = pd.DataFrame(rows)
    save_table(df, output_folder, f"S_Trajectory_{site_name}")


# -------------------------------------------------------------------
# 8. MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SUPPLEMENTARY MATERIAL GENERATOR")
    print("Per-region analysis (R2C6) + Trajectory plots (R3C15)")
    print("=" * 60)

    # ==========================================
    # PART A: Per-region IA figures (R2 Comment 6)
    # ==========================================
    print("\n" + "=" * 40)
    print("PART A: Per-Region Intensity Analysis")
    print("=" * 40)

    for region_name in REGION_ORDER:
        print(f"\n--- {region_name} ({REGION_FULLNAMES[region_name]}) ---")

        df_reg = all_df[all_df['region'] == region_name]
        if df_reg.empty:
            print(f"  No data, skipping.")
            continue

        matrices = build_matrices(df_reg)

        # L1
        df_int, U = compute_interval_metrics(matrices)
        save_table(df_int, output_folder, f"S_L1_Interval_{region_name}")
        plot_interval_level(df_int, U, region_name)
        print(f"  L1 Interval done (U = {U:.2f}%)")

        # L2
        analyze_category_level(matrices, region_name)
        print(f"  L2 Category done")

        # L3
        analyze_transition_level(matrices, region_name)
        print(f"  L3 Transition done")

    # ==========================================
    # PART B: Trajectory plots (R3 Comment 15)
    # ==========================================
    print("\n" + "=" * 40)
    print("PART B: Temporal Trajectory Plots")
    print("=" * 40)

    # AFRICA trajectories
    print("\n--- AFRICA ---")
    df_africa = all_df_with_africa[all_df_with_africa['region'] == 'AFRICA']
    plot_trajectory_single(df_africa, 'AFRICA')
    plot_trajectory_subplots(df_africa, 'AFRICA')
    export_trajectory_csv(df_africa, 'AFRICA')
    print("  Single + Split + CSV done")

    # Per-region trajectories
    for region_name in REGION_ORDER:
        print(f"\n--- {region_name} ---")
        df_reg = all_df[all_df['region'] == region_name]
        if df_reg.empty:
            print(f"  No data, skipping.")
            continue
        plot_trajectory_single(df_reg, region_name)
        plot_trajectory_subplots(df_reg, region_name)
        export_trajectory_csv(df_reg, region_name)
        print("  Single + Split + CSV done")

    # Combined multi-region summary
    print("\n--- Combined multi-region summary ---")
    plot_trajectory_all_regions_combined()
    print("  Combined figure done")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    print("\nPer region (×5):")
    print("  S_L1_Interval_<REGION>.png + .csv")
    print("  S_L2_Category_<REGION>.png + .csv")
    print("  S_L3_Stationarity_<REGION>.png")
    print("  S_L3_Heatmaps_<REGION>.png")
    print("  S_L3_Transition_<REGION>.csv")
    print("\nTrajectory plots:")
    print("  S_Trajectory_<SITE>.png          (all classes, single panel)")
    print("  S_Trajectory_split_<SITE>.png    (major/minor split panels)")
    print("  S_Trajectory_<SITE>.csv          (raw area data)")
    print("  S_Trajectory_AllRegions_Combined.png  (6-panel summary)")
    print("=" * 60)
