"""Create report figures from checked project validation evidence.

Mermaid source is retained in ../mermaid. Mermaid Ink renders only generic,
non-sensitive diagrams. Other visuals are local image assets and Matplotlib.
Run from any directory using the project Python environment.
"""
from pathlib import Path
import base64
import hashlib
import json
import shutil
import time
import urllib.request
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
FIG = HERE / 'figure'
MERMAID = HERE / 'mermaid'
SOURCE_FIG = ROOT / 'report' / 'figure'
EVIDENCE = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_mermaid(src, dest):
    state = {
        'code': src.read_text(encoding='utf-8'),
        'mermaid': {'theme': 'base', 'themeVariables': {
            'fontFamily': 'Arial, sans-serif', 'fontSize': '16px',
            'primaryColor': '#EAF2F8', 'primaryTextColor': '#17324D',
            'primaryBorderColor': '#6C91B4', 'lineColor': '#426B8E',
            'secondaryColor': '#F3F6F9', 'tertiaryColor': '#EDF4EE'}}}
    packed = zlib.compress(json.dumps(state).encode('utf-8'), level=9)
    payload = base64.urlsafe_b64encode(packed).decode('ascii').rstrip('=')
    url = f'https://mermaid.ink/img/pako:{payload}?bgColor=FFFFFF'
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 DentalSelfie-Progress-Report/1.0'})
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                blob = response.read()
            break
        except Exception as error:
            last_error = error
            time.sleep(attempt + 1)
    else:
        raise RuntimeError(f'Mermaid render failed after retries for {src.name}: {last_error}')
    if not blob.startswith(b'\xff\xd8'):
        raise RuntimeError(f'Mermaid renderer returned an unexpected image for {src.name}')
    dest.write_bytes(blob)
    EVIDENCE[src.relative_to(HERE).as_posix()] = digest(src)
    EVIDENCE[dest.relative_to(HERE).as_posix()] = digest(dest)


def copy_asset(source_name, dest_name):
    source = SOURCE_FIG / source_name
    if not source.is_file():
        raise FileNotFoundError(source)
    dest = FIG / dest_name
    shutil.copyfile(source, dest)
    EVIDENCE[source.relative_to(ROOT).as_posix()] = digest(source)
    EVIDENCE[dest.relative_to(HERE).as_posix()] = digest(dest)


def make_roi_chart():
    table = pd.read_csv(ROOT / 'docs/reports/synthesis/roi_experiment_matrix.csv')
    wanted = ['R0_old_CE', 'R1_raw_CE', 'R2_raw_CBCE', 'R3_raw_BalancedSoftmax',
              'P0_raw_frozen_ResNet18', 'P1_raw_frozen_DINOv3', 'CLS_plus_patch_mean',
              'last2_blocks_norm_CE']
    data = table.set_index('experiment').loc[wanted].copy()
    labels = ['R0 old\nCE', 'R1 raw\nCE', 'R2 raw\nCB-CE', 'R3 raw\nBalanced\nSoftmax',
              'Frozen\nResNet18', 'Frozen\nDINO CLS', 'DINO CLS +\npatch mean', 'DINO last-2\nblocks']
    colors = ['#8799A8']*4 + ['#607D9B']*2 + ['#397AA4', '#D08B43']
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(np.arange(len(data)), data.macro_f1*100, color=colors, width=.66)
    ax.axhline(57.40, color='#253746', linewidth=1, linestyle='--', alpha=.75)
    ax.set_ylim(0, 72); ax.set_ylabel('Validation macro-F1 (%)')
    ax.set_xticks(np.arange(len(data)), labels, fontsize=8)
    ax.set_title('Gingival ROI experiments: validation macro-F1')
    ax.grid(axis='y', alpha=.18)
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                f"{row.macro_f1*100:.1f}%\nNV {row.no_visible_recall*100:.0f}%",
                ha='center', va='bottom', fontsize=8)
    ax.text(.01, .98, 'Labels above bars: macro-F1 and no-visible recall', transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color='#44515c')
    fig.tight_layout()
    dest = FIG / '05_roi_results.png'
    fig.savefig(dest, dpi=190, bbox_inches='tight')
    plt.close(fig)
    matrix = ROOT / 'docs/reports/synthesis/roi_experiment_matrix.csv'
    EVIDENCE[matrix.relative_to(ROOT).as_posix()] = digest(matrix)
    EVIDENCE[dest.relative_to(HERE).as_posix()] = digest(dest)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for stem in ['01_research_pipeline', '02_data_split', '03_model_progression', '04_next_phase']:
        render_mermaid(MERMAID / f'{stem}.mmd', FIG / f'{stem}.png')
    copy_asset('technical_01_bbox_crop.png', '02_bbox_crop.png')
    copy_asset('technical_04_augmentation.png', '03_augmentation.png')
    copy_asset('technical_08_dino_predictions.png', '04_dino_prediction.png')
    copy_asset('technical_11_fresh_yolo.png', '06_caries_demo.png')
    make_roi_chart()
    (HERE / 'figure_manifest.json').write_text(json.dumps({
        'generated_at_note': 'Regenerate at report preparation; timestamps intentionally omitted.',
        'evidence_sha256': EVIDENCE,
        'mermaid_renderer': 'https://mermaid.ink',
        'test_images_opened': 0
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Created {len(list(FIG.glob("*.png")))} figures in {FIG}')


if __name__ == '__main__':
    main()
