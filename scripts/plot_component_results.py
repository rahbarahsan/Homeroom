"""Rebuild the component-teaching figure from the committed aggregate only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path


def chart(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (stratum, label, color) in enumerate((
        ('multiple', 'Competing rules (112)', '#3182ce'),
        ('all', 'All selection (476)', '#38a169'),
    )):
        xs, ys, labels = [], [], []
        for j, arm in enumerate(('J', 'F')):
            metric = result['scores'][arm]['selection'][stratum]
            if metric['vector_accuracy'] is not None:
                xs.append(j + (i - .5) * .28)
                ys.append(100 * metric['vector_accuracy'])
                labels.append(f"{metric['vector_correct']}/{metric['complete_cases']}")
        bars = axes[0].bar(xs, ys, .27, label=label, color=color)
        axes[0].bar_label(bars, labels, padding=3, fontsize=9)
    axes[0].set_xticks([0, 1], ['J: joint', 'F: independent'])
    axes[0].set_ylim(0, 114)
    axes[0].set_ylabel('All five predicates correct (%)')
    axes[0].legend(loc='upper left', fontsize=8)
    minutes = [result['resources']['segment_seconds'][arm] / 60 for arm in ('J', 'F')]
    bars = axes[1].bar([0, 1], minutes, color=['#718096', '#d69e2e'])
    axes[1].bar_label(bars, [f'{v:.2f} min' for v in minutes], padding=3)
    axes[1].set_xticks([0, 1], ['J: 1 call/case', 'F: 5 calls/case'])
    axes[1].set_ylabel('Resident minutes: load + train + infer + recovery')
    axes[1].set_ylim(0, max(minutes + [1]) * 1.2)
    fig.suptitle('S2g component teaching: joint versus independent judgments\n'
                 'Same 48 cases / 240 labels; routing uses an external priority resolver', fontsize=10)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=160, metadata={'Software': 'Homeroom S2g'})
    plt.close(fig)
    return buffer.getvalue()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path,
                        default=root / 'results/teaching_policy_v1/s2g_6dea579a0622.json')
    parser.add_argument('--output', type=Path, default=root / 'data/generated/component_results.png')
    args = parser.parse_args()
    raw = args.aggregate.read_bytes()
    result = json.loads(raw)
    if result.get('stage') != 'S2g' or result.get('status') != 'complete':
        raise ValueError('a completed S2g aggregate is required')
    content = chart(result)
    if args.output.resolve() == args.aggregate.resolve():
        raise ValueError('output must differ from input')
    if args.output.exists() and args.output.read_bytes() != content:
        raise ValueError('output already exists with different content; choose another path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_bytes(content)
    print(json.dumps({'aggregate_sha256': hashlib.sha256(raw).hexdigest(),
                      'png_sha256': hashlib.sha256(content).hexdigest(), 'new_model_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
