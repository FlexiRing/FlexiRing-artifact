from __future__ import annotations

from pathlib import Path

from dtw_core import DTWConfig, build_templates, classify_by_templates, load_csv_sequence


def main():
    release_root = Path(__file__).resolve().parents[1]
    data_root = release_root / 'main_method' / 'example_data' / 'custom_gestures' / 'P16_23_male'

    support_paths = [
        data_root / '0' / 'support' / 'Alpha_003_0_0.csv',
        data_root / '1' / 'support' / 'Ampersand_007_1_1.csv',
        data_root / '2' / 'support' / 'Beta_001_2_2.csv',
        data_root / '3' / 'support' / 'Digit1_005_3_3.csv',
    ]
    support_labels = [0, 1, 2, 3]

    query_path = data_root / '2' / 'test' / 'Beta_003_2_2.csv'

    support_sequences = [load_csv_sequence(p) for p in support_paths]
    query_sequence = load_csv_sequence(query_path)

    config = DTWConfig(sample_hz=60, window_ms=50, stride_ms=30, quant_levels=16)
    templates = build_templates(support_sequences, support_labels, templates_per_class=1, config=config)
    pred_label, best_distance = classify_by_templates(query_sequence, templates, config=config)

    print(f'Predicted label: {pred_label}')
    print(f'Best DTW distance: {best_distance:.4f}')
    print(f'Query file: {query_path.name}')


if __name__ == '__main__':
    main()
