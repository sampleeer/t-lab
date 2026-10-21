"""Post-test description only; never selects or fits models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts', default='artifacts/baseline')
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--reports', default='reports')
    args = p.parse_args()
    out, reports = Path(args.artifacts), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    # Require completed frozen-model test evaluation.
    json.loads((out / 'results.json').read_text())
    selection = json.loads((out / 'selection.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    cutoff = pd.Timestamp(manifest['snapshots'][-1]['cutoff'])
    events = pd.read_parquet(out / 'sample_events.parquet')
    history = events[events.last_watch_dt < cutoff]
    cb = pd.read_csv(out / 'test_users_catboost.csv').set_index('user_id')
    baseline = pd.read_csv(out / f"test_users_{selection['selected_control']}.csv").set_index('user_id')
    joined = cb[['ndcg@10', 'recall@10', 'candidate_recall']].rename(
        columns={'ndcg@10': 'catboost_ndcg', 'recall@10': 'catboost_recall'})
    joined['control_ndcg'] = baseline['ndcg@10']
    joined['control_recall'] = baseline['recall@10']
    joined['history_size'] = history.groupby('user_id').size().reindex(joined.index)
    joined['segment'] = pd.cut(joined.history_size, [0,1,5,20,np.inf],
                               labels=['1', '2–5', '6–20', '21+'])
    grouped = joined.groupby('segment', observed=True)
    segments = grouped[['catboost_ndcg','catboost_recall','control_ndcg',
                         'control_recall','candidate_recall']].mean()
    segments.insert(0, 'users', grouped.size())
    segments.to_csv(reports / 'history_segments.csv')
    predictions = pd.read_parquet(out / 'test_predictions.parquet')
    items = pd.read_csv(Path(args.raw) / 'items.csv').set_index('item_id')
    rng = np.random.default_rng(manifest['config']['seed'])
    examples = np.sort(rng.choice(np.sort(cb.index), min(3,len(cb)), replace=False))
    lines = ['# Примеры рекомендаций замороженной модели', '',
             f'Срез: {cutoff.date()}. Три случайных пользователя тестовой ranking-когорты, seed 42.',
             'Отбор примеров не зависит от качества их рекомендаций. Наличие целевой записи —',
             'offline-метка из KION, её отсутствие не означает, что фильм пользователю не нравится.', '']
    for uid in examples:
        past = history[history.user_id == uid].sort_values('last_watch_dt', ascending=False)
        titles = [str(items.loc[i,'title']) for i in past.item_id.head(5)]
        lines.extend([f'## Пользователь {uid}', '',
                      f'Строк в доступном прошлом: {len(past)}. Последние объекты: ' + '; '.join(titles) + '.', '',
                      '| Место | Фильм / сериал | Целевая запись на следующей неделе |',
                      '|---:|---|---|'])
        top = predictions[predictions.user_id == uid].sort_values(
            ['catboost','item_id'], ascending=[False,True]).head(10)
        for rank, row in enumerate(top.itertuples(), 1):
            title = str(items.loc[row.item_id,'title']).replace('|','/')
            lines.append(f"| {rank} | {title} | {'да' if row.label else 'нет'} |")
        lines.append('')
    (reports / '03_examples.md').write_text('\n'.join(lines), encoding='utf-8')
    print(segments.round(5).to_string())


if __name__ == '__main__':
    main()
