import json
import shutil
from collections import Counter
from pathlib import Path
import pandas as pd


def load(path):return json.loads(path.read_text())
def num(x):return f'{x:.5f}'.replace('.',',')
def ci(r):return f"{num(r['mean'])} [{num(r['low_95'])}; {num(r['high_95'])}]"


def main():
    out=Path('artifacts/robustness');destination=Path('reports/robust')
    destination.mkdir(parents=True,exist_ok=True)
    manifest=load(out/'manifest.json');selection=load(out/'selection.json');results=load(out/'results.json')
    for name in ['manifest.json','selection.json','results.json','all_results.csv','weekly_results.csv',
                 'paired_intervals.csv','summary.csv']:
        shutil.copyfile(out/name,destination/name)
    weekly=pd.read_csv(out/'weekly_results.csv')
    all_rows=pd.read_csv(out/'all_results.csv')
    intervals=pd.read_csv(out/'paired_intervals.csv')
    means=weekly.groupby('method')[['ndcg@10','recall@10']].mean()
    labels={'manual':'Ручной CatBoost','auto':'Автоматическое дополнение',
            'random':'Случайное дополнение','replacement':'Замена ручных агрегатов',
            'recent_popularity':'Популярность за 7 дней','popularity':'Популярность за всё прошлое'}
    order=['manual','auto','random','replacement','recent_popularity','popularity']
    aggregate='\n'.join(f"| {labels[n]} | {num(means.loc[n,'ndcg@10'])} | {num(means.loc[n,'recall@10'])} |" for n in order)
    grid=[];fold_lines=[];families=Counter();feature_freq=Counter();feature_details=[]
    for i,f in enumerate(selection['folds']):
        fd=destination/f'fold_{i}';fd.mkdir(exist_ok=True)
        for name in ['selection.json','screening.csv','screening.json']:
            shutil.copyfile(out/f'fold_{i}'/name,fd/name)
        fold_lines.append(f"| {f['dates']['test']} | {f['dates']['validation']} | {f['params']['depth']} | {f['params']['iterations']} | {f['auto_best']['k']} | {f['screening_groups']} |")
        for row in f['manual_grid']:
            grid.append({'fold':i,'validation':f['dates']['validation'],'depth':row['depth'],
                         'trees':row['trees'],**row['metrics']})
        for name in f['auto_best']['extra']:
            families['связи user–item' if name.split('_')[2]=='aff' else
                     ('объект' if name.split('_')[2]=='i' else 'пользователь')]+=1
            feature_freq[name]+=1
            feature_details.append({'fold':i,'test_cutoff':f['dates']['test'],'feature':name})
    pd.DataFrame(grid).to_csv(destination/'validation_grid.csv',index=False)
    pd.DataFrame(feature_details).to_csv(destination/'selected_features.csv',index=False)
    mean_rows=[];differences=[];seed_ranges=[]
    for i,f in enumerate(selection['folds']):
        row=weekly[weekly.fold==i].set_index('method')
        raw=all_rows[all_rows.fold==i]
        for name in order:
            part=raw[raw.method==name]
            mean_rows.append(f"| {f['dates']['test']} | {labels[name]} | {num(row.loc[name,'ndcg@10'])} | {num(row.loc[name,'recall@10'])} | {num(part['ndcg@10'].min())}–{num(part['ndcg@10'].max())} |")
            seed_ranges.append({'fold':i,'method':name,'ndcg_min':part['ndcg@10'].min(),
                                'ndcg_max':part['ndcg@10'].max()})
        deltas=[]
        for seed in manifest['config']['model_seeds']:
            sr=raw[raw.seed==seed].set_index('method')
            deltas.append(num(sr.loc['auto','ndcg@10']-sr.loc['manual','ndcg@10']))
        pair=intervals[(intervals.fold==i)&(intervals.comparison=='auto_minus_manual')].set_index('metric')
        differences.append(f"| {f['dates']['test']} | {deltas[0]} | {deltas[1]} | {ci(pair.loc['ndcg@10'])} | {ci(pair.loc['recall@10'])} |")
    pd.DataFrame(seed_ranges).to_csv(destination/'seed_ranges.csv',index=False)
    paired=intervals[(intervals.comparison=='auto_minus_manual')&(intervals.metric=='ndcg@10')]
    positive=int((paired.low_95>0).sum());negative=int((paired.high_95<0).sum())
    uncertain=len(paired)-positive-negative
    mask=intervals.metric=='ndcg@10'
    manual_wins=int((intervals.loc[mask&(intervals.comparison=='manual_minus_recent_popularity'),'low_95']>0).sum())
    replacement_losses=int((intervals.loc[mask&(intervals.comparison=='replacement_minus_manual'),'high_95']<0).sum())
    random_wins=int((intervals.loc[mask&(intervals.comparison=='auto_minus_random'),'low_95']>0).sum())
    controls=[]
    for comparison,label in [('manual_minus_recent_popularity','Ручной − недавняя популярность'),
                              ('auto_minus_random','Автоотбор − случайные признаки'),
                              ('replacement_minus_manual','Замена − ручной'),
                              ('replacement_minus_recent_popularity','Замена − недавняя популярность')]:
        for i,f in enumerate(selection['folds']):
            part=intervals[(intervals.fold==i)&(intervals.comparison==comparison)].set_index('metric')
            controls.append(f"| {f['dates']['test']} | {label} | {ci(part.loc['ndcg@10'])} | {ci(part.loc['recall@10'])} |")
    freq='\n'.join(f'| `{name}` | {count}/3 |' for name,count in feature_freq.most_common(12))
    family_text='; '.join(f'{name}: {count}' for name,count in families.items())
    boundary=sum(f['params']['iterations']==max(manifest['config']['tree_checkpoints']) for f in selection['folds'])
    new_final=all_rows[(all_rows.test_cutoff=='2021-08-16')&(all_rows.method=='manual')&(all_rows.seed==42)].iloc[0]
    old=load(Path('artifacts/baseline/results.json'))['test']['catboost']['ndcg@10']
    body=f'''# Автоматизация признаков в рекомендательной системе KION

## Результат

Сравниваются CatBoost на 46 ручных признаках, дополнение автоматически
отобранными агрегатами и замена ручных агрегатов. Из 144 признаков отбираются
12 или 24 по временным out-of-fold ошибкам. Глубина, число деревьев и размер
набора выбираются по validation NDCG@10. Оценка проведена на трёх тестовых
неделях с seed 42 и 2026.

Средние метрики: каждой неделе одинаковый вес, внутри недели усредняются два seed.
Это описательное среднее шести запусков, не шесть независимых экспериментов.

| Подход | Средний NDCG@10 | Средний Recall@10 |
|---|---:|---:|
{aggregate}

По парному bootstrap на отдельных неделях NDCG-прирост авто-дополнения над
ручным baseline: **на {positive} из 3 недель интервал выше нуля, на {negative} ниже нуля,
на {uncertain} включает ноль**. Наблюдаемый эффект невелик и зависит от периода;
универсальный выигрыш на произвольных будущих неделях не заявляется.

Ручной baseline превосходит недавнюю популярность по NDCG с положительным
парным интервалом на {manual_wins}/3 недель.
Автоотбор превосходит случайное дополнение с таким интервалом на {random_wins}/3
недель: превосходство конкретного фильтра над случайным поиском не универсально.
Полная замена ручных агрегатов уступает baseline с отрицательным интервалом
на {replacement_losses}/3 недель. Сокращение набора признаков привело к потере
качества на части периодов.

## Данные и задача

[KION](https://github.com/irsafilo/KION_DATASET) содержит
5 476 251 запись, 962 179 пользователей, 15 706 объектов; 13 марта — 22 августа
2021 года. Использован полный архив закреплённого коммита. Отдельный ZIP
interactions в репозитории обрезан и не используется. SHA256 и версии сохранены.

Случайная выборка 50 000 пользователей (seed=42), **все 285 343 доступные строки**
по ним. Истории не обрезались при сэмплировании пользователей. Это агрегированные
пары user–item с last_watch_dt: последнее взаимодействие, суммарная длительность
и доля просмотра. Полный журнал первых/повторных действий восстановить нельзя.

Задача — предсказывать записи просмотра на следующую неделю, исключая пары
из наблюдаемого прошлого. Это offline proxy, а не точная модель новых просмотров
или CTR: показов и достоверных отказов нет. Отсутствие записи — условный ноль.
Метаданные считаются статическими. На каждом срезе используются только строки
с last_watch_dt строго раньше начала недели, включая статистики других пользователей.

Источники кандидатов — общая и недавняя популярность, жанровые предпочтения,
соседи по совместным просмотрам; RRF и до 200 кандидатов. Ответы из будущего
не добавляются. Все методы на одной неделе получают **одни и те же кандидаты**.
Пропущенные генератором/холодные целевые фильмы остаются в знаменателях.
Оценка условна на warm-пользователях с целевыми записями. Cold-start и
пользователи без будущего просмотра не входят в средние ranking-метрики.

## Временная проверка и подбор baseline

Для каждой тестовой недели перед ней расположена validation-неделя, а перед
ней — три train-недели. Например, test 2–8 августа, validation 26 июля – 1 августа,
train 5–11 / 12–18 / 19–25 июля. Пользователи могут повторяться; это проверка
переноса во времени. Более ранние test-недели допустимы в истории последующих
моделей. Все настройки всех folds сохранены до расчёта новых тестовых метрик.

| Начало test | Начало validation | Глубина | Деревьев | Автопризнаков | OOF-групп отбора |
|---|---|---:|---:|---:|---:|
{chr(10).join(fold_lines)}

Ручной baseline — 46 признаков: пользовательские и объектные агрегаты за
7/30 дней и всё прошлое, давность, отношения активности, жанровые/страновые/
актёрские/режиссёрские affinities, метаданные и признаки генератора. ID в модель
не подаются. Внутренние категориальные статистики CatBoost обучаются на train,
Counter CTR=SkipTest.

Сетка: depth 5/7 × trees 16/32/64/128, learning_rate=0,06, l2=5. Выбор по
полной validation NDCG@10 на seed=42; при равенстве меньше деревьев, затем
меньшая глубина. Ранняя остановка по LogLoss не используется. На seed=2026 параметры
не подбираются заново. В {boundary}/3 folds выбрана верхняя граница числа деревьев:
оптимальность по неограниченному бюджету не доказана.

На неделе 16–22 августа ручной CatBoost с seed=42 даёт
**{num(new_final['ndcg@10'])}** против **{num(old)}** у пилота. Это результат
совместного изменения выбора деревьев и взвешенного сэмплирования;
его нельзя приписывать только новому критерию остановки.
Полная сетка: [validation_grid.csv](reports/robust/validation_grid.csv).

## Обучающие пары и вычислительный бюджет

На каждом train-срезе — 3 000 случайных пользователей независимо от будущего.
Сохраняются все положительные кандидаты и до 30 отрицательных на пользователя.
Вес отрицательного: N_negative / min(30,N_negative); вес положительного: 1.
Сумма весов равна числу исходных пар. Для фиксированной функции это
importance-weighted оценка полной суммы потерь, а не гарантия идентичности
полученной модели. Сэмплируются только обучающие пары, **не истории событий**.

Все сравниваемые методы используют ровно те же выбранные строки и веса.
Validation и test полные, без отрицательного сэмплирования. На train-неделю
получается около 90 тысяч пар вместо 590–600 тысяч. Это даёт возможность
проверить несколько периодов и seed на машине с 8 GiB RAM.

## Автоматизация и временной отбор

Пространство поиска: сущность × временное ядро × агрегат.
Жёсткие окна 1/3/5/10/14/21/45/60 дней и exp(−age/tau), tau=3/10/21/60.
На каждое ядро — 12 признаков: числа/средние/доли пользователя и объекта,
affinities через жанры, страны, режиссёров, актёров. Всего 144; числа и средняя
длительность преобразованы log1p. Для актёров тот же лимит 5 имён по алфавиту.
Набор сущностей, ядер и агрегатов фиксирован до отбора.

Для train-недели 2 модель обучается
только на неделе 1; для недели 3 — только на неделях 1–2. Prefix-модели имеют
фиксированные depth=6, trees=64, seed=42, независимо от validation/test.
Отбор использует эти **временные OOF-ошибки** и только группы с положительным
кандидатом. Финальная модель обучается на всех трёх train-неделях.

Из признака и остатка label−prediction вычитается взвешенное среднее по группе.
Score — абсолютная взвешенная корреляция; жадно удаляем избыточность при
|corr|≥0,97. Веса корректируют отрицательное сэмплирование. Validation/test-метки
не используются в screening. Размер добавления 12/24 выбирается по validation.
User-only признаки после центрирования константны — это ограничение метода.

Проверяются четыре метода при одинаковых depth/trees/train/весах:

- manual: 46 ручных признаков;
- auto: manual + k выбранных признаков;
- random: manual + k случайных признаков из 144;
- replacement: 11 метаданных + 3 признака генератора + k автоматических агрегатов.

На seed=2026 используются те же выбранные k и параметры. Случайные наборы
определяются fold и model seed: два набора на неделю, шесть в сумме.
Абляции не используются для переключения основного метода по тесту.

Самые часто выбираемые признаки:

| Признак | В скольких folds выбран |
|---|---:|
{freq}

Состав наборов и OOF scores: [selected_features.csv](reports/robust/selected_features.csv)
и `reports/robust/fold_*/screening.csv`. Число включений по всем трём наборам:
{family_text}. Это характеристика отбора, не причинное доказательство
полезности отдельных признаков.

## Результаты по неделям и seed

Среднее по двум seed; последний столбец показывает разброс NDCG по seed,
а не доверительный интервал. У популярных контролей прогноз один.

| Начало недели | Метод | NDCG@10 | Recall@10 | NDCG min–max по seed |
|---|---|---:|---:|---|
{chr(10).join(mean_rows)}

## Парное сравнение основного метода

На каждой неделе сначала усредняются per-user метрики двух seed, затем
bootstrap пользователей (2 000 повторов, seed=42) даёт парный 95%-интервал.

| Неделя | Δ NDCG seed42 | Δ NDCG seed2026 | Δ NDCG [95% CI] | Δ Recall [95% CI] |
|---|---:|---:|---|---|
{chr(10).join(differences)}

Другие заранее заданные сравнения:

| Неделя | Сравнение | Δ NDCG [95% CI] | Δ Recall [95% CI] |
|---|---|---|---|
{chr(10).join(controls)}

Интервалы условны на этих обученных моделях и когортах. Пользователи и train
между неделями пересекаются; шесть запусков нельзя считать независимыми.
Для общего среднего трёх недель CI/p-value не заявляется. Поправки на
множественные вторичные сравнения нет. Отсутствие обнаруженного различия
не является доказательством эквивалентности; non-inferiority margin не задавался.
95%-покрытие относится к отдельному интервалу, а не одновременно ко всей
совокупности недель и сравнений.

## Интерпретация и ограничения

Эффект автоотбора зависит от недели. По среднему NDCG он немного лучше ручного
набора, но преимущество над случайными признаками подтверждается лишь на одной
неделе. Параметры после теста не менялись. Причины различий можно проверить
отдельными абляциями компонентов.

Корреляционный screening ограничен линейным сигналом и теряет user-only
контекст; признаки могут конкурировать за разбиения деревьев. Отбор не исправляет
ответы, отсутствующие в кандидатах. Нужны дополнительные исследования для
установления причин, а не только feature importance.

Данные этих периодов ранее использовались в пилоте. Это out-of-time оценка относительно каждой новой
модели, но не новый полностью слепой внешний holdout. Три недели и два seed
не покрывают все режимы сервиса. Ограничения last_watch, отсутствия показов,
статических метаданных и warm-user когорты сохраняются.

Сокращение числа признаков replacement не означает экономии всего pipeline:
144 признака перебирались, генератор остаётся ручным. Масштабирование на
миллиарды событий не доказано; item-item co-occurrence и расширение грамматики
требуют блочных/приближённых и инкрементальных расчётов.

## Ресурсы и воспроизведение

CPU, 4 потока CatBoost, macOS ARM64, 8 GiB RAM. Полная история та же;
зависимости закреплены. Измеренные затраты:

| Этап | Время, с | Пик RSS, MiB |
|---|---:|---:|
| 9 наборов пар и автопризнаки | {manifest['prepare_seconds']:.2f} | {manifest['prepare_peak_rss_mb']:.0f} |
| Подбор, OOF и модели двух seed | {selection['fit_seconds']:.2f} | {selection['fit_peak_rss_mb']:.0f} |
| Оценка тестов, интервалы, сохранение | {results['evaluate_seconds']:.2f} | {results['evaluate_peak_rss_mb']:.0f} |

Время подготовки переиспользует исходные таблицы пилота для части срезов;
это дополнительные затраты, не полный запуск от скачивания. Время fit включает
36 обучений: конечные модели, поисковые кандидаты и 6 prefix-моделей.
Фиксация победителя не требует повторного обучения выбранного seed=42.

Для имеющегося baseline:

```bash
python -m unittest discover -s tests -v
python -m src.robustness prepare
python -m src.robustness fit
python -m src.robustness evaluate
python -m src.robust_report
```

Есть режим `python -m src.robustness all`. Fit возобновляется по сохранённым
моделям с проверкой спецификаций; prepare — по завершённым наборам. При смене
конфигурации используйте отдельный `--output`. Большие данные/кэши и веса
в `artifacts/` не включены в архив кода, воспроизводятся командами.

С нуля: окружение и загрузка по [README](README.md), затем `src.pipeline all`;
для полного воспроизведения пилота ещё `src.auto_pipeline all` и
`src.final_report`. Проверка на трёх неделях запускается командами выше. 11 тестов проверяют:
время, будущие события, метрики, пропуски, ядра, веса и сохранение positives.

Материалы: [конфигурация](robust_config.json),
[протокол до новых метрик](reports/06_robustness_protocol.md),
[все fold×seed результаты](reports/robust/all_results.csv),
[интервалы](reports/robust/paired_intervals.csv),
[выбор](reports/robust/selection.json).
Исходное сравнение на одной неделе — в [PILOT_REPORT.md](PILOT_REPORT.md).

Источник: Petrov et al. (2022), *MTS Kion Implicit Contextualised Sequential
Dataset for Movie Recommendation*, RecSys CARS Workshop;
[описание полей и дат](https://ods.ai/competitions/competition-recsys-21/data).
'''
    if not Path('PILOT_REPORT.md').exists():shutil.copyfile('REPORT.md','PILOT_REPORT.md')
    Path('REPORT.md').write_text(body,encoding='utf-8')
    print('Saved REPORT.md')


if __name__=='__main__':main()
