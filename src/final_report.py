import json
import shutil
from pathlib import Path
import pandas as pd
from .metrics import bootstrap


def load(path):
    return json.loads(path.read_text())


def number(x, digits=5):
    return f'{x:.{digits}f}'.replace('.', ',')


def interval(x):
    return f"{number(x['mean'])} [{number(x['low_95'])}; {number(x['high_95'])}]"


def main():
    baseline=Path('artifacts/baseline')
    auto=Path('artifacts/auto_features')
    export=Path('reports/auto')
    export.mkdir(parents=True,exist_ok=True)
    manifest=load(baseline/'manifest.json')
    bs=load(baseline/'selection.json')
    br=load(baseline/'results.json')
    am=load(auto/'manifest.json')
    selection=load(auto/'selection.json')
    results=load(auto/'results.json')
    primary=selection['selected_main']
    chosen=next(x for x in selection['main_candidates'] if x['name']==primary)
    for name in ['manifest.json','catalog.json','screening.json','screening.csv','selection.json','results.json']:
        shutil.copyfile(auto/name,export/name)
    for variant in [chosen,selection['random_control'],selection['replacement']]:
        shutil.copyfile(auto/f"importance_{variant['name']}.csv",export/f"importance_{variant['name']}.csv")

    replacement=pd.read_csv(auto/'test_users_auto_replacement.csv').set_index('user_id')
    secondary={}
    for control in ['catboost','recent_popularity']:
        comparator=pd.read_csv(baseline/f'test_users_{control}.csv').set_index('user_id')
        assert replacement.index.equals(comparator.index)
        secondary[f'auto_replacement_minus_{control}']={
            key:bootstrap(replacement[key]-comparator[key],selection['config']['seed'],
                           selection['config']['bootstrap_repetitions']) for key in ['ndcg@10','recall@10']}
    (export/'secondary_comparisons.json').write_text(json.dumps(secondary,indent=2))
    table=[]
    for name,label,features,val in [
        ('recent_popularity','Популярность за 7 дней',1,bs['controls']['recent_popularity']),
        ('catboost','Ручной CatBoost',len(bs['features']),bs['models'][bs['selected_model']]['validation']),
        ('catboost_no_pair','Без признаков пары',len(bs['ablation']['features']),bs['ablation']['validation'])]:
        table.append({'model':name,'label':label,'features':features,'validation_ndcg':val['ndcg@10'],
                      **{key:br['test'][name][key] for key in ['ndcg@10','recall@10']}})
    for variant,label in [(chosen,'Ручные + автоотбор (основной метод)'),
                           (selection['random_control'],'Ручные + случайные признаки'),
                           (selection['replacement'],'Замена ручных агрегатов (абляция)')]:
        name=variant['name']
        table.append({'model':name,'label':label,'features':len(variant['features']),
                      'validation_ndcg':variant['validation']['ndcg@10'],
                      **{key:results['test'][name]['metrics'][key] for key in ['ndcg@10','recall@10']}})
    pd.DataFrame(table).to_csv('reports/final_comparison.csv',index=False)
    comparison='\n'.join(f"| {r['label']} | {r['features']} | {number(r['validation_ndcg'])} | {number(r['ndcg@10'])} | {number(r['recall@10'])} |" for r in table)
    paired='\n'.join(f"| Основной метод − {label} | {interval(results['paired_differences'][f'{primary}_minus_{key}']['ndcg@10'])} | {interval(results['paired_differences'][f'{primary}_minus_{key}']['recall@10'])} |"
        for key,label in [('catboost','ручной CatBoost'),('recent_popularity','популярность за 7 дней'),
                          ('random_matched','случайное дополнение'),('auto_replacement','замена агрегатов')])
    selected_rows=[]
    catalog=load(auto/'catalog.json')
    meanings={'i_count':'log1p числа просмотров объекта','i_share':'доля веса недавних просмотров объекта',
              'i_duration':'log1p средней длительности просмотра объекта'}
    for name in chosen['extra_features']:
        entry=catalog[name]
        kind='Окно' if entry['kind']=='window' else 'Экспонента, tau'
        selected_rows.append(f"| `{name}` | {kind} {entry['days']} дней | {meanings.get(entry['expression'],entry['expression'])} |")
    chosen_table='\n'.join(selected_rows)
    search='\n'.join(f"| {len(v['extra_features'])} | {len(v['features'])} | {number(v['validation']['ndcg@10'])} | {number(v['validation']['recall@10'])} |" for v in selection['main_candidates'])
    secondary_table='\n'.join(f"| Замена − {name} | {interval(secondary[f'auto_replacement_minus_{key}']['ndcg@10'])} | {interval(secondary[f'auto_replacement_minus_{key}']['recall@10'])} |"
        for key,name in [('catboost','ручной CatBoost'),('recent_popularity','недавняя популярность')])
    bauc=br['test']['catboost']['sampled_pair_auc']
    aauc=results['test'][primary]['metrics']['sampled_pair_auc']
    report=f'''# Автоматическая генерация и отбор признаков для рекомендаций KION

Пилотный эксперимент: одна тестовая неделя, seed 42.
Проверка на трёх неделях описана в [REPORT.md](REPORT.md).

## Результат

Реализованы базовый CatBoost на ручных признаках и алгоритм автоматического
перечисления/отбора агрегатов. **Подтверждённого выигрыша основного метода
над ручным бустингом по NDCG@10 нет.** На тесте прирост небольшой и его парный
95%-интервал включает ноль; Recall не улучшился. Отбор оказался лучше одного
случайного набора того же размера. Проверка замены агрегатов дала сопоставимые
точечные метрики при 26 признаках вместо 46, но формальная эквивалентность
качества не доказана.

| Подход | Признаков | Validation NDCG@10 | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|---:|---:|
{comparison}

Основной метод выбран по validation **только между дополнениями +12 и +24**.
Замена агрегатов и случайный набор — заранее заданные абляции. Их результаты
не использованы, чтобы сменить основной метод после просмотра теста.

## Данные и задача

Источник — [официальный KION](https://github.com/irsafilo/KION_DATASET), полный
архив из закреплённого коммита. Фактически проверено 5 476 251 взаимодействие,
962 179 пользователей, 15 706 объектов, даты 13 марта — 22 августа 2021 года.
Отдельный ZIP взаимодействий в репозитории оказался обрезанным и не используется.
SHA256 источников сохранены в [манифесте](reports/manifest.json).

Случайно выбраны 50 000 пользователей, seed 42. Сохранены **все 285 343 строки**
по ним; события не сэмплировались. Статистики рассчитаны внутри этой выборки.
Для обучения берём 3 000 случайных warm-пользователей на каждом срезе,
независимо от будущей активности. Всего {bs['training_rows']:,} обучающих пар,
{bs['training_positives']} положительных меток. Отсутствующая демография не
ведёт к исключению пользователя, значения заменяются на unknown.

Цель — ранжировать объекты, чья запись просмотра относится к следующей неделе,
исключая пары из наблюдаемого прошлого. **В источнике last_watch_dt — дата
последнего просмотра пары, а не полный журнал событий.** Поэтому реальную
историю первых/повторных просмотров восстановить нельзя: решается offline
proxy-задача предсказания агрегированных записей, не CTR и не точная имитация
сервиса. Показов и достоверных отказов нет; отсутствие просмотра — условный ноль.

## Время, кандидаты и отсутствие утечек

Три train-недели: 19–25 июля, 26 июля – 1 августа, 2–8 августа 2021.
Validation: 9–15 августа. Test: 16–22 августа. Каждый срез видит только строки
с last_watch_dt строго меньше начала недели; целевая неделя целиком закрыта
для признаков. Статические метаданные общие; их исторических версий нет.

Генератор объединяет общую/недавнюю популярность, любимые жанры и соседство
фильмов по совместным просмотрам; оставляет до 200 кандидатов. Ответы из будущего
не добавляются. Для всех моделей сохранены одни и те же пары и их порядок;
на втором этапе проверены хеши кандидатов и совпадение test-пар/меток с первым.

Ранжирование оценивается по 5 698 warm-пользователям с целевыми записями.
2 939 пользователей без прошлого не входят в основную оценку; качество для
них не измерено. Пользователи без целевых записей на неделе также не входят
в средние NDCG/Recall. Каталог содержит только объекты, наблюдавшиеся до среза.
**Пропущенные генератором и холодные целевые фильмы остаются в знаменателях.**
Средняя полнота кандидатов — 0,50771; NDCG@10 идеального ранжирования тех же
кандидатов — 0,55524. У генерации и ранжирования остаётся запас улучшения.

## Базовый подход

46 ручных признаков: статистики пользователя/объекта за 7/30 дней и всё
прошлое, давность/возраст истории, отношения активности, affinities к жанрам,
странам, актёрам и режиссёрам, score/rank генератора, демография и метаданные.
User/item ID не подаются в модель. Прямые user–item счётчики нулевые из-за
исключения наблюдавшихся пар. CatBoost обучается на всех кандидатах выбранных
train-пользователей, без class weights и отрицательного сэмплирования.

На этапе 1 проверены глубины 5/7, ранняя остановка по validation LogLoss;
конфигурация выбрана по validation NDCG@10. Выбран CatBoost: depth=7,
learning_rate=0,06, l2=5, seed=42, 27 деревьев после остановки. Для новых
моделей используются те же параметры и **фиксированные 27 деревьев**, без
новой остановки/перебора learner-гиперпараметров. Доступная информация одинакова.
Подробности baseline, абляция и ошибки: [отчёт этапа 1](reports/02_baseline_report.md).

## Исследовательский метод

Признаки строятся по грамматике «сущность × временное ядро × агрегат».
Набор допустимых операций фиксирован; алгоритм перечисляет комбинации
и отбирает их по ошибкам baseline.

Жёсткие окна: 1/3/5/10/14/21/45/60 дней. Плавные ядра: exp(−age/tau),
tau=3/10/21/60 дней. Для каждого из 12 ядер — 12 признаков: числа строк,
средние доля и длительность просмотра, доля недавнего веса для пользователя
и объекта, affinities пары через 4 вида категорий. Числа и средняя длительность
преобразуются log1p. В affinities нормируем вес общих категорий на вес истории
внутри ядра и число категорий кандидата. Итого **144 признака**.

На train берём группы (срез, пользователь) с положительным кандидатом:
{selection['screening_groups']} группы, {selection['screening_rows']:,} строк. Остаток — label минус
предсказание замороженного baseline. Из остатков и признаков вычитается
среднее по группе; score — абсолютная корреляция с центрированным остатком.
Жадно удаляем избыточные признаки при |corr| ≥ 0,97 на фиксированной выборке
до 100 000 train-строк. После фильтра осталось 75 признаков. Для финального
обучения используются все исходные train-пары, не только screening-группы.

Проверка размера набора на validation:

| Добавлено | Всего | NDCG@10 | Recall@10 |
|---:|---:|---:|---:|
{search}

Выбран набор +{selection['selected_count']}. Его состав:

| Признак | Временное ядро | Смысл |
|---|---|---|
{chosen_table}

Все выбранные признаки описывают **объект**. Жанровые/актёрские связи в
короткий набор не вошли. Константные внутри группы user-only признаки
отбрасываются по конструкции фильтра. Человеческие решения о грамматике,
способе screening и лимите поиска сохраняются — автоматизация не полная.
Каталог, scores и выбор: [артефакты отбора](reports/auto/screening.csv),
[зафиксированная selection](reports/auto/selection.json).

## Статистическое сравнение и абляции

В таблице — средняя разность и парный 95%-интервал, 2 000 bootstrap-выборок
пользователей, seed 42. Интервалы условны на обученных моделях и этой неделе.
Это не оценка вариативности переобучения или временных периодов.
Поправка на множественные вторичные сравнения не применялась; дополнительные
выводы рассматриваются как исследовательские.

| Сравнение | Разность NDCG@10 [95% CI] | Разность Recall@10 [95% CI] |
|---|---|---|
{paired}

У основного метода нет подтверждённого NDCG-преимущества над baseline или
недавней популярностью. Его Recall ниже недавней популярности, и парный
интервал разности не включает ноль. По сравнению с **одним** случайным набором
отбор улучшает обе ranking-метрики; обобщение на все случайные наборы не проверено.

При замене ручных агрегатов сохранены 11 метаданных, 3 признака генератора
и 12 выбранных статистик — всего 26. Генератор по-прежнему ручной и общий:
это не замена всей рекомендательной системы. Дополнительные сравнения:

| Сравнение | Разность NDCG@10 [95% CI] | Разность Recall@10 [95% CI] |
|---|---|---|
{secondary_table}

Замена даёт перспективный компромисс числа признаков и точечных метрик.
Сокращение входной размерности не означает сокращение общих затрат:
алгоритм всё равно перебирал 144 признака, а ручной генератор сохранён.
Однако «различие не обнаружено» не означает доказанную эквивалентность:
допуск non-inferiority не задавался, тест на неухудшение не проводился.
Основной метод после просмотра этих результатов не менялся.

## Обсуждение

Отобранный набор лучше фиксированного случайного контроля; увеличение
набора с 12 до 24 ухудшило validation NDCG. Автоматическое дополнение
повысило глобальный ROC-AUC по кандидатам с {number(bauc)} до {number(aauc)},
но это не дало подтверждённого выигрыша в ранжировании внутри пользователя.
LogLoss/AUC не являются метриками CTR и не заменяют NDCG/Recall.

Ограничения метода объясняют возможные причины, но не доказывают их:

- In-sample остатки baseline и линейный score могут выбирать популярность,
  а не признаки с устойчивым приростом на новых неделях.
- Центрирование теряет user-only признаки и их нелинейные взаимодействия;
  корреляционное удаление также может убрать полезные комбинации.
- Бюджет 27 деревьев ограничен: больше признаков конкурируют за те же разбиения.
- Отбор работает только с покрытыми генератором train-запросами и не способен
  исправить пропущенные фильмы. Истории KION разрежены и агрегированы.

Для проверки причин нужны отдельные исследования: временной out-of-fold
screening, wrapper-отбор по ranking-метрике, несколько seed/периодов и
изменение бюджета деревьев с равноценным подбором baseline. Они в этот
эксперимент не включены; после теста параметры не дорабатывались.

## Вычислительные затраты и масштаб

CPU, ARM64/macOS, 8 GiB RAM, 4 потока CatBoost. Версии зависимостей закреплены.
Время ниже — измеренные этапы, не прогноз промышленной производительности.

| Этап | Время, с | Пик RSS, MiB |
|---|---:|---:|
| Baseline: все 5 срезов | {manifest['prepare_seconds']:.2f} | {manifest['prepare_peak_rss_mb']:.0f} |
| Baseline: поиск конфигурации и абляция | {bs['fit_seconds']:.2f} | {bs['fit_peak_rss_mb']:.0f} |
| Автопризнаки: train и validation | {am['prepare_seconds']:.2f} | {am['prepare_peak_rss_mb']:.0f} |
| Screening и обучение четырёх новых моделей | {selection['fit_seconds']:.2f} | {selection['fit_peak_rss_mb']:.0f} |
| Автопризнаки test, предсказания и метрики | {results['evaluate_seconds']:.2f} | {results['evaluate_peak_rss_mb']:.0f} |

Предсказание основной новой модели для 1 128 618 пар —
{results['test'][primary]['predict_seconds']:.2f} с, без построения признаков.
Это дополнительные затраты поверх подготовленного baseline: кандидаты и
его таблицы переиспользуются. Меньшее время обучения второго этапа не означает
превосходства алгоритма по скорости: в нём фиксированы 27 деревьев, а первый
этап также выполнял раннюю остановку. Архив данных, скачивание, документация
и прерванный из-за RAM первый запуск в эти времена не включены.

Автопризнаки считаются блоками по 12 столбцов; для screening не создаётся
полная train-матрица 1,8 млн × 144. Но расширение грамматики умножает стоимость
агрегатов, а исходный item-item co-occurrence может сильно расти по памяти.
Работоспособность на этой выборке показана; масштабируемость до миллиардов
событий не доказана. Нужны инкрементальные агрегаты, блочные/приближённые
соседства, ограничение числа сущностей и бюджет поиска.

## Ограничения достоверности

Одна выборка пользователей, одна validation- и одна test-неделя, один seed.
Bootstrap не учитывает общую историю/каталог и вариативность обучения.
Тест baseline уже был открыт до разработки этапа 2: это **не полностью
слепой holdout**. Новый протокол записан до новых метрик; после фиксации
выбора новые модели один раз оценены на том же тесте. Для внешнего подтверждения
нужен другой период/датасет. Протоколы и исторические результаты сохранены.

Из KION нельзя получить полную последовательность повторных действий,
журнал показов и достоверные отрицательные отклики. Полная доступная история
пользователей сохранена, но это не восстанавливает отсутствующие события.
Результаты warm-user оценки нельзя переносить на cold-start или онлайн-эффект.

## Воспроизведение и материалы

Из корня проекта, Python 3.10:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download
python -m unittest discover -s tests -v
python -m src.pipeline all
python -m src.diagnostics
python -m src.auto_pipeline prepare
python -m src.auto_pipeline fit
python -m src.auto_pipeline evaluate
python -m src.final_report
```

Если baseline уже подготовлен, начинайте с `src.auto_pipeline prepare`.
Конфигурации: [baseline](config.json), [автоотбор](auto_config.json).
Тесты проверяют временные границы, влияние будущих записей, кандидаты, метрики,
пропуски категорий, нормализацию и затухание автопризнаков.
Полные результаты: [CSV сравнения](reports/final_comparison.csv),
[JSON нового эксперимента](reports/auto/results.json),
[дополнительные сравнения](reports/auto/secondary_comparisons.json).
Протокол этапа 2: [04_research_protocol.md](reports/04_research_protocol.md).
Исходные данные и большие кэши не включены в архив кода; они воспроизводятся
скачиванием закреплённого источника и командами выше. Межплатформенная
идентичность последних битов не гарантируется.

Источник: Petrov et al. (2022), *MTS Kion Implicit Contextualised Sequential
Dataset for Movie Recommendation*, RecSys CARS Workshop;
[поля и даты данных](https://ods.ai/competitions/competition-recsys-21/data).
'''
    Path('PILOT_REPORT.md').write_text(report,encoding='utf-8')
    if Path('artifacts/robustness/results.json').exists():
        from .robust_report import main as render_robust
        render_robust()
    else:
        Path('REPORT.md').write_text(report,encoding='utf-8')
    print('Saved reports')


if __name__=='__main__':main()
