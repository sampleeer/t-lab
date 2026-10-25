"""Generate a Russian report and figures exclusively from saved run outputs."""
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rl_recsys_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT, dump, digest


LABELS = {'sft':'GPT-2 SFT', 'popularity':'Популярность', 'last_item':'Последний + популярность',
          'fraction_weak':'Доля, KL=0.01', 'fraction_kl':'Доля, KL=0.2',
          'time_kl':'Время, KL=0.2', 'balanced_5step':'Смесь, 5 шагов, KL=0.2'}


def label(name):
    for suffix in ['_selected','_endpoint']:
        if name.endswith(suffix): return LABELS[name[:-len(suffix)]]
    return LABELS.get(name,name)


def md(frame, digits=5):
    columns=list(frame.columns)
    def fmt(x):
        if isinstance(x,(float,np.floating)): return f'{x:.{digits}f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,columns))+' |',
        '| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    reports=ROOT/'reports'; figures=reports/'figures'; figures.mkdir(exist_ok=True)
    config=json.loads((ROOT/'config.json').read_text())
    data=json.loads((reports/'data_checks.json').read_text())
    fit=json.loads((reports/'fit_manifest.json').read_text())
    selected=json.loads((reports/'selection.json').read_text())
    quality=json.loads((reports/'reward_quality.json').read_text())
    intervals=json.loads((reports/'paired_intervals.json').read_text())
    rollouts=json.loads((reports/'rollout_metrics.json').read_text())
    results=pd.read_csv(reports/'test_metrics.csv')
    mean=results.groupby('model').mean(numeric_only=True)
    mean.to_csv(reports/'mean_metrics.csv')
    order=['popularity','last_item','sft']+[v['name']+'_selected' for v in config['variants']]
    main_table=mean.loc[order,['ndcg10','recall10','pred_fraction','audit_fraction','pred_seconds','kl']].reset_index()
    main_table['model']=main_table.model.map(label)
    main_table.columns=['Метод','NDCG@10','Recall@10','RM: доля','Аудитор: доля','RM: секунды','KL']
    endpoints=[v['name']+'_endpoint' for v in config['variants']]
    end_table=mean.loc[['sft']+endpoints,['ndcg10','recall10','pred_fraction','pred_seconds',
                    'duration_seconds','catalog_coverage_top10','top1_max_share','recent_repeat_top10']].reset_index()
    end_table['model']=end_table.model.map(label)
    end_table.columns=['Метод','NDCG@10','Recall@10','RM: доля','RM: секунды','Длина, с','Охват@10','Max top-1','Повтор@10']
    choices=[]; ci=[]; reward_rows=[]
    for seed in config['seeds']:
        folder=ROOT/f'artifacts/seed_{seed}'
        for file in folder.glob('*_validation.json'):
            shutil.copy2(file,reports/f'{seed}_{file.name}')
        for file in folder.glob('*_training.json'):
            shutil.copy2(file,reports/f'{seed}_{file.name}')
        for v in selected[str(seed)]:
            choices.append({'Seed':seed,'Вариант':label(v['name']),'Шаг':v['selected_step'],
                            'Min val NDCG':v['validation_ndcg_floor']})
            for suffix in ['selected','endpoint']:
                record=intervals[str(seed)][v['name']+'_'+suffix]
                ci.append({'Seed':seed,'Вариант':label(v['name'])+' / '+suffix,
                           'ΔNDCG':record['ndcg10']['delta'],'95% low':record['ndcg10']['lower95'],
                           '95% high':record['ndcg10']['upper95'],
                           'ΔRM':record['pred_fraction']['delta']})
        for name, values in quality[str(seed)]['optimizer'].items():
            if name=='fraction_calibration':continue
            reward_rows.append({'Seed':seed,'Модель':name,'RMSE доли':values['fraction_rmse'],
                                'MAE доли':values['fraction_mae'],'Brier продолжения':values['continuation_brier'],
                                'AUC продолжения':values['continuation_auc']})
    trajectory_rows=[]
    for seed in config['seeds']:
        for name in ['sft']+endpoints:
            trajectory=pd.read_csv(ROOT/f'artifacts/seed_{seed}/{name}_rollouts.csv')
            discount=0.95**(trajectory.step-1)
            trajectory['fraction_return']=discount*trajectory.pred_fraction
            trajectory['audit_return']=discount*trajectory.audit_fraction
            trajectory['mixed_return']=discount*(0.5*trajectory.pred_fraction+
                0.25*trajectory.pred_fraction*trajectory.duration_seconds.clip(upper=600)/300+
                0.25*trajectory.pred_continuation)
            returns=trajectory.groupby(['user','event'])[['fraction_return','audit_return','mixed_return']].sum().groupby('user').mean().mean()
            trajectory_rows.append({'seed':seed,'model':name,**returns.to_dict()})
    trajectory_table=pd.DataFrame(trajectory_rows).groupby('model').mean(numeric_only=True).loc[['sft']+endpoints].drop(columns='seed').reset_index()
    trajectory_table['model']=trajectory_table.model.map(label)
    trajectory_table.columns=['Метод','Σγ^t RM-доля','Σγ^t аудитор','Σγ^t смесь']
    pd.DataFrame(trajectory_rows).to_csv(reports/'trajectory_returns.csv',index=False)
    pd.DataFrame(ci).to_csv(reports/'paired_intervals.csv',index=False)

    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    colors=['#d95f02','#1b9e77','#7570b3','#e7298a']
    for seed in config['seeds']:
        for v,color in zip(config['variants'],colors):
            history=json.loads((reports/f"{seed}_{v['name']}_validation.json").read_text())
            axes[0].plot([h['ndcg10'] for h in history],[h['pred_fraction'] for h in history],
                         marker='o',color=color,alpha=.7,label=v['name'] if seed==config['seeds'][0] else None)
    axes[0].set(xlabel='Validation NDCG@10',ylabel='Expected predicted fraction',title='Validation trajectories (0, 100, 200, 300)')
    axes[0].legend(fontsize=8)
    methods=['sft']+endpoints
    x=np.arange(len(methods))
    axes[1].bar(x,mean.loc[methods,'ndcg10'],color=['#444444']+colors)
    axes[1].set_xticks(x,['SFT','frac weak','frac KL','time KL','mix 5-step'],rotation=20)
    axes[1].set(ylabel='Test NDCG@10',title='Final endpoints, mean of two seeds')
    fig.tight_layout();fig.savefig(figures/'tradeoff.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,color in [('sft','#444444'),('fraction_weak_endpoint',colors[0]),('balanced_5step_endpoint',colors[3])]:
        grouped=pd.concat([pd.DataFrame(rollouts[f'{s}/{name}']) for s in config['seeds']]).groupby('step').mean(numeric_only=True)
        axes[0].plot(grouped.index,grouped.pred_fraction,marker='o',label=name,color=color)
        axes[1].plot(grouped.index,grouped.repeat_generated,marker='o',color=color)
        axes[2].plot(grouped.index,grouped.pred_fraction-grouped.frozen_fraction,marker='o',color=color)
    axes[0].set(ylabel='RM predicted fraction',title='Synthetic-context reward')
    axes[1].set(ylabel='Fraction',title='Repeated generated item')
    axes[2].set(ylabel='RM fraction difference',title='Synthetic minus frozen context')
    for ax in axes: ax.set(xlabel='Generation step',xticks=range(1,6))
    axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(figures/'rollout.png');plt.close(fig)

    examples=[]
    seed=config['seeds'][0]
    catalog=dict(np.load(ROOT/'artifacts/catalog.npz'))
    with np.load(ROOT/'artifacts/test.npz') as test:
        for i in [0, min(100,len(test['y'])-1),min(500,len(test['y'])-1)]:
            examples.append({'user':int(test['user'][i]),'example':i,
                'last_items':[int(catalog['item_id'][t]) if t>=2 else 'UNK' for t in test['x'][i] if t],
                'target_item':int(catalog['item_id'][test['y'][i]]) if test['y'][i]>=2 else 'OOV',
                'observed_fraction':float(test['fraction'][i]),
                'recommendations':{name:[int(v) for v in catalog['item_id'][np.load(ROOT/f'artifacts/seed_{seed}/{name}_top10.npy')[i]]]
                                   for name in ['sft','fraction_weak_endpoint','balanced_5step_endpoint']}})
    dump(reports/'examples.json',examples)
    source=data['source'];dump(reports/'source_manifest.json',source)
    sft=mean.loc['sft'];weak=mean.loc['fraction_weak_endpoint'];strong=mean.loc['fraction_kl_endpoint']
    best_selected=max([v['name']+'_selected' for v in config['variants']],key=lambda k:mean.loc[k,'ndcg10'])
    gain=100*(weak.pred_fraction/sft.pred_fraction-1)
    relevance=100*(weak.ndcg10/sft.ndcg10-1)
    selected_fraction=mean.loc['fraction_kl_selected']
    selected_time=mean.loc['time_kl_selected']
    quality_mean=np.mean([quality[str(s)]['optimizer']['reward_model']['fraction_rmse'] for s in config['seeds']])
    item_quality=quality[str(config['seeds'][0])]['optimizer']['item_mean']['fraction_rmse']
    mode_rows=[]
    for seed in config['seeds']:
        top=np.load(ROOT/f'artifacts/seed_{seed}/fraction_weak_endpoint_top10.npy')
        unique,counts=np.unique(top[:,0],return_counts=True);position=counts.argmax();token=unique[position]
        mode_rows.append({'Seed':seed,'Частый top-1 ID':int(catalog['item_id'][token]),
                          'Доля top-1':counts[position]/len(top),'Длина, с':float(catalog['duration'][token]),
                          'Train-событий':int(catalog['popularity'][token]),
                          'Train-средняя доля (smooth)':float(catalog['item_fraction'][token])})
    weak_fifth=np.mean([rollouts[f'{s}/fraction_weak_endpoint'][-1]['repeat_generated'] for s in config['seeds']])
    reward_table=md(pd.DataFrame(reward_rows))
    sizes=pd.DataFrame([{'Срез':k,'Событий для оценки/обучения':v['examples'],'Пользователей':v['users'],
                        'Цели в каталоге':v['target_catalog_coverage']} for k,v in data['splits'].items()])
    text=f'''# RL-дообучение последовательной рекомендательной модели

## Результат эксперимента

Обучена небольшая GPT-2 с нуля на идентификаторах треков Yambda, построены
модель отклика и отдельный аудитор, выполнены четыре варианта RL на двух seed.
Все числа ниже получены выполнением кода в этой папке. Это вычислительно
ограниченный offline-эксперимент, а не репликация промышленных результатов статей.

На последнем шаге варианта «доля, KL=0.01» ожидаемая доля по оптимизируемой RM
изменилась на **{gain:+.2f}%**, NDCG@10 — на **{relevance:+.2f}%** относительно SFT.
Они измеряют разные вещи: первое — прогноз обученной модели на выбранных
рекомендациях, второе — совпадение с реальными отложенными следующими событиями.
Прирост RM нельзя интерпретировать как наблюдаемый прирост удовлетворённости.

При сильном KL выбранная модель увеличила ожидаемую долю на
**{100*(selected_fraction.pred_fraction/sft.pred_fraction-1):.2f}%**, средний NDCG — на
{100*(selected_fraction.ndcg10/sft.ndcg10-1):.2f}%. Парные интервалы ΔNDCG включают
ноль на обоих seed, поэтому **значимое улучшение релевантности не обнаружено**.
Это локальный компромисс между прогнозируемой наградой и точностью, без
подтверждения реального выигрыша пользователей.

Основная таблица — checkpoint, выбранный **только на validation** с ограничением
NDCG@10 ≥ 95% от SFT. Если все RL-checkpoint нарушают ограничение или не улучшают
целевое вознаграждение, выбран шаг 0. Он тождествен SFT и не является успешным RL.
Усреднение двух seed описательное: test-данные у них общие.

{md(main_table)}

Все значения доли — в [0,1]. «RM: секунды» = E_pi[предсказанная доля × train median
длительности], без 600-секундного cap; в самой функции оптимизации времени cap
есть. Поэтому этот столбец может отличаться от целевой функции `time` × 300.
Аудиторская модель обучалась отдельно; её согласие с RM не доказывает верность.
Популярность и «последний + популярность» имеют детерминированное top-1 для
ожидаемой награды, тогда как GPT-2 оценивается по полному распределению. Их
reward/KL сравнения следует читать с учётом этого различия; ranking протокол общий.

## Данные и защита от утечек

Источник — [Yambda](https://huggingface.co/datasets/yandex/yambda), файл
`flat/50m/listens.parquet`, ревизия `{source['revision']}`.
SHA256: `{source['sha256']}`. Размер {source['bytes']:,} байт.
Полный файл содержит {data['raw_rows']:,} событий. Детерминированный хеш ID
выбирает примерно 10% пользователей: **{data['selected_users']:,} пользователей,
{data['selected_rows']:,} событий**; все события выбранных пользователей сохранены.
Это подвыборка пользователей, а не обучение на полном Yambda-50M.

Время Yambda выражено в секундах относительно общего начала, округлённых до
пяти секунд: floor((T−T_start)/5)×5, согласно
[разделу 3.3 статьи о данных](https://arxiv.org/html/2505.22238v2#S3.SS3).
Границы по перцентилям времени выбранных пользователей: train < {data['train_end_exclusive']},
validation [{data['train_end_exclusive']}, {data['validation_end_exclusive']}), test ≥ {data['validation_end_exclusive']}.
Реальные календарные даты из этих индексов не выводятся. Для continuation
из каждого среза исключены последние 1800 секунд, то есть 30 минут.

{md(sizes)}

Словарь — {data['catalog_items']} наиболее частых треков **по train** плюс PAD=0
и UNK=1. Доля train-событий, покрытая каталогом: {data['train_catalog_event_coverage']:.2%}.
Неизвестные треки сохраняются в последовательностях как UNK; они не становятся
действиями policy. OOV-цель остаётся в denominator Recall/NDCG как промах.
Этот выбор ограничивает абсолютное качество и перенос выводов на длинный хвост.
Никакой test-item не добавляется в каталог или список кандидатов по ответу.
Скоринг — полный каталог, без сэмплированных отрицательных при оценке.

Каждая цель имеет минимум три предшествующих события. Контекст — последние
24 события с **timestamp строго меньше целевого**, включая неизвестные.
События с одинаковым timestamp не упорядочиваются искусственно и не попадают
друг к другу в историю. Входы не содержат долю прослушанного целевого трека.
Статистики и медианные длины треков вычислены только по train.

Validation/test содержат случайно до восьми целей на пользователя, максимум
4000 событий на срез. Сначала метрики усредняются по событиям пользователя,
затем по пользователям; активные пользователи не получают больший вес.
Это rolling next-event evaluation: для позднего test-события разрешены реальные
более ранние test-события в истории. Обучение на этих событиях не выполняется.
Исключённые холодные контексты и пользователи без событий не покрыты оценкой.

`played_ratio_pct` >100 встречается {data['played_ratio_gt100']:,} раз: replay
частей трека допускается описанием датасета. Здесь целевая доля клиппируется
к 100%; информация о прослушивании сверх одной длины теряется намеренно.
Organic и recommendation-driven события смешаны; это дополнительное ограничение
для причинных выводов о рекомендательном воздействии.

## Базовая GPT-2

Используется настоящий `transformers.GPT2Model`, случайная инициализация,
2 decoder-блока, hidden size 64, 4 attention heads, learned position embeddings,
pre-layernorm, GELU и causal self-attention. Выходная матрица связана с item
embeddings, добавлен bias. Всего **{fit['parameters']:,} параметров**.
Это малая модель на архитектуре GPT-2, не предобученная текстовая GPT-2.

Из train-контекстов случайно выбираются до 80 000 известных целей.
Оптимизируется full-catalog next-item cross entropy на последней позиции,
пять эпох AdamW, lr=0.001, batch=128, weight decay=0.01, gradient clip=1.
Dropout отключён, чтобы распределения sampling и gradient evaluation совпадали
на RL-этапе. Эпоха выбирается по validation NDCG@10 отдельно для каждого seed.
Веса SFT не переобучаются после выбора. Таблицы эпох доступны в reports.
На обоих seed выбрана пятая, последняя доступная эпоха; validation-качество
ещё росло. Поэтому baseline ограничен вычислительным бюджетом, сходимость
и оптимальность длительности pretraining не доказаны.

Метрики релевантности определены для одного следующего зарегистрированного
объекта: Recall@10 = 1 при попадании; NDCG@10 = 1/log2(rank+1);
MRR@10 = 1/rank, иначе ноль. Это предсказание следующего события, включая
неполные прослушивания; оно не равно точности предсказания лайка или хорошего
пользовательского опыта. Повтор ранее прослушанного трека разрешён всем методам:
в музыке повтор сам по себе не означает ошибку. Доля повторов измеряется отдельно.

## Модель вознаграждения

RM получает среднее embedding последних 24 item ID и embedding последнего
трека, пропускает их через MLP; два bilinear head с item embeddings и item bias
предсказывают долю и вероятность следующего зарегистрированного прослушивания
не позднее 30 минут. Item bias инициализирован сглаженными train item means
с 20 псевдонаблюдениями. Обе цели обучаются совместной MSE; для бинарного
выхода это Brier score. Восемь эпох AdamW, lr=0.003, batch=256;
выбор по средней validation MSE двух выходов.

До 160 000 train-целей делятся по чётности индекса на два непересекающихся
набора строк: половина для оптимизируемой RM, половина для аудитора.
Они используют разные seed, не делят веса, но имеют общий train item-mean prior
и пересекающихся пользователей. Поэтому аудитор — коррелированная диагностика
экстраполяции, не независимая истинная награда и не отдельный causal estimator.

Качество на **реальных известных test-событиях**, до выбора policy-действия:

{reward_table}

`train_mean` — константа по train-событиям; `item_mean` — сглаженные train-средние;
`shuffled_history` — та же RM с контекстами, случайно переставленными между
test-примерами. Контроль показывает, насколько RM использует соответствие
истории пользователю; хорошие агрегатные ошибки ещё не гарантируют точность
на рекомендациях, которых не было в логах. Калибровка по децилям и результаты
аудитора находятся в `reports/reward_quality.json`.

RM оценивает реакцию **условно на записанное событие прослушивания**. Без
экспозиций, непрослушанных показов и propensity нельзя считать её прогнозом
полной вероятности отклика на произвольный показ. Во время RL эта условная
модель экстраполируется на весь каталог, что и создаёт риск reward hacking.

Фактически RM почти не превосходит item-only контроль: средний RMSE доли
{quality_mean:.5f} против {item_quality:.5f} у сглаженного item mean, то есть немного
хуже. Перестановка истории не даёт устойчивого ухудшения: для seed=42 ошибка
даже чуть ниже, для 2026 — чуть выше. Это слабое свидетельство персонализации
RM: основную предсказательную силу обеспечивают item statistics. Все эти ошибки
посчитаны по {quality[str(config['seeds'][0])]['optimizer']['reward_model']['n']} известным
test-событиям и усредняют события, а не пользователей.
Continuation AUC около 0.58, против 0.50 у константы: сигнал слабый.
В обеих RM лучшая validation-эпоха ранняя; дальнейшее уменьшение train loss
сопровождается ростом validation loss. RL оптимизирует ранний выбранный checkpoint.

Согласие аудитора со слабым RL особенно показательно: средняя аудиторская
награда {weak.audit_fraction:.5f} тоже намного выше исходной {sft.audit_fraction:.5f},
но релевантность почти исчезает. Это надёжно показывает несоответствие
оптимизируемой цели ranking-качеству; **эксплуатация именно ошибок RM не доказана**.
Две модели могут одинаково правильно предсказывать высокий completion
определённых треков и при этом поощрять неперсональные рекомендации.

## RL: алгоритм и функции награды

Используется on-policy REINFORCE внутри замороженного симулятора RM.
State — история item ID, action — трек полного каталога. Actions сэмплируются
из текущей policy, без top-k truncation и temperature scaling. Реальных новых
откликов нет: это model-based offline fine-tuning, а не on-policy эксперимент
с пользователями и не off-policy REINFORCE по логам.

Для одного шага loss = −(R(s,a)−b(s)) log π(a|s) + β KL(π||π_SFT),
где b(s)=Σ_a π(a|s)R(s,a) вычисляется точно и отсоединяется от gradient.
KL также вычисляется точно по всему каталогу, reference SFT заморожен.
Для пяти шагов используется reward-to-go с gamma=0.95 и множителем gamma^t;
из каждого return вычитается state-only одношаговый baseline. Такой baseline
не смещает REINFORCE, но оставляет дисперсию будущих наград. KL добавляется
как средний локальный regularizer по посещённым состояниям; полный градиент
влияния ранних действий на распределение будущего KL отдельно не оценивается.

Один rollout используется для одного AdamW update, без повторного переигрывания
старых действий, clipping ratios и critic. 300 обновлений, batch 64,
lr=0.0001, weight decay=0, gradient clip=1. Все варианты стартуют от одной
SFT-модели внутри seed. Размеры и правила заданы до просмотра test в
[протоколе](reports/PROTOCOL.md) и `config.json`.

Четыре постановки:

1. **Доля, слабый KL:** R=fraction, β=0.01, H=1 — проверка агрессивной оптимизации RM.
2. **Доля, сильный KL:** R=fraction, β=0.2, H=1 — ограничение ухода от исходной модели.
3. **Время:** R=fraction × min(train median duration,600)/300, β=0.2, H=1.
   Cap снижает стимул рекомендовать аномально длинные треки, но не решает вопрос релевантности.
4. **Смесь, пять шагов:** R=0.5 fraction + 0.25 time + 0.25 continuation,
   β=0.2, H=5. Это иллюстративная scalarization, коэффициенты не подбирались по test.

Ограничение NDCG проверяется на шагах 0/100/200/300. Из допустимых checkpoint
выбирается максимальное **собственное одношаговое E_pi[R]** варианта на реальной
validation-истории. Для H=5 это также одношаговый surrogate отбора, не оценка
полного пятишагового return; это ограничение данного эксперимента.
Между разными функциями значения objective напрямую несопоставимы.

{md(pd.DataFrame(choices))}

## Изменения поведения и компромисс с релевантностью

Следующая таблица использует **последнюю итерацию 300**, даже когда она
отклонена validation-ограничением. Это stress endpoint, не итоговая выбранная
модель. Она позволяет отделить эффект самого RL от возврата к SFT при выборе.

{md(end_table)}

Охват — число разных треков в test top-10 / размер обученного каталога.
Max top-1 — доля самого часто встречающегося первого трека среди test-событий.
Повтор@10 — доля рекомендаций, уже присутствующих в последних 24 событиях.
Охват и max top-1 считаются по событиям; остальные средние — по пользователям.
Полная таблица дополнительно содержит entropy, concentration HHI, mean log
train popularity, MRR@10, пересечение top-10 с SFT и оценку continuation.

У слабого KL финальный NDCG@10={weak.ndcg10:.5f}, у сильного
KL={strong.ndcg10:.5f}, у SFT={sft.ndcg10:.5f}. Сравнивать их нужно вместе
с вознаграждением и индивидуальными seed. Рост награды при снижении охвата
или концентрации на небольшом наборе треков — признак оптимизации RM,
а не доказательство персонализации. Изменение средней длины помогает
проверить гипотезу о предпочтении коротких объектов; этот эффект не следует
предполагать заранее, он может отсутствовать либо иметь обратный знак.

Здесь collapse наблюдается явно: охват top-10 упал с
{sft.catalog_coverage_top10:.2%} до {weak.catalog_coverage_top10:.2%}, энтропия — с
{sft.entropy:.3f} до {weak.entropy:.3f} нат, доля самого частого top-1 выросла с
{sft.top1_max_share:.2%} до {weak.top1_max_share:.2%}. Почти всем пользователям
ставится первым один трек; у разных seed выбраны разные треки с высокими
train completion rates:

{md(pd.DataFrame(mode_rows))}

Гипотеза «модель выбирает короткие треки» для слабого KL **не подтвердилась**:
ожидаемая длина увеличилась с {sft.duration_seconds:.1f} до {weak.duration_seconds:.1f} с.
Повторы из реальных последних 24 событий тоже не стали основной проблемой:
их доля в top-10 снизилась с {sft.recent_repeat_top10:.2%} до {weak.recent_repeat_top10:.2%}.
Нежелательный эффект — общий неперсональный выбор и повторы уже внутри
искусственно генерируемого списка, а не возврат к знакомым реальным трекам.

Сильный fraction KL немного уменьшил ожидаемую длину до
{selected_fraction.duration_seconds:.1f} с при сохранении охвата около
{selected_fraction.catalog_coverage_top10:.2%}. Оптимизация времени увеличила
ожидаемую длину до {selected_time.duration_seconds:.1f} с и прогнозируемые секунды
на {100*(selected_time.pred_seconds/sft.pred_seconds-1):.2f}%, но средний NDCG снизился
на {100*(1-selected_time.ndcg10/sft.ndcg10):.2f}%. У time_kl seed=2026 интервал
ΔNDCG целиком ниже нуля, хотя checkpoint прошёл validation-порог.
Таким образом, возможность увеличить модельную награду с небольшой потерей
ranking-точности зависит и от reward, и от регуляризации; гарантии улучшения нет.
Одинаковое β не означает одинаковую эффективную силу ограничения при разных
масштабах reward. Здесь β и коэффициенты смеси не объявляются оптимальными.

![Компромисс награды и релевантности](reports/figures/tradeoff.png)

Для оценки неопределённости применяется парный bootstrap пользователей,
2000 повторов, отдельно для каждого seed. Внутри bootstrap берутся все
выбранные события пользователя. Интервалы описывают вариативность test-пользователей
в одном временном периоде, не вариативность датасетов, будущих периодов
или всего процесса обучения. Примеров одного пользователя недостаточно
для независимого event-wise bootstrap.

{md(pd.DataFrame(ci))}

Только положительный интервал изменения релевантности поддерживает вывод
о статистически различимом улучшении на этом срезе. Интервал, включающий
ноль, не доказывает эквивалентность. Ограничение 95% на validation также не
гарантирует такую же потерю на test. Сравнение четырёх вариантов разведочное,
поправка на множественные сравнения не вводилась; победитель по test не выбирается.

## Генерация на пять шагов и сдвиг контекста

Для SFT и всех RL-endpoints сэмплируются пять треков, каждый добавляется
в контекст как будто он прослушан. Используются одни и те же случайно выбранные до 1024
test-контекстов и одинаковые seeds генератора. Поднабор фиксирован отдельным
seed и одинаков для всех моделей и обоих training seed. RM заново
оценивает реакцию на каждом синтетическом состоянии.

Параллельно **тот же сэмплированный трек** оценивается RM с исходной замороженной
историей. Разность synthetic/frozen показывает чувствительность RM к допущению
«рекомендованное = прослушанное». Это не второй реальный rollout с отказами,
и его нельзя назвать истинным откликом пользователя.

![Пятишаговая генерация](reports/figures/rollout.png)

Дисконтированные пятишаговые суммы в том же симуляторе, сначала по траектории,
затем по пользователю и по двум seed. Все столбцы сравнимы между моделями,
но являются модельными оценками; аудиторский return использует только долю.

{md(trajectory_table)}

В `reports/rollout_metrics.json` есть показатели по каждому шагу: награда RM
и аудитора, frozen-context reward, повторы внутри сгенерированной последовательности,
повторы из реальной истории, KL, NLL сэмплированного действия исходной SFT
на синтетической и исходной истории. NLL — дополнительный индикатор сдвига,
а не оценка причинной правдоподобности пользовательской траектории.
На шаге 1 synthetic и frozen reward совпадают по построению; дальнейшие
различия порождаются только изменением контекста.

На пятом шаге слабого RL повтор одного из ранее сгенерированных треков
встречается в среднем в **{weak_fifth:.2%}** траекторий. Для seed=42 NLL этого
действия у исходной SFT падает с 13.76 на первом шаге до 0.16 на пятом,
хотя NLL того же пятого действия на исходной frozen-истории остаётся 13.86.
Модель сама создаёт историю, в которой повтор выглядит правдоподобным.
Разность RM на synthetic/frozen контексте составляет около одного процентного
пункта, но сдвиг reference propensity гораздо сильнее. Поэтому одной проверки
стабильности прогнозов RM недостаточно для контроля многошаговой генерации.

Смесь H=5 не показывает убедительного преимущества над сильным одношаговым
RL: её mixed return в симуляторе ниже, чем у fraction_kl endpoint, при близкой
релевантности. Это не чистая абляция горизонта — одновременно отличается
функция награды. Чтобы приписать различие H, нужны H=1 и H=5 при одной и той
же награде, бюджете и процедуре отбора на новом validation-протоколе.

Технически это модель последовательного построения списка рекомендаций,
а не полноценная модель переходов состояния пользователя. Для реалистичного
MDP понадобятся отдельные вероятности принятия/пропуска/выхода и обновление
истории только по смоделированному отклику. Следует обучить переходы на логах,
проверить их на настоящих многошаговых траекториях и учитывать неопределённость.
Пятишаговый return текущего симулятора не доказывает долгосрочное улучшение.

Примеры с настоящими item ID, реальными целями и top-10 сохранены в
`reports/examples.json`. Названия песен не восстанавливаются: используемый
файл содержит анонимные идентификаторы. Полные сэмплированные траектории —
в `artifacts/seed_*/..._rollouts.csv`.

## Что можно сказать о долгосрочных целях

Сигнал «ещё одно событие в течение 30 минут» доступен из временных меток и
проверяется на holdout (Brier/AUC выше). Он коррелирует с продолжением сессии,
но не отделяет интерес пользователя от его свободного времени, автопроигрывания,
системы показа и позиции трека. Высокая вероятность продолжения сама по себе
не означает возвращение завтра или удовлетворённость.

Более содержательная постановка: дисконтированная сумма качественных
прослушиваний в сессии, затем отдельный бонус за возвращение через 1/7 дней,
со штрафом за быстрый пропуск/дизлайк и ограничением повторов/концентрации.
Возврат нужно задавать относительно конца сессии, применять одинаковое
окно наблюдения и исключать right-censored пользователей. Положительный
бонус следует ограничить, чтобы система не максимизировала бесконечное
прослушивание в ущерб выполнению задачи пользователя.

В расширении доступны likes/dislikes и is_organic; полезно разделить типы
трафика, анализировать музыкальные сессии и проверять calibration по группам.
Длительность сессии и число событий могут отражать причинно разные процессы,
поэтому веса reward стоит задавать с продуктовой гипотезой, а не выбирать
по максимальному simulator return.

Офлайн можно проверить предсказуемость таких меток во времени, temporal
generalization reward model, чувствительность к reward/β/горизонту,
support и uncertainty, повторы и концентрацию. При известных propensity
и достаточном overlap можно исследовать IPS/DR с effective sample size и
анализом clipping. Здесь propensity не предоставлены, IPS/DR не заявляются.
Для влияния рекомендаций на реальное возвращение и удовлетворённость нужен
онлайн A/B-тест с удержанием, качественными сигналами и ограничениями
по разнообразию/жалобам, а также период наблюдения, покрывающий целевой горизонт.

## Связь с предложенными работами

[Petrov & Macdonald, 2024](https://arxiv.org/abs/2403.04875) рассматривают
двухэтапное обучение GPTRec и Next-K generation для beyond-accuracy целей,
включая разнообразие и popularity bias. Здесь заимствована идея отделить
next-item pretraining от RL-выравнивания; teacher-student stage и PPO
этой работы не воспроизводятся.

[Matveev et al., 2026](https://arxiv.org/abs/2607.02818) используют session-level
off-policy REINFORCE, многошаговую поправку importance weights и user feedback
model для оценки. Здесь используется GPT-2 и on-policy sampling из симулятора,
без их off-policy correction, two-tower retrieval и doubly robust estimator.
Работа мотивирует явно обсуждать переходы состояния и достоверность offline
return, но её численные результаты к нашему эксперименту не относятся.

## Ограничения и следующие эксперименты

Размер модели, train subsampling, один временной срез, ограниченный каталог
и только два seed не позволяют делать универсальный вывод об эффективности RL
в рекомендательных системах. В исходном Yambda пользователи отбирались с
условием активности в конце периода наблюдения (статья о данных, §3.3);
это survivorship bias, особенно существенный для изучения ухода из сервиса.
RM обучена на условно наблюдаемых событиях;
её ошибки на действиях новой policy могут быть больше test RMSE.
Коррелированный аудитор обнаруживает лишь часть расхождений. Одинаковая
история у пользователей с разными неучтёнными обстоятельствами не задаёт
одинаковые потенциальные отклики.

Практические продолжения: расширить каталог/бюджет и проверить SFT saturation;
повторить на новых temporal folds; обучить ансамбль RM с uncertainty penalty;
добавить support-constrained generation; сравнить RL с простым reward reranking
и continued SFT при одинаковом бюджете; перейти к feedback-aware transitions;
подобрать β на заранее заданной validation Pareto curve; провести онлайн-проверку.
Эти эксперименты здесь не выполнены и не выдаются за результаты.

## Воспроизведение и артефакты

Полные команды в [README](README.md). Среда выполнения: Python {fit['python']},
PyTorch {fit['torch']}, transformers {fit['transformers']}, CPU, четыре потока.
Обучение заняло {fit['seconds']/60:.1f} минут без загрузки/подготовки/финальной оценки.
Исходные веса, optimizer state не нужны для оценки: все выбранные и финальные
state_dict сохранены. Для переобучения используются сохранённые конфигурации,
ревизия данных, seed, таблицы выбора и SHA256 файлов кода в fit_manifest.
Побитовое совпадение на иной ОС/версии библиотек не гарантируется.

В текущей среде восемь unit-тестов прошли. Независимый повтор загрузки весов
и пересчёта воспроизвёл метрики и top-10 для 10/10 моделей (SFT и четыре
selected-варианта на каждом seed). Максимальное отличие агрегатных метрик
от CSV — менее 3×10⁻¹⁴; результаты — `reports/reproduction_checks.json`.

Основные результаты: `reports/test_metrics.csv`, `mean_metrics.csv`,
`paired_intervals.json`, `reward_quality.json`, `rollout_metrics.json`,
`selection.json`, `data_checks.json`, `fit_manifest.json`. Обучающие и validation
кривые сохранены отдельно по seed. Tests проверяют строгий timestamp cutoff,
OOV denominator, train-only catalog/prior, padding/causality GPT-2, переход
контекста, согласованность RM scoring и ожидаемый градиент REINFORCE.
'''
    (ROOT/'REPORT.md').write_text(text)
    readme=f'''# GPT-2 + RL для рекомендаций Yambda

Завершённый offline-эксперимент: next-item GPT-2, reward model, четыре варианта
REINFORCE, два seed, временной holdout и анализ пятишаговой генерации.

**[Полный отчёт](REPORT.md)** · [Протокол](reports/PROTOCOL.md) · [Метрики](reports/test_metrics.csv).

| Метод | NDCG@10 | Recall@10 | Модельная доля |
|---|---:|---:|---:|
| GPT-2 SFT | {sft.ndcg10:.5f} | {sft.recall10:.5f} | {sft.pred_fraction:.5f} |
| RL доля, слабый KL, последний шаг | {weak.ndcg10:.5f} | {weak.recall10:.5f} | {weak.pred_fraction:.5f} |
| RL доля, сильный KL, последний шаг | {strong.ndcg10:.5f} | {strong.recall10:.5f} | {strong.pred_fraction:.5f} |

Среднее двух seed на одном test. Таблица показывает endpoint, включая
checkpoints, отклонённые ограничением по validation. Выбранные модели и
интервалы — в отчёте. Модельная награда не равна реальному эффекту на пользователей.

## Запуск из корня исходного репозитория

```bash
python3 -m venv rl_recsys/.venv
source rl_recsys/.venv/bin/activate
python -m pip install -r rl_recsys/requirements.txt
python -m unittest discover -s rl_recsys/tests -v
python -m rl_recsys.download_ranges
python -m rl_recsys.experiment prepare
python -m rl_recsys.experiment fit
python -m rl_recsys.experiment evaluate
python -m rl_recsys.report
python -m rl_recsys.verify
```

Если папка получена отдельно из ZIP, расположите её как `rl_recsys` внутри
любого рабочего каталога и запускайте команды из родительской папки.
Скачивается только открытый `flat/50m/listens.parquet`, {source['bytes']/1e6:.1f} MB;
range downloader проверяет SHA256 по `reports/source_manifest.json`.
Обычный downloader также доступен как `python -m rl_recsys.experiment download`.
Результаты и пути привязаны к расположению пакета, а не к текущей директории.

Параметры — `config.json`. При изменении настроек сохраните исходные результаты
и используйте отдельную копию папки. `prepare`/`fit` перезаписывают свои артефакты.
`evaluate` читает сохранённые веса и test-контексты, ничего не дообучает.
В архиве включены веса, каталог и test-контексты для оценки без повторной
загрузки данных; полный train-cache и исходный Parquet не включены.

Посмотреть рекомендации на сохранённом реальном контексте:

```bash
python -m rl_recsys.recommend --model sft --example-index 0
python -m rl_recsys.recommend --model fraction_kl_selected --example-index 0 --save recommendations.csv
```

## Структура

- `data.py`: временные срезы, train-only словарь и статистики, контексты.
- `models.py`: GPT2Model, reward model, награды, обновление истории.
- `experiment.py`: обучение, validation selection, оценка, bootstrap, rollout.
- `report.py`: таблицы, иллюстрации и отчёт из сохранённых результатов.
- `recommend.py`: просмотр top-10 по реальным сохранённым контекстам.
- `tests/`: проверки утечек, причинной маски и градиента REINFORCE.
- `reports/`: результаты, конфигурации выбора, источники, примеры, графики.
- `artifacts/`: локальные state_dict, контексты, пользовательские метрики и траектории.

Использована подвыборка {data['selected_users']} пользователей и
{data['selected_rows']:,} событий Yambda-50M. Каталог {data['catalog_items']} треков
из train, OOV-цели считаются промахами. Есть logged-support bias, условная RM,
синтетические переходы без модели принятия и один test-период. Доказательств
роста реального retention нет. Подробности и содержательные отрицательные
результаты приведены в отчёте.
'''
    (ROOT/'README.md').write_text(readme)
    print('Generated',ROOT/'REPORT.md',flush=True)


if __name__=='__main__': main()
