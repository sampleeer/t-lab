# GPT-2 + RL для рекомендаций Yambda

Завершённый offline-эксперимент: next-item GPT-2, reward model, четыре варианта
REINFORCE, два seed, временной holdout и анализ пятишаговой генерации.

**[Полный отчёт](REPORT.md)** · [Протокол](reports/PROTOCOL.md) · [Метрики](reports/test_metrics.csv).

| Метод | NDCG@10 | Recall@10 | Модельная доля |
|---|---:|---:|---:|
| GPT-2 SFT | 0.02243 | 0.03720 | 0.65966 |
| RL доля, слабый KL, последний шаг | 0.00130 | 0.00201 | 0.93972 |
| RL доля, сильный KL, последний шаг | 0.02297 | 0.03895 | 0.67417 |

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
Скачивается только открытый `flat/50m/listens.parquet`, 369.1 MB;
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

Использована подвыборка 916 пользователей и
4,484,526 событий Yambda-50M. Каталог 4096 треков
из train, OOV-цели считаются промахами. Есть logged-support bias, условная RM,
синтетические переходы без модели принятия и один test-период. Доказательств
роста реального retention нет. Подробности и содержательные отрицательные
результаты приведены в отчёте.
