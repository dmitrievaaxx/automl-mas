# AutoML Pipeline Guide  

.\.venv\Scripts\activate

## Как обрабатывать любой датасет

1. **Структура**  
   Поместите `train.csv` в `data/raw/<dataset_name>/`

2. **DataAgent**  
   ```bash
   python scripts/run_data_agent.py --dataset <dataset_name> --target <target_column>
   ```
   - Обработанные CSV → `data/processed/<dataset_name>/`
   - Метаданные → `reports/<dataset_name>_metadata.json`

3. **AutoML (LightAutoML)**  
   ```bash
   python scripts/run_automl.py --dataset <dataset_name>
   ```
   - Отчёт → `reports/<dataset_name>_automl.json`

4. **ResearchAgent**  
   ```bash
   python scripts/run_research_agent.py --dataset <dataset_name>
   ```
   - Исследование → `reports/<dataset_name>_research.json`

---

## Пример: Titanic

### 1. DataAgent
```bash
python scripts/run_data_agent.py --dataset titanic --target Survived
```
- [`reports/titanic_metadata.json`](reports/titanic_metadata.json)
- [`data/processed/titanic/train.csv`](data/processed/titanic/train.csv)

### 2. AutoML
```bash
python scripts/run_automl.py --dataset titanic
```
- [`reports/titanic_automl.json`](reports/titanic_automl.json)

```jsonc
{
  "dataset": "titanic",
  "automl": {
    "best_model": {
      "name": "5 averaged models Lvl_0_Pipe_1_Mod_2_CatBoost",
      "score": 0.8534914361001318
    },
    "test_metrics": {
      "auc": 0.8534914361001318
    }
  }
}
```

### 3. ResearchAgent
```bash
python scripts/run_research_agent.py --dataset titanic
```
- [`reports/titanic_research.json`](reports/titanic_research.json)

```jsonc
{
  "dataset": "titanic",
  "baseline": {
    "metric": "auc",
    "value": 0.8535
  },
  "recommendations": [
    {
      "title": "Random Forest Hyperparameter Tuning",
      "description": "Optimize hyperparameters via random search.",
      "source": "Tavily",
      "url": "https://jaketae.github.io/study/sklearn-pipeline/",
      "expected_gain": "+0.0165 AUC"
    }
  ]
}
```

---

### Быстрые команды
| Этап | Команда |
|------|---------|
| DataAgent | `python scripts/run_data_agent.py --dataset <name> --target <column>` |
| AutoML | `python scripts/run_automl.py --dataset <name>` |
| ResearchAgent | `python scripts/run_research_agent.py --dataset <name>` |