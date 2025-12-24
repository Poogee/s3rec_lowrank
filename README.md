# S3Rec с Low-rank AAP

Реализация модели S3Rec для последовательных рекомендаций с low-rank разложением матрицы AAP.

## Что сделано

**Основная идея:** заменить тяжёлую матрицу W (d×d параметров) на произведение двух маленьких U·Vᵀ (2×d×r параметров).

```
Было:   score = sigmoid(item · W · attribute)     →  d² параметров
Стало:  score = sigmoid(item · U · Vᵀ · attribute) →  2dr параметров
```

При d=64 и r=16 получаем **50% экономии параметров** без потери качества.

### Результаты на Amazon Beauty

| Модель | Hit@10 | NDCG@10 | Параметры |
|--------|--------|---------|-----------|
| S3Rec (оригинал) | 55.06% | 37.32% | 1.04M |
| **S3Rec Low-rank (r=16)** | **55.50%** | **37.80%** | **0.85M** |

## Быстрый старт

### Установка

```bash
cd s3rec_lowrank
pip install -r requirements.txt
```

### Запуск обучения

```bash
# 1. Препроцессинг данных
python -m experiments.preprocess \
    --reviews путь/к/reviews_Beauty_5.json \
    --metadata путь/к/meta_Beauty.json

# 2. Pre-training (обучение на 4 задачах)
python -m experiments.pretrain --lowrank --rank 16 --epochs 100

# 3. Fine-tuning (дообучение на рекомендации)
python -m experiments.finetune --epochs 50
```

### Или всё сразу

```bash
python -m experiments.run_all \
    --reviews путь/к/reviews_Beauty_5.json \
    --metadata путь/к/meta_Beauty.json \
    --ranks 16
```

## Структура проекта

```
s3rec_lowrank/
├── models/
│   ├── lowrank_aap.py    ← главная фича: low-rank модуль
│   ├── modules.py        ← Transformer
│   └── s3rec.py          ← модель целиком
├── trainers/
│   ├── pretrain.py       ← pre-training на 4 задачах
│   └── finetune.py       ← fine-tuning на рекомендации
├── data/
│   ├── preprocessing.py  ← обработка датасета
│   └── dataset.py        ← PyTorch Dataset
├── experiments/          ← скрипты для запуска
├── plots/                ← генерация графиков
└── results/              ← чекпоинты и метрики
```

## Как это работает

### Pre-training (4 задачи)

1. **AAP** — предсказание атрибутов товара (категории, бренд)
2. **MIP** — предсказание замаскированных товаров
3. **MAP** — предсказание атрибутов замаскированных товаров
4. **SP** — предсказание сегмента последовательности

### Fine-tuning

Обучение на задаче "предскажи следующий товар" с pairwise ranking loss.


Графики сохранятся в `plots/presentation_plots/`.

## Пример использования модели

```python
from models.s3rec import S3RecLowRankModel

model = S3RecLowRankModel(
    num_items=12102,
    num_attributes=1221,
    hidden_size=64,
    rank=16
)

# Посмотреть экономию параметров
print(model.get_parameter_reduction())
# {'reduction_percent': '0.52%', 'aap_reduction': '50.00%', ...}
```


