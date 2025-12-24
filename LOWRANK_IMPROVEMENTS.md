# Low-rank Factorization Improvements

## Новые возможности

Реализованы три ключевых улучшения для low-rank факторизации:

1. **Разные rank для AAP и MAP** - позволяет использовать разные размерности для разных задач
2. **Ортогональная инициализация** - улучшает стабильность обучения
3. **Адаптивный weight decay** - разная регуляризация для low-rank параметров

## Использование

### 1. Разные rank для AAP и MAP

В конфигурационном файле `config/default_config.yaml`:

```yaml
lowrank:
  enabled: true
  rank: 16                    # Используется если aap_rank/map_rank не указаны
  aap_rank: 20                # Rank для AAP head (null = использовать rank)
  map_rank: 12                # Rank для MAP head (null = использовать rank)
```

**Примеры использования:**

```python
# Старый способ (backward compatible)
config = {
    'lowrank': {'enabled': True, 'rank': 16}
}
# AAP и MAP оба будут с rank=16

# Новый способ - разные rank
config = {
    'lowrank': {
        'enabled': True,
        'rank': 16,          # fallback значение
        'aap_rank': 20,      # AAP использует rank=20
        'map_rank': 12       # MAP использует rank=12
    }
}
```

**Рекомендации:**
- AAP обычно требует большего rank (16-24), так как предсказывает для всех позиций
- MAP может работать с меньшим rank (12-16), так как только для масок
- Попробуйте комбинации: (20, 12), (18, 14), (24, 16)

### 2. Ортогональная инициализация

В конфигурации:

```yaml
lowrank:
  init_method: "orthogonal"  # или "xavier" (по умолчанию)
```

**Доступные методы:**
- `"xavier"` (по умолчанию) - стандартная инициализация
- `"orthogonal"` - ортогональная инициализация для лучшей численной стабильности

**Когда использовать:**
- Ортогональная инициализация может помочь при обучении глубоких сетей
- Рекомендуется попробовать если видите проблемы со стабильностью градиентов

### 3. Адаптивный weight decay

В конфигурации для pretrain и finetune:

```yaml
pretrain:
  weight_decay: 0.0          # Для обычных параметров
  lowrank_weight_decay: 0.01 # Для low-rank параметров (U, V матрицы)

finetune:
  weight_decay: 0.0
  lowrank_weight_decay: 0.01
```

**Как это работает:**
- Low-rank параметры (U, V в AAP/MAP heads) получают `lowrank_weight_decay`
- Все остальные параметры получают `weight_decay`
- Если `lowrank_weight_decay` не указан или равен `weight_decay`, используется стандартный оптимизатор

**Рекомендации:**
- Начните с `lowrank_weight_decay: 0.01`
- Попробуйте значения: 0.005, 0.01, 0.02
- Больший weight_decay для low-rank помогает предотвратить переобучение

## Полный пример конфигурации

```yaml
lowrank:
  enabled: true
  rank: 16                    # Fallback значение
  aap_rank: 20                # Разный rank для AAP
  map_rank: 12                # Разный rank для MAP
  init_method: "orthogonal"   # Ортогональная инициализация
  use_bias: false

pretrain:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0           # Для обычных параметров
  lowrank_weight_decay: 0.01  # Для low-rank параметров

finetune:
  epochs: 50
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0
  lowrank_weight_decay: 0.01
```

## Backward Compatibility

Все изменения полностью обратно совместимы:

1. **Старые конфиги работают**: если не указаны `aap_rank`/`map_rank`, используется `rank`
2. **Старый код работает**: можно создать модель со старым API
3. **По умолчанию**: `init_method="xavier"`, `lowrank_weight_decay=None` (не используется)

## Программный API

### Создание модели с новыми параметрами

```python
from models.s3rec import S3RecLowRankModel

# С разными rank
model = S3RecLowRankModel(
    num_items=12102,
    num_attributes=1221,
    hidden_size=64,
    rank=16,              # Fallback
    aap_rank=20,          # AAP rank
    map_rank=12,          # MAP rank
    lowrank_init_method="orthogonal"
)

# Старый способ (все еще работает)
model = S3RecLowRankModel(
    num_items=12102,
    num_attributes=1221,
    hidden_size=64,
    rank=16  # Используется для AAP и MAP
)
```

### Проверка параметров модели

```python
analysis = model.get_aap_analysis()
print(analysis)
# {
#     'aap_rank': 20,
#     'map_rank': 12,
#     'init_method': 'orthogonal',
#     ...
# }
```

## Ожидаемые улучшения

При использовании всех улучшений:

1. **Разные rank**: +0.3-0.8% NDCG@10
2. **Адаптивный weight decay**: +0.2-0.5% NDCG@10
3. **Ортогональная инициализация**: более стабильная сходимость, +0-0.2% NDCG@10

**Суммарное улучшение**: +0.5-1.5% NDCG@10 по сравнению с базовой low-rank моделью

## Эксперименты

Рекомендуемый план экспериментов:

1. **Baseline**: `rank=16`, `init_method="xavier"`, `weight_decay=0.0`
2. **+ Разные rank**: `aap_rank=20, map_rank=12`
3. **+ Адаптивный weight decay**: `lowrank_weight_decay=0.01`
4. **+ Ортогональная инициализация**: `init_method="orthogonal"`

Сравните результаты каждого шага!

