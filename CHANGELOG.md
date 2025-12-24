# Changelog - Low-rank Improvements

## Новые возможности

### 1. Разные rank для AAP и MAP

**Изменения:**
- Добавлены параметры `aap_rank` и `map_rank` в `S3RecLowRankModel`
- Если не указаны, используется `rank` (backward compatible)
- Обновлен конфиг для поддержки `aap_rank` и `map_rank`

**Файлы:**
- `models/s3rec.py`: `S3RecLowRankModel.__init__()` принимает `aap_rank`, `map_rank`
- `models/s3rec.py`: `create_model()` читает из конфига
- `config/default_config.yaml`: добавлены `aap_rank`, `map_rank`

**Пример:**
```yaml
lowrank:
  rank: 16      # Fallback
  aap_rank: 20  # AAP head
  map_rank: 12  # MAP head
```

### 2. Ортогональная инициализация

**Изменения:**
- Добавлен параметр `init_method` в `LowRankAAP` ("xavier" или "orthogonal")
- По умолчанию "xavier" (backward compatible)
- Обновлен конфиг для поддержки `init_method`

**Файлы:**
- `models/lowrank_aap.py`: `LowRankAAP.__init__()` принимает `init_method`
- `models/s3rec.py`: передает `lowrank_init_method` в heads
- `config/default_config.yaml`: добавлен `init_method`

**Пример:**
```yaml
lowrank:
  init_method: "orthogonal"  # или "xavier"
```

### 3. Адаптивный weight decay

**Изменения:**
- Добавлен `lowrank_weight_decay` в конфиг для pretrain и finetune
- Trainers автоматически разделяют параметры на low-rank и остальные
- Если `lowrank_weight_decay` не указан, используется стандартный optimizer (backward compatible)

**Файлы:**
- `trainers/pretrain.py`: `_setup_optimizer()` поддерживает parameter groups
- `trainers/finetune.py`: `_setup_optimizer()` поддерживает parameter groups
- `config/default_config.yaml`: добавлен `lowrank_weight_decay` в pretrain и finetune

**Пример:**
```yaml
pretrain:
  weight_decay: 0.0          # Для обычных параметров
  lowrank_weight_decay: 0.01 # Для low-rank (U, V)
```

## Backward Compatibility

✅ Все изменения полностью обратно совместимы:

1. Старые конфиги работают (используются значения по умолчанию)
2. Старый API работает (если не указывать новые параметры)
3. По умолчанию поведение не меняется

## Измененные файлы

1. `models/lowrank_aap.py` - добавлен `init_method`
2. `models/s3rec.py` - добавлены `aap_rank`, `map_rank`, `lowrank_init_method`
3. `config/default_config.yaml` - добавлены новые параметры
4. `trainers/pretrain.py` - адаптивный weight decay
5. `trainers/finetune.py` - адаптивный weight decay

## Использование

См. `LOWRANK_IMPROVEMENTS.md` для подробной документации.

