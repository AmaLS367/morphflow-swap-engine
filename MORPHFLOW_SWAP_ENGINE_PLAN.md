# MorphFlow Swap Engine — План реализации

## Status Note

- As of 2026-03-07, the repository no longer carries the preserved `facefusion/` baseline.
- This is an explicit deviation from the original phased migration plan, executed by direct user request.
- The plan remains useful as architecture and sequencing context, but any instructions that require keeping the legacy runtime callable are now historical rather than operative.

## Цель

Собрать новый `morphflow-swap-engine` как замену текущему слабому ядру face swap.

Новый движок должен:
- перестать зависеть от дефолтного поведения FaceFusion
- использовать современный стек: детекция, трекинг, swap, восстановление, temporal
- быть заточен под сервер с RTX 5090, 32 GB VRAM, 31 GB RAM
- быть модульным — замена одной технологии не ломает весь движок
- интегрироваться обратно в MorphFlow через адаптер

---

## Почему форкаем FaceFusion

| Проблема | Причина |
|----------|---------|
| Лицо не меняется визуально | `inswapper_128` работает в 128px — слишком мало деталей |
| Детектор находит 4 лица вместо 1 | `RetinaFace` даёт ложные срабатывания на фоне и руках |
| Мерцание между кадрами | Нет temporal consistency — каждый кадр обрабатывается независимо |
| Слабый enhancer | `GFPGAN 1.4` устарел, артефакты на коже |
| Слабое использование GPU | Нет батчинга, нет fp16, нет torch.compile |

---

## Новый стек

| Компонент | Было | Стало |
|-----------|------|-------|
| Детектор | RetinaFace | InsightFace `buffalo_l` (SCRFD) |
| Swap-модель | inswapper_128 (128px) | Ghost (512px), SimSwap++ как fallback |
| Enhancer | GFPGAN 1.4 | CodeFormer (fidelity_weight=0.7) |
| Temporal | отсутствует | FILM (Frame Interpolation for Large Motion) |
| Precision | fp32, 1 кадр | fp16 + torch.compile + batch 32 кадра |
| VRAM | ~4GB | ~18-22GB (5090: 32GB) |

---

## Стратегическое решение

Не патчим куски старого пайплайна. Вместо этого:

1. Форкаем текущий репозиторий
2. Создаём отдельный модуль `morphflow-swap-engine`
3. Оставляем текущий API/pipeline shell где полезно
4. Поэтапно заменяем CV-стек на новые компоненты
5. Старый движок остаётся как fallback на время перехода

---

## Фазы реализации

### Фаза 0. Форк и фиксация baseline

Задачи:
- Форкнуть текущий MorphFlow
- Создать ветку `feature/swap-engine-foundation`
- Тегнуть текущее состояние как `baseline-facefusion-old`
- Не удалять старую реализацию — она нужна для сравнения

Результат:
- Форкнутый репо с чистой веткой
- Старый пайплайн всё ещё можно запустить

---

### Фаза 1. Скелет нового движка

Создать пакет:

```
src/morphflow_swap_engine/
    core/
        entities/
        value_objects/
        contracts/
        services/
    application/
        use_cases/
        orchestrators/
    infrastructure/
        detection/
        tracking/
        alignment/
        swapping/
        restoration/
        temporal/
        video/
        diagnostics/
        runtime/
    adapters/
        morphflow/
        cli/
    config/
    tests/
```

Правила:
- Один модуль = одна ответственность
- Без циклических зависимостей
- Сначала контракты, потом реализации
- Все обёртки моделей живут в infrastructure
- Старый MorphFlow вызывает новый движок только через адаптер

Контракты:
- `IFaceDetector`
- `IFaceTracker`
- `IFaceAligner`
- `IFaceSwapper`
- `IFaceRestorer`
- `ITemporalStabilizer`
- `IVideoDecoder`
- `IVideoEncoder`
- `IArtifactStore`

Entities / value objects:
- `ReferenceFaceAsset`
- `TargetVideoAsset`
- `DetectedFace`
- `TrackedFaceSequence`
- `SwapRequest`
- `SwapResult`
- `StageArtifact`
- `EngineProfile`
- `RuntimeReport`

Результат:
- Пустой но рабочий пакет
- Типизированные контракты
- Конфиг
- Адаптер-точка входа из текущего MorphFlow

Критерии:
- Пакет импортируется без ошибок
- Type checker проходит
- Нет хардкода моделей в application/core слоях

---

### Фаза 2. Замена детектора

Текущая проблема: `RetinaFace` слабый, ложные срабатывания.

Замена: **InsightFace buffalo_l** (SCRFD backbone)

```python
import insightface

app = insightface.app.FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
faces = app.get(frame)
```

Что должен уметь детектор:
- Детектировать лица от 8px (vs 20px у RetinaFace)
- Настраиваемый confidence threshold
- 5 ключевых точек + embedding для идентификации
- Batch-детекция по кадрам
- Выбор главного лица по размеру, confidence, центральности
- Фильтрация ложных срабатываний

Анализ референса:
- Размер изображения, количество лиц, primary face box, confidence, face size ratio, warnings

Анализ таргета:
- Сэмплированные кадры, детекции на кадр, face size ratios, warnings

Результат:
- Модуль детектора за `IFaceDetector`
- Модуль фильтрации
- Анализатор референса и таргета

---

### Фаза 3. Трекинг

Критическая фаза. Текущий FaceFusion обрабатывает каждый кадр как отдельную вселенную.

`IFaceTracker` должен:
- Связывать детекции между кадрами
- Находить доминантный трек целевого лица
- Держать стабильный выбор идентичности через весь клип
- Экспортировать кропы и метаданные трека

Выходные данные трека:
- Track ID
- Box на каждый кадр
- Landmarks на каждый кадр (если есть)
- История confidence
- Длина трека, количество пропущенных кадров
- Stability score

Политика выбора главного трека:
- Стабильность через кадры
- Средний размер лица
- Средний confidence
- Центральность

Результат:
- Модуль трекинга
- Скоринг треков
- Политика выбора целевого лица

---

### Фаза 4. Alignment и crop pipeline

Свопер работает лучше когда таргет и референс нормализованы.

Задачи:
- Выравнивание референса по landmarks
- Консистентные кропы таргета по трекам
- Нормализация размеров под ожидания свопера
- Настраиваемый margin кропа
- Сохранение достаточного контекста (лоб, челюсть)
- Поддержка crop-to-swap для мелких лиц в полном кадре

Результат:
- Модуль выравнивания
- Модуль crop-стратегии

---

### Фаза 5. Замена swap-модели

Текущая проблема: `inswapper_128` — 128px, мало деталей.

Замена: **Ghost (512px)** — основной, **SimSwap++** — fallback.

```python
# Batch processing на 5090
batch_size = 32  # 32GB позволяет
for batch in chunks(frames, batch_size):
    swapped = ghost_model.swap_batch(batch, source_embedding)
```

Профили:
- `balanced`
- `high_quality`
- `aggressive`

Каждый свопер-модуль:
- Config schema
- Warmup/load
- Inference
- Memory-safe batch
- Артефакты выхода

Требования:
- Своперы за `IFaceSwapper`
- Переключение профиля без изменения архитектуры
- Сохранение raw-кропов/фреймов до restoration

Результат:
- Primary swapper (Ghost)
- Fallback swapper (SimSwap++ или stub)
- Реестр профилей

---

### Фаза 6. Restoration

Текущая проблема: `GFPGAN 1.4` устарел, артефакты на коже.

Замена: **CodeFormer** (fidelity_weight=0.7)

```python
codeformer.enhance(
    face_img,
    fidelity_weight=0.7,
    has_aligned=False,
    only_center_face=True
)
```

Профили:
- `off` — отключено
- `standard` — fidelity 0.7
- `high_quality` — fidelity 0.5

Правила:
- Restoration опциональна
- Можно отключить глобально для отладки
- Не должна over-smoothить или возвращать identity таргета

Результат:
- Restoration за `IFaceRestorer`
- Профильный конфиг
- ON/OFF поддержка

---

### Фаза 7. Temporal stabilization

Текущая проблема: полностью отсутствует. Мерцание между кадрами — алгоритм TikTok это детектирует.

Замена: **FILM** (Frame Interpolation for Large Motion)

Задачи:
- Сглаживание frame-to-frame вариаций
- Уменьшение identity drift
- Уменьшение текстурного мерцания
- Работает после swap или после restore в зависимости от профиля

Правила:
- Модуль изолирован
- Включается/выключается по профилю

Результат:
- Модуль temporal за `ITemporalStabilizer`
- Интеграция с профилями

---

### Фаза 8. Реконструкция видео

На этом этапе движок должен поддерживать:
- swap-only выход
- swap + restore
- swap + restore + temporal
- опциональные downstream hooks

Требования к видео-слою:
- Робастный decode кадров
- Детерминированный порядок кадров
- Сохранение аудио
- Качественный encode с настраиваемыми параметрами по профилю

Для crop-to-swap режима:
- Реинтеграция свопнутой области обратно в полный кадр
- Сохранение позиционирования
- Без видимых границ патча

Результат:
- Video decoder модуль
- Video encoder модуль
- Crop reintegration модуль

---

### Фаза 9. Оптимизация под RTX 5090

Цель: выжать железо, а не работать как на ноутбуке.

```python
# torch.compile — ускорение ~30% на Ampere+
model = torch.compile(ghost_model, mode='max-autotune')

# fp16 — в 2x быстрее, в 2x меньше VRAM
with torch.autocast(device_type='cuda', dtype=torch.float16):
    result = model(batch)

# Batch 32 кадра за раз вместо по одному
```

Оптимизации:
- fp16 где безопасно
- Batched inference
- Frame chunking
- Предзагрузка моделей
- Memory-aware пресеты
- torch.compile для стабильных путей

Профили:
- `balanced`
- `quality_max`
- `throughput_max`

Результат:
- GPU-aware runtime config
- Стабильная работа на 5090

---

### Фаза 10. Диагностика

Структура debug-артефактов:

```
storage/debug/<job_id>/
    metadata/
    logs/
    artifacts/
        01_detection/
        02_tracking/
        03_alignment/
        04_swap/
        05_restore/
        06_temporal/
        07_reconstruction/
```

Что записывать:
- Stage records
- Warnings
- Artifacts manifest
- Использованный профиль
- Выбранные модули (detector/tracker/swapper/restorer/temporal)
- Тайминг по стадиям
- Снепшот окружения и версий моделей

Результат:
- Artifact store
- Runtime report
- Debug manifest

---

### Фаза 11. Интеграция с MorphFlow

Правила:
- Фронтенд и API не переписываются
- Текущий flow вызывает новый движок через адаптер
- Старый движок остаётся за флагом на время перехода

Требования:
- Маппинг process request → new engine request
- Поддержка выбора профиля
- swap-only режим
- Диагностика пробрасывается в существующий API

Результат:
- MorphFlow адаптер
- Feature flag для выбора движка
- Migration notes

---

## Архитектура репозитория

```
repo/
    src/
        morphflow_swap_engine/
    docs/
        architecture/
        deployment/
    scripts/
        debug/
        deployment/
    tests/
        unit/
        integration/
```

---

## Порядок коммитов

1. `chore: fork baseline and tag legacy swap pipeline`
2. `feat: add morphflow swap engine package skeleton and core contracts`
3. `feat: add detector — InsightFace buffalo_l behind IFaceDetector`
4. `feat: add target face tracking and selection strategy`
5. `feat: add alignment and crop pipeline`
6. `feat: add Ghost swapper behind IFaceSwapper`
7. `feat: add swapper profile registry and SimSwap++ fallback`
8. `feat: add CodeFormer restoration layer with profile toggles`
9. `feat: add FILM temporal stabilization stage`
10. `feat: add video reconstruction and export pipeline`
11. `feat: add diagnostics manifest and debug artifact store`
12. `feat: add gpu runtime profiles for rtx 5090`
13. `feat: integrate new engine into morphflow through adapter`
14. `docs: add architecture and deployment documentation`

---

## Волны выполнения

### Волна 1 — Фундамент
Фазы 0, 1. Форк, скелет, контракты, конфиг, адаптер.

### Волна 2 — Детекция и трекинг
Фазы 2, 3, 4. Детектор, трекер, alignment/crop.

### Волна 3 — Swap и restoration
Фазы 5, 6. Ghost свопер, CodeFormer restoration.

### Волна 4 — Temporal и реконструкция
Фазы 7, 8. FILM stabilization, видео export.

### Волна 5 — Оптимизация и диагностика
Фазы 9, 10. GPU профили, debug артефакты.

### Волна 6 — Интеграция
Фаза 11. Адаптер в MorphFlow, feature flag, rollout.

---

## Правила для всех агентов

1. Не переписывать весь репо вслепую
2. Не мешать архитектуру и замену моделей в одном коммите
3. Каждая стадия должна оставлять проект в рабочем состоянии
4. Каждая модель/инструмент за чистым контрактом
5. Каждая стадия добавляет артефакты/логи для инспекции
6. Не удалять старый путь пока новый не победит
7. Код модульный и заменяемый
8. Локальные инструкции для Windows где нужно
9. Комментарии в коде только на английском

---

## Не-цели

Не тратить время сейчас на:
- Редизайн фронтенда
- Маркетинговые страницы
- Масштабный редизайн API не связанный с движком
- Рандомные рефакторы не связанные с миграцией swap engine

---

## Целевое состояние

По завершении плана должно быть:
- Новый `morphflow-swap-engine`
- Модульный современный CV-стек (InsightFace → Ghost → CodeFormer → FILM)
- Стабильный трекинг целевого лица
- Качество swap выше текущего baseline
- Контролируемые стадии restoration и temporal
- RTX 5090 оптимизированные runtime профили
- Чистая интеграция обратно в MorphFlow

Критерий: swap_verification delta > 0.1 (вместо текущих 0.018).
