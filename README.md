## Проект: Сегментация человека

Этот проект предоставляет инструменты для сегментации человека с использованием различных архитектур глубокого обучения, таких как DeepLabv3+, U-Net и U^2-Net.

**Модули:**

* **Обработка данных:** Подготовка и аугментация наборов данных для сегментации человека.
* **Обучение моделей:** Обучение моделей DeepLabv3+, U-Net и U^2-Net на подготовленных наборах данных.
* **Валидация:** Оценка производительности обученных моделей.
* **Визуализация:** Визуализация результатов сегментации.
* **Документация:** Предоставление подробной документации по проекту и его API.

**Установка:**

1. **Создайте виртуальную среду:**
   ```bash
   make env
   ```
2. **Установите зависимости:**
   ```bash
   make dependencies
   ```

**Использование:**

**Важно:** Все скрипты проекта должны запускаться **внутри активированной виртуальной среды**. Для активации виртуальной среды используйте команду:
```bash
make shell
```

**Все параметры скриптов настраиваются в папке с конфигурацией Hydra (`/config/`).** 

**Запуск скриптов:**

* **Обработка данных:**
    ```bash
    make data_proc
    ```
* **Обучение:**
    ```bash
    make train
    ```
* **Валидация:**
    ```bash
    make valid
    ```
* **Визуализация:**
    ```bash
    make visual
    ```

**Документация:**

Документация по этому проекту находится в папке `docs`. Вы можете создать документацию с помощью:

```bash
make docs
```

**Лицензия:**

Этот проект лицензирован под лицензией MIT.
