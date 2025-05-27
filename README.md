# MNIST Digit Classification with CNN

Проект по классификации рукописных цифр из датасета MNIST с использованием сверточных нейронных сетей (CNN).

## Цель
Классифицировать изображения рукописных цифр (0–9) с помощью CNN.

## Результаты
- Accuracy: 0.994 на тестовом наборе
- Дополнительно: Recall, Precision, F1 по классам, Confusion Matrix (см. MNIST_results_analysis.ipynb)

## Структура
- MNIST.ipynb: Анализ данных (проверка диапазона значений, баланса классов, подсчет mean/std для нормализации, настройка поворотов и смещений).
- MNIST_mp.py: Обучение CNN.
- MNIST_results_analysis.ipynb: Визуализация метрик (accuracy, recall, precision, F1 по классам и эпохам, confusion matrix).
- Папки с датасетом и результатами не выгружены из-за объема.

## Применяемые библиотеки
- numpy
- torch, torchvision
- matplotlib
- seaborn
- sklearn
- стандартные библиотеки python

## Датасет
- Использован датасет MNIST (Modified National Institute of Standards and Technology database)
- Описание датасета: https://en.wikipedia.org/wiki/MNIST_database
- Можно скачать с помощью torchvision: 'train_data = torchvision.datasets.MNIST("./", train=True, download=True)' и 'test_data = torchvision.datasets.MNIST("./", train=False, download=True)'
