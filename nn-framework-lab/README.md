# Создание своего нейросетевого фреймворка

Учебный фреймворк для создания и обучения нейронных сетей, позволяющий комбинировать в архитектуре различные слои, активаторы, оптимизаторы, функции потерь и метрики. Также имеется возможность визуализировать процесс обучения и результаты в виде html отчета, а еще можно поиграть в песочнице с визуальным конструктором нейросетей. 

## Видеопрезентация:
<video src="./assets/aura_nn_playground_preview.mp4" controls width="100%"></video>

**Моя роль:** разработчик функций активации, loss-функций, Web Playground

## Стек:
Python, NumPy, Streamlit, Matplotlib

## Реализовано:
* 5 функций активации (ReLU, Sigmoid, Tanh, Linear, Softmax) с прямым и обратным проходом и кэшированием для эффективного обучения.  
* 3 функции потерь (MSELoss, BCELoss, CrossEntropyLoss) для задач регрессии, бинарной и многоклассовой классификации.  
* Web Playground - визуальный NoCode-конструктор нейросетей на Streamlit с возможностью выбора датасета, настройки архитектуры, оптимизатора и визуализации обучения.

## Полная архитектура фреймворка:
```
nn-framework-aura_team/
├── nn/                          # ядро нейросети
│   ├── __init__.py
│   ├── activations.py           # функции активации (Linear, Sigmoid, Tanh, ReLU, Softmax)
│   ├── data.py                  # Dataset - загрузка и предобработка данных              
│   ├── layers.py                # слои (Dense, Conv2D, MaxPool2D, Dropout, Flatten)
│   ├── losses.py                # функции потерь (MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss)
│   ├── model.py                 # Sequential - контейнер для слоев
│   ├── optimizers.py            # оптимизаторы (SGD, MomentumSGD, Adam)
│   ├── utils.py                 # вспомогательные функции (one_hot, accuracy, mae, rmse, precision, recall, f1)
│   └── visualization.py         # визуализация результатов
├── examples/                    # примеры использования
│   ├── activations_comparison.py # сравнение ReLU, Tanh, Sigmoid
│   ├── iris_classification.py   # классификация ирисов (3 класса)
│   ├── mnist_classification.py  # распознавание/классификация цифр (Digits, аналог MNIST)
│   └── regression_demo.py       # аппроксимация синусоиды (задача регрессии)
├── tests/                       # модульные тесты
│   ├── test_data.py             # тесты Dataset, batch, shuffle
│   ├── test_layers.py           # тесты слоев
│   ├── test_losses.py           # тесты функций потерь
│   └── test_optimizers.py       # тесты SGD, MomentumSGD, Adam
├── reports/                     # HTML-отчеты обучения
│   ├── iris_report.html         # отчет по классификации ирисов
│   ├── mnist_report.html        # отчет по классификации цифр
│   └── regression_report.html   # отчет по аппроксимации синусоиды (задача регрессии)
├── playground.py                # NoCode визуальный конструктор (Streamlit)
├── README.md                    # документация
├── requirements.txt             # зависимости
└── .gitignore                   
```