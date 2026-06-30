"""
визуальный конструктор нейросетей в стиле TensorFlow Playground
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nn import Sequential, Dense, ReLU, Tanh, Sigmoid, Softmax, Linear
from nn.losses import MSELoss, CrossEntropyLoss
from nn.optimizers import SGD, Adam, MomentumSGD
from nn.utils import one_hot, accuracy, precision, recall, f1_score
from nn.data import Dataset

st.set_page_config(page_title="Aura Neural Network Playground", layout="wide")
st.title("Aura Neural Network Playground")
st.markdown("Настрой и обучи сеть визуально")

# боковушка
with st.sidebar:
    st.header("Настройки данных")
    
    # TODO: загрузка своего датасета (подумать какие форматы csv, xlsx, txt?)
    # пока готовые датасеты из sklearn, потом подгрузим то что напишет Алина
    dataset = st.selectbox(
        "Датасет",
        ["Круги", "Луны", "Линейный", "Кластеры"]
    )
    
    noise = st.slider("Шум", 0.0, 0.5, 0.1, 0.05)
    n_samples = st.slider("Количество примеров", 200, 1000, 500)
    
    st.header("Архитектура сети")
    
    n_layers = st.number_input("Количество слоев", min_value=1, max_value=5, value=2)
    
    layers = []
    for i in range(n_layers):
        st.subheader(f"Слой {i+1}")
        
        if i < n_layers - 1:
            neurons = st.number_input(
                f"Нейронов (слой {i+1})", 
                min_value=2, max_value=128, value=16, key=f"neurons_{i}"
            )
            activation = st.selectbox(
                f"Активация (слой {i+1})",
                ["ReLU", "Tanh", "Sigmoid"],
                key=f"act_{i}"
            )
            layers.append(("dense", neurons, activation))
        else:
            task = st.selectbox("Тип задачи", ["Классификация", "Регрессия"])
            
            if task == "Классификация":
                output_activation = "Softmax"
                loss_name = "CrossEntropyLoss"
            else:
                output_activation = "Linear"
                loss_name = "MSELoss"
            
            layers.append(("output", output_activation, loss_name))
    
    st.header("Параметры обучения")
    
    optimizer_name = st.selectbox("Оптимизатор", ["SGD", "MomentumSGD", "Adam"])
    learning_rate = st.number_input("Learning rate", 0.0001, 0.1, 0.01, format="%.4f")
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    epochs = st.slider("Эпохи", 10, 200, 50)
    
    train_btn = st.button("Обучить сеть", type="primary")


# сами данные
def generate_dataset(name, n_samples, noise):
    if name == "Круги":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif name == "Луны":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif name == "Линейный":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1, random_state=42
        )
    else:
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=2, random_state=42
        )
    
    X = StandardScaler().fit_transform(X)
    return X, y

X, y = generate_dataset(dataset, n_samples, noise)

n_classes = len(np.unique(y))
task = "classification"


# график
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Исходные данные")
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='white')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(f"{dataset} (шум={noise})")
    st.pyplot(fig)


# модель
def build_model(layers, input_dim, output_dim):
    model_layers = []
    
    for layer in layers:
        if layer[0] == "dense":
            _, neurons, activation = layer
            model_layers.append(Dense(input_dim, neurons, init="he"))
            input_dim = neurons
            
            if activation == "ReLU":
                model_layers.append(ReLU())
            elif activation == "Tanh":
                model_layers.append(Tanh())
            elif activation == "Sigmoid":
                model_layers.append(Sigmoid())
        
        elif layer[0] == "output":
            _, output_activation, _ = layer
            model_layers.append(Dense(input_dim, output_dim))
            
            if output_activation == "Softmax":
                model_layers.append(Softmax())
            elif output_activation == "Linear":
                model_layers.append(Linear())
    
    return Sequential(model_layers)

output_dim = n_classes


# обучение
if train_btn:
    with st.spinner("Обучение нейросети..."):
        
        y_train = one_hot(y, output_dim)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_train, test_size=0.2, random_state=42
        )
        
        train_ds = Dataset(X_train, y_train).shuffle().batch(batch_size)
        val_ds = Dataset(X_val, y_val)
        
        model = build_model(layers, X.shape[1], output_dim)
        
        # loss = CrossEntropyLoss()
        # metrics = ["accuracy"]
        
        if task == "Классификация":
            loss = CrossEntropyLoss()
            metrics = ["accuracy"]
        else:
            loss = MSELoss()
            metrics = []
        
        if optimizer_name == "SGD":
            optimizer = SGD(lr=learning_rate)
        elif optimizer_name == "MomentumSGD":
            optimizer = MomentumSGD(lr=learning_rate)
        else:
            optimizer = Adam(lr=learning_rate)
        
        history = model.fit(
            train_data=train_ds,
            epochs=epochs,
            loss=loss,
            optimizer=optimizer,
            val_data=val_ds,
            metrics=metrics,
            verbose=False
        )
        
        st.session_state['history'] = history
        st.session_state['model'] = model
        st.session_state['X_val'] = X_val
        st.session_state['y_val'] = y_val


# рядом колонка с результатами (справа)
with col2:
    st.subheader("Результаты")
    
    if 'history' in st.session_state:
        history = st.session_state['history']
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        if 'accuracy' in history:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(history['accuracy'], label='Accuracy')
            if 'val_accuracy' in history:
                ax.plot(history['val_accuracy'], label='Val Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        model = st.session_state['model']
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        
        y_pred = model.predict(X_val)
        acc = accuracy(y_pred, y_val)
        st.metric("Точность (accuracy) на проверочной выборке:", f"{acc:.2%}")
        
        p = precision(y_pred, y_val)
        r = recall(y_pred, y_val)
        f1 = f1_score(y_pred, y_val)
        
        col_met1, col_met2, col_met3 = st.columns(3)
        col_met1.metric("Precision", f"{p:.2%}")
        col_met2.metric("Recall", f"{r:.2%}")
        col_met3.metric("F1-score", f"{f1:.2%}")
        
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_val, axis=1)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        scatter = ax.scatter(X_val[:, 0], X_val[:, 1], c=y_pred_labels, 
                             cmap='coolwarm', alpha=0.7, edgecolors='white')
        ax.set_title("Предсказания модели")
        st.pyplot(fig)


# итоговый конфиг модели
st.divider()
with st.expander("Архитектура модели"):
    if 'model' in st.session_state:
        st.code(str(st.session_state['model']))
    else:
        st.info("Нажмите 'Обучить сеть' для создания модели")

if 'model' in st.session_state:
    with st.expander("Сгенерировать код модели"):
        st.code(f"""
from nn import Sequential, Dense, ReLU, Softmax, CrossEntropyLoss, {optimizer_name}

model = Sequential([
    Dense(2, {layers[0][1] if len(layers) > 0 else 16}, init="he"),
    ReLU(),
    Dense({layers[0][1] if len(layers) > 0 else 16}, {output_dim}),
    Softmax()
])

loss = CrossEntropyLoss()
optimizer = {optimizer_name}(lr={learning_rate})
history = model.fit(train_data, epochs={epochs}, loss=loss, optimizer=optimizer)
""")
        
st.caption("Aura Neural Network Playground using Aura Framework (mainly NumPy backend)")