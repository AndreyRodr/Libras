import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.client import device_lib


# Importa configurações e define Workers
from config import * 
# ----------------------------------------

# --- NOVAS FUNÇÕES DE PLOTAGEM ---
def plot_confusion_matrix(cm, classes, output_dir):
    """Plota e salva a Matriz de Confusão."""
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão Final')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Verdadeira')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close() # Fecha a figura para liberar memória
    print(f"✅ Matriz de Confusão salva em '{os.path.join(output_dir, 'confusion_matrix.png')}'")

def plot_training_history(history, history_fine_tune, output_dir):
    """Plota e salva a evolução de Loss e Accuracy ao longo das duas fases."""
    
    # Combina históricos da Fase 1 e Fase 2
    hist_combined = {}
    for key in history.history.keys():
        hist_combined[key] = history.history[key] + history_fine_tune.history[key]

    epochs = range(1, len(hist_combined['accuracy']) + 1)

    # 1. Plot de Acurácia
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_combined['accuracy'], 'b', label='Acurácia Treinamento')
    plt.plot(epochs, hist_combined['val_accuracy'], 'r', label='Acurácia Validação')
    plt.title('Acurácia ao Longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    # 2. Plot de Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist_combined['loss'], 'b', label='Loss Treinamento')
    plt.plot(epochs, hist_combined['val_loss'], 'r', label='Loss Validação')
    plt.title('Loss ao Longo das Épocas')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    print(f"✅ Histórico de Treinamento salvo em '{os.path.join(output_dir, 'training_history.png')}'")
# --- FIM DAS FUNÇÕES DE PLOTAGEM ---


def create_model(num_classes):
    """Cria modelo MobileNetV2 congelado para Fase 1."""
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False 

    model = Sequential([
        base_model, GlobalAveragePooling2D(),
        Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    return model

def run_training():
    
    # Verificações básicas (pressupõe que download_data.py foi executado)
    if not (os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR)):
        print("❌ ERRO: Dados não encontrados.")
        print("Execute 'python download_data.py' antes de iniciar o treinamento.")
        sys.exit(1)
        
    print(f"--- Treinamento: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE} ---")

    # Geradores de Dados
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
        validation_split=0.2
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
    validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')
    test_generator = val_test_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    num_train_samples = train_generator.samples
    num_validation_samples = validation_generator.samples
    classes = list(train_generator.class_indices.keys())
    
    with open(CLASSES_FILE, 'w') as f:
        for cls in classes: f.write(f"{cls}\n")

    # --- FASE 1: TREINAMENTO DO TOPO ---
    model = create_model(len(classes))
    model.summary()
    
    callbacks_fase1 = [
        ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    ]

    print("\n--- FASE 1: TREINANDO TOPO ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // BATCH_SIZE,
        callbacks=callbacks_fase1,
        verbose=1
    )

    # --- FASE 2: FINE-TUNING ---
    print("\n--- FASE 2: FINE-TUNING (DESCONGELANDO) ---")
    model = load_model(MODEL_PATH)
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Congelar as primeiras 50 camadas (preserva features básicas)
    for layer in base_model.layers[:50]:
        layer.trainable = False

    # Compila com LR inicial de 1e-5, mas usaremos ReduceLROnPlateau para baixar se precisar
    model.compile(
        optimizer=Adam(learning_rate=0.00001), 
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    
    TOTAL_EPOCHS = history.epoch[-1] + 15 # +15 épocas para ajuste fino
    
    callbacks_fase2 = [
        ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, verbose=1, restore_best_weights=True),
        # NOVO: Reduz LR se estagnar, ajudando a chegar no ótimo global
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    ]

    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        validation_steps=num_validation_samples // BATCH_SIZE,
        callbacks=callbacks_fase2,
        verbose=1
    )

    # --- AVALIAÇÃO E SALVAMENTO DE MÉTRICAS ---
    print("\n--- AVALIAÇÃO FINAL ---")
    final_model = load_model(MODEL_PATH)
    loss, acc = final_model.evaluate(test_generator)
    
    # Matriz de Confusão
    Y_pred = final_model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    
    # ----------------------------------------------------
    # --- CÁLCULO E SALVAMENTO DE MÉTRICAS E GRÁFICOS ---
    # ----------------------------------------------------
    OUTPUT_DIR = "metrics_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Relatório de Classificação para Precisão e Recall
    # Nota: Precisão Média (Weighted Avg) e Recall Médio (Weighted Avg) são as melhores
    # métricas para sumarizar a performance em datasets desbalanceados.
    report = classification_report(test_generator.classes, y_pred, target_names=classes, output_dict=True)
    avg_precision = report['weighted avg']['precision']
    avg_recall = report['weighted avg']['recall']

    print("\n--- MÉTRICAS FINAIS DO TESTE ---")
    print(f"Loss Final: {loss:.4f}")
    print(f"Acurácia Final: {acc:.4f}")
    print(f"Precisão Média (Weighted Avg): {avg_precision:.4f}")
    print(f"Recall Médio (Weighted Avg): {avg_recall:.4f}")
    
    # 2. Plot e Salvamento da Matriz de Confusão
    plot_confusion_matrix(cm, classes, OUTPUT_DIR)

    # 3. Plot e Salvamento do Histórico de Treinamento
    plot_training_history(history, history_fine_tune, OUTPUT_DIR)
    
    # 4. Salvamento de Métricas em arquivo de texto
    with open(os.path.join(OUTPUT_DIR, 'final_metrics.txt'), 'w') as f:
        f.write("--- MÉTRICAS FINAIS DO TESTE ---\n")
        f.write(f"Loss Final: {loss:.4f}\n")
        f.write(f"Acurácia Final: {acc:.4f}\n")
        f.write(f"Precisão Média (Weighted Avg): {avg_precision:.4f}\n")
        f.write(f"Recall Médio (Weighted Avg): {avg_recall:.4f}\n")
    print(f"✅ Métricas salvas em '{os.path.join(OUTPUT_DIR, 'final_metrics.txt')}'")
    # ----------------------------------------------------


if __name__ == "__main__":
    run_training()