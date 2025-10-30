import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')

data = pd.concat([normal_data, bearing_fault_data, rotor_fault_data], ignore_index=True)

X = data[['value']]
y = data[['target']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_test = X_test + np.random.normal(0, 0.1, X_test.shape)
X_test = X_test + 0.05        

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_num_samples = X_train.shape[0] 
X_test_num_samples = X_test.shape[0]

X_train = X_train.reshape(X_train_num_samples, 1, 1) 
X_test = X_test.reshape(X_test_num_samples, 1, 1) 

y_train_samples = y_train['target'].values 
y_test_samples = y_test['target'].values

encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train_samples.reshape(-1, 1))
y_test = encoder.fit_transform(y_test_samples.reshape(-1, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu', input_shape=(1, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

loss_history = []
accuracy_history = []

batch_loss_history = []

class LossHistory(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        batch_loss_history.append(logs['loss'])

history = model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=1, callbacks=[LossHistory()])

loss_history.extend(history.history['loss'])
accuracy_history.extend(history.history['accuracy'])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss')
plt.title('Loss Function (Epoch Level)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label='Accuracy', color='orange')
plt.title('Accuracy (Epoch Level)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = model.predict(X_train)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Total Predictions: ", len(y_pred_classes))
import time
Test_start_time = time.time()  # 시작 시간 기록
y_test_pred_classes = np.argmax(model.predict(X_test), axis=1)
Test_end_time = time.time()  # 종료 시간 기록

Test__time = Test_end_time - Test_start_time  # 소요 시간 계산
print(f"총 소요 시간 : 테스트 시간 = {Test__time:.4f}s")

y_test_true_classes = np.argmax(y_test, axis=1)

y_train_pred_classes = np.argmax(model.predict(X_train), axis=1)
y_train_true_classes = np.argmax(y_train, axis=1)

confusion_mtx_train = confusion_matrix(y_train_true_classes, y_train_pred_classes)
confusion_mtx_test = confusion_matrix(y_test_true_classes, y_test_pred_classes)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx_train, display_labels=encoder.categories_[0])
disp_train.plot(ax=ax[0], cmap=plt.cm.Blues)
ax[0].set_title('Training Data Confusion Matrix')

disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx_test, display_labels=encoder.categories_[0])
disp_test.plot(ax=ax[1], cmap=plt.cm.Blues)
ax[1].set_title('Test Data Confusion Matrix')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(batch_loss_history, label='Batch Loss', color='green')
plt.title('Batch Loss Over All Epochs')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Classification Report (Test Data):")
print(classification_report(y_test_true_classes, y_test_pred_classes))
