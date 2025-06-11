import os
import numpy as np
import pandas as pd
from nltk.app.nemo_app import colors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import label_binarize
from itertools import cycle




def load_and_preprocess_data(folder_path, max_samples_per_class=2000, sequence_length=100):

    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    X, y = [], []

    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        df = df.head(50000)  
        label = df.iloc[0, -1]  
        data = df.iloc[:, :3].values

        # 截断或填充序列
        if len(data) > sequence_length:
            for i in range(0, len(data) - sequence_length + 1, sequence_length // 2):
                X.append(data[i:i + sequence_length])
                y.append(label)
        else:
            padded = np.zeros((sequence_length, 3))
            padded[:len(data)] = data
            X.append(padded)
            y.append(label)

    # 均衡采样
    X, y = np.array(X), np.array(y)
    unique_labels = np.unique(y)
    sampled_X, sampled_y = [], []

    for label in unique_labels:
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        sampled_idx = idx[:min(max_samples_per_class, len(idx))]
        sampled_X.append(X[sampled_idx])
        sampled_y.append(y[sampled_idx])

    X = np.vstack(sampled_X)
    y = np.hstack(sampled_y)

    # 检查是否有重复样本
    unique_X, unique_indices = np.unique(X, axis=0, return_index=True)
    if len(unique_indices) != len(X):
        print(f"发现重复样本，原始样本数量: {len(X)}，去重后样本数量: {len(unique_X)}")
        X = X[unique_indices]
        y = y[unique_indices]

    return X, y


def preprocess(X, y):
    # 标准化
    scalers = {}
    for i in range(X.shape[2]):
        scalers[i] = StandardScaler()
        X[:, :, i] = scalers[i].fit_transform(X[:, :, i])


    np.savez('scaler_params.npz',
             means=[scaler.mean_ for scaler in scalers.values()],
             scales=[scaler.scale_ for scaler in scalers.values()])


    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)


    np.save('label_encoder_classes.npy', le.classes_)

    return X, y_categorical, le, scalers



folder_path = "G:\pythonfiles\pzj\pjzpython\merged"
X, y = load_and_preprocess_data(folder_path, max_samples_per_class=3846, sequence_length=100)
X, y, label_encoder, scalers = preprocess(X, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)  # 学习率调整
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


epochs = len(history.history['loss'])

loss_df = pd.DataFrame({
    'epoch': range(1, epochs + 1),
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})
loss_df.to_csv('loss_data.csv', index=False)

acc_df = pd.DataFrame({
    'epoch': range(1, epochs + 1),
    'train_accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy']
})
acc_df.to_csv('accuracy_data.csv', index=False)


#可视化

window_length = 11
polyorder = 2

train_loss_smoothed = savgol_filter(history.history['loss'], window_length, polyorder)
val_loss_smoothed = savgol_filter(history.history['val_loss'], window_length, polyorder)
train_acc_smoothed = savgol_filter(history.history['accuracy'], window_length, polyorder)
val_acc_smoothed = savgol_filter(history.history['val_accuracy'], window_length, polyorder)

#数据平滑
smoothed_loss_df = pd.DataFrame({
    'epoch': range(1, len(train_loss_smoothed) + 1),
    'train_loss_smoothed': train_loss_smoothed,
    'val_loss_smoothed': val_loss_smoothed
})
smoothed_loss_df.to_csv('smoothed_loss_data.csv', index=False)


smoothed_acc_df = pd.DataFrame({
    'epoch': range(1, len(train_acc_smoothed) + 1),
    'train_accuracy_smoothed': train_acc_smoothed,
    'val_accuracy_smoothed': val_acc_smoothed
})
smoothed_acc_df.to_csv('smoothed_accuracy_data.csv', index=False)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_acc_smoothed, label='Train Accuracy (Smoothed)')
plt.plot(val_acc_smoothed, label='Validation Accuracy (Smoothed)')
plt.title('Accuracy over Epochs (Smoothed)')
plt.legend()
plt.savefig('accuracy_curves_smoothed.png')
plt.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 2)
plt.plot(train_loss_smoothed, label='Train Loss (Smoothed)')
plt.plot(val_loss_smoothed, label='Validation Loss (Smoothed)')
plt.title('Loss over Epochs (Smoothed)')
plt.legend()
plt.savefig('loss_curves_smoothed.png')
plt.close()

# 混淆矩阵
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)

cm_percentage = []
for i, row in enumerate(cm):
    row_sum = np.sum(row)
    if row_sum == 0:
        cm_percentage.append([0] * len(row))
    else:
        current_percentage = (row / row_sum)

        if np.any(current_percentage > 100):
            print(f"第 {i} 行计算百分比时出现超过 100% 的值: {current_percentage}")
        cm_percentage.append(current_percentage)
cm_percentage = np.array(cm_percentage)
cm_percentage_df = pd.DataFrame(cm_percentage, columns=label_encoder.classes_, index=label_encoder.classes_)
cm_percentage_df.to_csv('confusion_matrix_data.csv', index=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = '25'
plt.figure(figsize=(15, 12))
colors=['#FAFEFD','#E6FCF5','#C3FAE8','#96F2D7','#63E6BE','#38D9A9','#20C997','#12B886','#0CA678','#099268','#087F5B']
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap=cmap,
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix ', fontweight='bold',fontsize=25)
plt.xlabel('Predicted', fontweight='bold',fontsize=22)
plt.ylabel('True', fontweight='bold',fontsize=22)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# 分类报告：Precision, Recall, F1-Score
class_report = classification_report(y_true_labels, y_pred_labels,
                                     target_names=label_encoder.classes_,
                                     output_dict=True)

# 保存分类报告到CSV
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv('classification_report.csv', index=True)

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))


# 输出结果
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print("Loss data saved to loss_data.csv")
print("Accuracy data saved to accuracy_data.csv")
print("Smoothed loss data saved to smoothed_loss_data.csv")
print("Smoothed accuracy data saved to smoothed_accuracy_data.csv")
print("Confusion matrix data saved to confusion_matrix_data.csv")
print("Classification report saved to classification_report.csv")


model.save('final_lstm_classifier.h5')
print("模型已保存为 final_lstm_classifier.h5")
print("标准化器参数已保存为 scaler_params.npz")
print("标签编码器类别已保存为 label_encoder_classes.npy")
