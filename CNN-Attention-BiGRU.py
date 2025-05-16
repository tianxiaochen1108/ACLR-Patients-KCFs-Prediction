import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# 数据加载
all_sheets = pd.read_excel('./TestData.xlsx', sheet_name=None, header=None)
X_sheet_names, y_sheet_names = [], []
i = 0
for sheet_name in all_sheets.keys():
    if i < 18:
        X_sheet_names.append(sheet_name)
        i += 1
    else:
        y_sheet_names.append(sheet_name)

# 特征处理
all_series = []
for sheet_name in X_sheet_names:
    df = pd.read_excel('./TestData.xlsx', sheet_name=sheet_name, header=None)
    all_series.append(np.array(df.T).flatten())
final_df = pd.DataFrame(all_series).T

# 目标处理
y_df = pd.read_excel('./TestData.xlsx', sheet_name=y_sheet_names, header=None)
input_data = final_df.values
output_data = np.array(y_df['膝关节接触力'].T).reshape(-1, 1)

print(f"输入维度: {input_data.shape}, 输出维度: {output_data.shape}")

# 随机种子设置
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型定义
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        return torch.matmul(attention_weights, V)

class AttentionCNNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.CNN1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm1d(16)
        self.ReLU1 = nn.ReLU()
        self.CNN2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm1d(32)
        self.ReLU2 = nn.ReLU()
        self.gru = nn.GRU(32, hidden_size, bidirectional=True, batch_first=True)
        self.attention = SelfAttention(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.ReLU1(self.BN1(self.CNN1(x)))
        x = self.ReLU2(self.BN2(self.CNN2(x)))
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        attended = self.attention(gru_out)
        return self.fc(attended[:, -1, :])

# 交叉验证配置
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=False)
results = {
    'train_loss': [], 'val_loss': [], 'r2': [],
    'mae': [], 'mape': [], 'y_true': [], 'y_pred': []
}
all_true = np.zeros(output_data.shape[0])  # 全量真实值存储
all_pred = np.zeros(output_data.shape[0])  # 全量预测值存储

for fold, (train_idx, val_idx) in enumerate(kf.split(input_data)):
    print(f"\n=== Fold {fold+1}/{n_splits} ===")
    X_train, X_val = input_data[train_idx], input_data[val_idx]
    y_train, y_val = output_data[train_idx], output_data[val_idx]
    
    # 数据标准化
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # Tensor转换
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).squeeze().to(device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).squeeze().to(device)
    
    # 数据加载器
    batch_size = 256
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                          batch_size=batch_size, shuffle=False)
    
    # 模型初始化
    model = AttentionCNNGRU(
        input_size=X_train_tensor.shape[1],
        hidden_size=128,
        output_size=1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练循环
    num_epochs = 200
    best_r2 = -np.inf
    fold_train_loss, fold_val_loss = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        fold_train_loss.append(epoch_loss/len(train_loader))
        
        # 验证评估
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).cpu().numpy()
                val_preds.append(preds)
                val_true.append(y_batch.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_loss = mean_squared_error(val_true, val_preds)
        fold_val_loss.append(val_loss)
        r2 = r2_score(val_true, val_preds)
        
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), f"Attention_CNN_GRU_fold{fold+1}.pth")

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss {fold_train_loss[-1]:.4f}, Val Loss {val_loss:.4f}, R² {r2:.4f}")
    
    # 记录结果
    results['train_loss'].append(fold_train_loss)
    results['val_loss'].append(fold_val_loss)
    results['r2'].append(best_r2)
    
    # 最终预测
    model.load_state_dict(torch.load(f"Attention_CNN_GRU_fold{fold+1}.pth"))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_tensor).cpu().numpy()
    
    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler_y.inverse_transform(y_val_scaled)
    
    # 保存当前fold预测结果
    fold_df = pd.DataFrame({
        'True': y_true.flatten(),
        'Pred': y_pred.flatten()
    })
    fold_df.to_csv(f'Attention_CNN_GRU_fold_{fold+1}_predictions.csv', index=False)
    
    # 更新全量预测
    all_true[val_idx] = y_true.flatten()
    all_pred[val_idx] = y_pred.flatten()
    
    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    results['mae'].append(mae)
    results['mape'].append(mape)
    results['y_true'].append(y_true)
    results['y_pred'].append(y_pred)
    print(f"Fold {fold+1} Metrics: R²={best_r2:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

# ==== 结果保存 ====
# 保存全量预测结果
pd.DataFrame({'True': all_true, 'Pred': all_pred}).to_csv('Attention_CNN_GRU_all_predictions.csv', index=False)

# 保存评估指标
metrics_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(n_splits)],
    'R2': results['r2'],
    'MAE': results['mae'],
    'MAPE (%)': results['mape']
})
metrics_df.to_csv('Attention_CNN_GRU_metrics.csv', index=False)

# 保存训练曲线数据
pd.DataFrame(results['train_loss']).T.to_csv('attention_cnn_gru_train_loss.csv', index=False)
pd.DataFrame(results['val_loss']).T.to_csv('attention_cnn_gru_val_loss.csv', index=False)

# 结果汇总
print("\n=== 交叉验证结果 ===")
print(f"平均R²: {np.mean(results['r2']):.4f} (±{np.std(results['r2']):.4f})")
print(f"平均MAE: {np.mean(results['mae']):.4f} (±{np.std(results['mae']):.4f})")
print(f"平均MAPE: {np.mean(results['mape']):.2f}% (±{np.std(results['mape']):.2f}%)")

# 训练曲线可视化
plt.figure(figsize=(12, 6))
for i in range(n_splits):
    plt.plot(results['train_loss'][i], alpha=0.3, label=f'Train Fold {i+1}')
    plt.plot(results['val_loss'][i], '--', alpha=0.3, label=f'Val Fold {i+1}')
plt.title('Training/Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("Attention_CNN_GRU_loss_curves.png")
plt.show()

# 预测对比图
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(results['y_true'][-1][-100:], 'r-', label='Actual')
plt.plot(results['y_pred'][-1][-100:], 'b--', label='Predicted')
plt.xlabel("Time Steps")
plt.ylabel("Force Value")
plt.legend()
plt.title("Prediction Comparison (Last 100 Samples)")
plt.savefig("Attention_CNN_GRU_prediction_comparison.png")
plt.show()

# ==== SHAP分析 ====
# 使用最后一个fold的模型
model.load_state_dict(torch.load(f"Attention_CNN_GRU_fold{n_splits}.pth"))
model.eval()

# 准备背景数据和测试数据
background = X_train_tensor[:1000].to(device)  # 使用前1000个样本作为背景
test_samples = X_val_tensor[:100].to(device)    # 分析前100个测试样本

# 创建解释器
explainer = shap.GradientExplainer(model, background)

# 计算SHAP值
with torch.enable_grad():
    model.train()
    shap_values = explainer.shap_values(test_samples)
    model.eval()

# 调整维度
shap_values = np.array(shap_values).reshape(-1, input_data.shape[1])

# 特征名称（根据实际情况修改）
feature_names = [f'Feature_{i+1}' for i in range(input_data.shape[1])]

# 保存SHAP摘要图
plt.figure()
shap.summary_plot(shap_values, test_samples.cpu().numpy(), feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("Attention_CNN_GRU_SHAP_summary.png")
plt.close()

# 保存SHAP热力图
plt.figure()
shap.plots.heatmap(
    shap.Explanation(
        values=shap_values,
        data=test_samples.cpu().numpy(),
        feature_names=feature_names
    ), 
    show=False
)
plt.tight_layout()
plt.savefig("Attention_CNN_GRU_SHAP_heatmap.png")
plt.close()