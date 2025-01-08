import scipy.io
import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

# 函数：读取 .mat 文件
def load_mat_file(file_path):
    """
    读取 .mat 文件并返回数据字典。
    :param file_path: .mat 文件路径
    :return: 数据字典
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        print(f"Loaded .mat file from {file_path}")
        return mat_data
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None


# 函数：测试模型
def test_model(model, test_data, test_labels, loss_fn, batch_size=64, cuda=False,idx=None):

    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
    test_data_tensor = test_data_tensor[:,:,idx,:]
    model.eval()
    pred_list = []
    act_list = []
    with torch.no_grad():
        outputs = model(test_data_tensor)
        predictions = outputs.squeeze()  
        total_loss = loss_fn(predictions, test_labels_tensor).item()
        pred_list = predictions.cpu().numpy()
        act_list = test_labels_tensor.cpu().numpy()
    pred_array = np.array(pred_list)
    act_array = np.array(act_list)
    mae = np.mean(np.abs(pred_array - act_array))

    return mae, pred_array, act_array


# 函数：保存测试结果
def save_test_results(pred, act, output_dir):
    """
    保存预测值和真实值。
    :param pred: 预测值 (np.ndarray)
    :param act: 真实值 (np.ndarray)
    :param output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    # 保存为 .npy 文件
    np.save(os.path.join(output_dir, "predictions.npy"), pred)
    np.save(os.path.join(output_dir, "ground_truth.npy"), act)

    # 保存为 .csv 文件
    df = pd.DataFrame({'Predictions': pred.flatten(), 'Ground Truth': act.flatten()})
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions and ground truth to {csv_path}")


# 函数：可视化测试结果
def visualize_results(pred, act):
    """
    绘制预测值和真实值的对比图。
    :param pred: 预测值 (np.ndarray)
    :param act: 真实值 (np.ndarray)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pred[:100], label="Predictions", marker='o')
    plt.plot(act[:100], label="Ground Truth", marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Model Predictions vs Ground Truth")
    plt.legend()
    plt.savefig("results.png")


# 主函数：加载测试数据，测试模型并保存结果
if __name__ == "__main__":

    mat_file_path = "data/s01.mat"
    model_path = "min-loss.pth"
    
    output_dir = "test_outputs"
    batch_size = 64
    cuda = torch.cuda.is_available()

    # 2. 读取 .mat 文件
    mat_data = load_mat_file(mat_file_path)
    if mat_data is None:
        exit()

    # 打印 .mat 文件中的变量名
    print("Variables in .mat file:")
    for key in mat_data.keys():
        if not key.startswith("__"):
            print(f" - {key}")

    # 用户选择变量
    data_var_name = "data"
    label_var_name = "label"

    if data_var_name not in mat_data or label_var_name not in mat_data:
        print("Error: Selected variable names not found in .mat file.")
        exit()

    # 提取测试数据和标签
    test_data = mat_data[data_var_name]
    test_labels = mat_data[label_var_name]
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")
    for i in range(test_labels.shape[0]):
        if test_labels[i] <4:
            test_labels[i] = 4
    # print(f"test_data.shape[0]: {test_data.shape[0]}, test_data.shape[1]: {test_data.shape[1]}, test_data.shape[2]: {test_data.shape[2]}") 
    
    graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                               ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], [
                                   'CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], [
                                   'PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
    original_order = [
            'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
            'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
            'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
            'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
            'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
            'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
            'Oz', 'O1', 'O2']
    idx = []
    num_chan_local_graph = []
    for group in graph_gen_DEAP:
        num_chan_local_graph.append(len(group))
        for chan in group:
            idx.append(original_order.index(chan))

    # 3. 加载模型
    from networks import LGGNet  # 替换为你的模型定义
    model = LGGNet(num_classes=0,  # 根据模型定义配置参数
                   input_size=(1,32, 500),
                   sampling_rate=500,  # 替换为你的采样率
                   num_T=64,
                   out_graph=32,
                   dropout_rate=0.5,
                   pool=16,
                   pool_step_rate=0.25,
                   idx_graph=num_chan_local_graph) 

    model.load_state_dict(torch.load(model_path))

    # 4. 定义损失函数
    loss_fn = torch.nn.MSELoss()

    # 5. 测试模型
    mae, pred, act = test_model(model, test_data, test_labels, loss_fn, batch_size,idx=idx)

    # 6. 输出结果
    print(f"Mean Absolute Error (MAE): {mae}")
    save_test_results(pred, act, output_dir)

    # 7. 可视化结果
    visualize_results(pred, act)
