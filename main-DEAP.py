from cross_validation import *
from prepare_data_DEAP import *
from predict import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str, default='/root/kylin/LGG/sample_deap_data/')
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='L', choices=['A', 'V', 'D', 'L'])
    parser.add_argument('--segment', type=int, default=4)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--scale-coefficient', type=float, default=1)
    parser.add_argument('--input-shape', type=tuple, default=(1, 32, 512))
    parser.add_argument('--data-format', type=str, default='eeg')
    ######## Training Process ########
    parser.add_argument('--train', action='store_true', help="Run model training")
    parser.add_argument('--random-seed', type=int, default=2021)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--patient', type=int, default=20)
    parser.add_argument('--patient-cmb', type=int, default=8)
    parser.add_argument('--max-epoch-cmb', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mae'])

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--save_pred', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='LGGNet')
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--graph-type', type=str, default='hem', choices=['fro', 'gen', 'hem', 'BL'])
    parser.add_argument('--hidden', type=int, default=32)
    
     ######## Test Model ########
    parser.add_argument('--test', action='store_true', help="Run model testing")
    parser.add_argument('--test-mat-file', type=str, default="data/s01.mat", help="Path to .mat file for testing")
    parser.add_argument('--load-model-path', type=str, default="min-loss.pth", help="Path to model file for testing")
    parser.add_argument('--output-dir', type=str, default="test_outputs", help="Directory to save test outputs")

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    if args.train:
        sub_to_run = np.arange(args.subjects)
        pd = PrepareData(args)
        pd.run(sub_to_run, split=True, expand=True)
        cv = CrossValidation(args)
        seed_all(args.random_seed)
        cv.n_fold_CV(subject=sub_to_run)
    
    if args.test:
        pd = PrepareData(args)
        mat_data = load_mat_file(args.test_mat_file)
        if mat_data is None:
            exit()
        data_var_name = "data"
        label_var_name = "label"

        if data_var_name not in mat_data or label_var_name not in mat_data:
            print("Error: Selected variable names not found in .mat file.")
            exit()

        test_data = mat_data[data_var_name]
        test_labels = mat_data[label_var_name]
        # TODO: # labels may not be in the correct order
        
        idx = []
        num_chan_local_graph = []
        for group in pd.graph_gen_DEAP:
            num_chan_local_graph.append(len(group))
            for chan in group:
                idx.append(pd.original_order.index(chan))
        
        from networks import LGGNet  # Replace with your model definition
        model = LGGNet(num_classes=0,  # 根据模型定义配置参数
                   input_size=(1,32, 500),
                   sampling_rate=500,  # 替换为你的采样率
                   num_T=64,
                   out_graph=32,
                   dropout_rate=0.5,
                   pool=16,
                   pool_step_rate=0.25,
                   idx_graph=num_chan_local_graph)
        model.load_state_dict(torch.load(args.load_model_path))
        loss_fn = torch.nn.MSELoss()
        mae, pred, act = test_model(model, test_data, test_labels, loss_fn, args.batch_size, idx=idx)
        print(f"Mean Absolute Error (MAE): {mae}")
        save_test_results(pred, act, args.output_dir)
        visualize_results(pred, act)
