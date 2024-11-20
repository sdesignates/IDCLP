from Utils import *
import argparse
from network import IDCLP
from run_HGAN import *
import os

import sys
current_directory = os.path.dirname(os.path.abspath(__file__))

def main():
    # args = parse_args()
    # drug_representation = Pre_train(args)#
    # Training settings
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--epochs', type=int, default=50,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        metavar='FLOAT', help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=128,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        metavar='FLOAT', help='weight decay')
    parser.add_argument('--droprate', type=float, default=0.01,
                        metavar='FLOAT', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--rawpath', type=str, default=current_directory+'/',
                        metavar='STRING', help='rawpath')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--weight_path', type=str, default='best2',
                        help='filepath for pretrained weights')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))

    # drug_representation = Pre_train(graph=graph, num_steps=4000)
    drug_representation = np.load(args.rawpath+'data/Drug/drug_representation.npy')
    train_loader, test_loader, val_loader = load_data(args,drug_representation)
    gene_embed_dim = 128
    pert_type_emb_dim = 4
    hid_dim = 128 
    num_gene = 978
    model = IDCLP(num_gene, val_loader.dataset[0][2].shape[0], val_loader.dataset[0][3].shape[0], val_loader.dataset[0][0].shape[1], gene_embed_dim, val_loader.dataset[0][1].shape[0],
                 pert_type_emb_dim, args.embed_dim, val_loader.dataset[0][5].shape[0], args.batch_size, args.droprate, args.droprate).to(
        args.device)
    if args.mode == 'train':
        Regression_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_folder = os.path.join(os.getcwd(), current_directory+"result/log_gene_best2", model._get_name())
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        stopper = EarlyStopping(mode='lower', patience=args.patience)
        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_loss = train(model, train_loader, Regression_criterion, optimizer, args.device)
            # fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)

            # fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)
            print('Evaluating...')
            rmse, _, _, _, _, _, _, _, _ = validate(model, val_loader, args.device)
            print("Validation rmse:{}".format(rmse))
            # fitlog.add_metric({'val': {'RMSE': rmse}}, step=epoch)

            early_stop = stopper.step(rmse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Testing...')
        stopper.load_checkpoint(model)
        torch.save(model.state_dict(), current_directory+'w/o_cell_line_dosage{}.pth'.format(args.weight_path))  #_dosage
        train_rmse, train_MAE, train_r2, train_r, train_spearman, train_precision_neg, train_precision_pos, _, _ = validate(model, train_loader, args.device)
        val_rmse, val_MAE, val_r2, val_r, val_spearman, val_precision_neg, val_precision_pos, _, _ = validate(model, val_loader, args.device)
        test_rmse, test_MAE, test_r2, test_r, test_spearman, test_precision_neg, test_precision_pos, _, _ = validate(model, test_loader, args.device)
        print('Train reslut: rmse:{} mae:{} r:{} spearman:{} precision_neg:{} precision_pos:{}'.format(train_rmse, train_MAE, train_r,  train_spearman, train_precision_neg, train_precision_pos))
        print('Val reslut: rmse:{} mae:{} r:{} spearman:{} precision_neg:{} precision_pos:{}'.format(val_rmse, val_MAE, val_r, val_spearman, val_precision_neg, val_precision_pos))
        print('Test reslut: rmse:{} mae:{}  r:{} spearman:{} precision_neg:{} precision_pos:{}'.format(test_rmse, test_MAE, test_r, test_spearman, test_precision_neg, test_precision_pos))

        # fitlog.add_best_metric(
        #     {'epoch': epoch - args.patience,
        #      "train": {'RMSE': train_rmse, 'MAE': train_MAE, 'pearson': train_r, "R2": train_r2, "spearman": train_spearman, "precision_neg":train_precision_neg, "precision_pos": train_precision_pos},
        #      "valid": {'RMSE': stopper.best_score, 'MAE': val_MAE, 'pearson': val_r, 'R2': val_r2, "spearman": val_spearman, "precision_neg":val_precision_neg, "precision_pos": val_precision_pos},
        #      "test": {'RMSE': test_rmse, 'MAE': test_MAE, 'pearson': test_r, 'R2': test_r2, "spearman": test_spearman, "precision_neg":test_precision_neg, "precision_pos": test_precision_pos}})

    elif args.mode == 'test':
        weight = ""
        model.load_state_dict(
            torch.load(current_directory+'{}.pth'.format(args.weight_path), map_location=args.device))#['model_state_dict']
        test_rmse, test_MAE, test_r2, test_r, test_spearman, test_precision_neg, test_precision_pos, y_true1, y_pred1 = validate(model, test_loader, args.device)
        print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 4), round(test_MAE, 4),
                                                             round(test_r2, 4), round(test_r, 4)))
        
        ####### 保存预测结果y_true1, y_pred1#####################
        # 我们首先将所有的张量转换为numpy数组
        y_true1 = [t.view(-1).numpy() for t in y_true1]
        y_pred1 = [t.view(-1).numpy() for t in y_pred1]

        # 然后我们可以将这个列表转换为Pandas的DataFrame
        y_true1 = pd.DataFrame(y_true1)
        y_pred1 = pd.DataFrame(y_pred1)

        # 最后我们将DataFrame保存为CSV文件
        y_true1.to_csv(current_directory+'y_true1.csv', index=False)
        y_pred1.to_csv(current_directory+'y_pred1.csv', index=False)
        # np.savetxt(current_directory+'y_true.csv',y_true1.detach().numpy(),fmt='%.2f',delimiter=',')

        
        
    

if __name__ == "__main__":
    main()
    print('hello zxy log_gene_best2')