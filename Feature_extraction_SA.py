
import argparse
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms
from utils.flow_utils import  resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite
import torch
import scipy.io as sio
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import time
import datetime



def get_accuracy(y_true, y_pred):
    num_perclass = np.zeros((y_true.max() + 1))
    num = np.zeros((y_true.max() + 1))
    for i in range(len(y_true)):
        num_perclass[y_true[i]] += 1
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            num[y_pred[i]] += 1
    for i in range(len(num)):
        num[i] = num[i] / num_perclass[i]
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ac = np.zeros((y_true.max() + 1 + 2))
    ac[:y_true.max() + 1] = num
    ac[-1] = acc
    ac[-2] = kappa
    return ac  # acc,num.mean(),kappa


class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, img1,img2):
        img1=self.input_transform(img1).unsqueeze(0)
        img2=self.input_transform(img2).unsqueeze(0)

        img_pair = torch.cat((img1,img2), 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='./outputs/models/HSI_Raw_ckpt_BO+CH+KSC+HO13.pth.tar')
    parser.add_argument('-s', '--test_shape', default=[512, 192], type=int, nargs=2)

    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': 2,
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }
    ts = TestHelper(cfg)


    experiment_num=10
    dataname = 'SA'
    model_name = 'SMF-UL'
    num_range=[20]

    path='/HSI_data/'
    img=sio.loadmat(path+'Salinas_corrected.mat')
    img=img['salinas_corrected']

    gt=sio.loadmat(path+'Salinas_gt.mat')
    gt=gt['salinas_gt']

    spec=img.copy()
    spec=spec/spec.max()
    m,n,b=img.shape

    hsv = np.zeros((m, n, 3))

    time1 = time.time()

    for i in tqdm(range(3, b)):
        x1 = img[:, :, i - 3:i]
        x2 = img[:, :, i - 2:i + 1]

        tenOne = torch.FloatTensor(np.ascontiguousarray(x1.astype(np.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(np.ascontiguousarray(x2.astype(np.float32) * (1.0 / 255.0)))

        flow_12 = ts.run(tenOne,tenTwo)['flows_fw'][0]

        flow_12 = resize_flow(flow_12, (m, n))
        np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])


        if i == 3:
            feature = np_flow_12
        else:
            feature = np.concatenate((feature, np_flow_12),2)

    print(time.time() - time1)

    v_min = feature.min()
    v_max = feature.max()
    feature = (feature - v_min) / (v_max - v_min)

    feature = np.concatenate((feature, spec), 2)


    label_num = gt.max()
    data = []
    label = []
    data_global = []

    gt_index = []
    for i in tqdm(range(m)):
        for j in range(n):
            if gt[i, j] == 0:
                continue
            else:
                temp_data = feature[i, j, :]
                temp_label = np.zeros((1, label_num), dtype=np.int8)
                temp_label[0, gt[i, j] - 1] = 1
                data.append(temp_data)
                label.append(temp_label)
                gt_index.append((i) * n + j)
    #   print (i,j)

    for i in tqdm(range(m)):
        for j in range(n):
            temp_data = feature[i, j, :]
            data_global.append(temp_data)

    print('end')
    data = np.array(data)
    data = np.squeeze(data)

    data_global = np.array(data_global)
    data_global = np.squeeze(data_global)

    label = np.array(label)
    label = np.squeeze(label)
    label = label.argmax(1)




    data = np.float32(data)
    data_global = np.float32(data_global)



    Experiment_result = np.zeros([label_num + 5, experiment_num + 2])
    for i_num in range(num_range.__len__()):
        num = num_range[i_num]
        for iter_num in range(experiment_num):

            # np.random.seed(123456789)
            np.random.seed(iter_num+123456)
            indices = np.arange(data.shape[0])
            shuffled_indices = np.random.permutation(indices)

            images = data[shuffled_indices]
            labels = label[shuffled_indices]
            y = labels  # np.array([numpy.arange(9)[l==1][0] for l in labels])
            n_classes = y.max() + 1
            i_labeled = []
            for c in range(n_classes):
                if dataname=='IP':
                    if num == 10:
                        i = indices[y == c][:num]
                    if num == 20:

                        if c + 1 == 7:
                            i = indices[y == c][:10]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:10]  # 50
                        else:
                            i = indices[y == c][:num]  # 50

                    if num == 50:
                        if c + 1 == 1:
                            i = indices[y == c][:26]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:16]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:11]  # 50
                        else:
                            i = indices[y == c][:num]  # 50

                    if num == 80:
                        if c + 1 == 1:
                            i = indices[y == c][:26]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:16]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:11]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:60]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                    if num == 100:
                        if c + 1 == 1:
                            i = indices[y == c][:33]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:20]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:14]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:75]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                    if num == 150:
                        if c + 1 == 1:
                            i = indices[y == c][:36]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:22]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:16]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:80]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                    if num == 200:
                        if c + 1 == 1:
                            i = indices[y == c][:39]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:24]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:18]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:85]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                else:
                    i = indices[y == c][:num]
                    # i = indices[y==c][:10]#50
                i_labeled += list(i)
            l_images = images[i_labeled]
            l_labels = y[i_labeled]

            svc = SVC(kernel='rbf', class_weight='balanced', )
            c_range = np.logspace(-5, 15, 11, base=2)
            gamma_range = np.logspace(-9, 3, 13, base=2)
            # 网格搜索交叉验证的参数范围，cv=3,3折交叉，n_jobs=-1，多核计算
            param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
            grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)

            train_time1 = time.time()
            # 训练模型
            clf = grid.fit(l_images, l_labels)
            train_time2 = time.time()

            # 计算测试集精度
            score = grid.score(data, label)
            print('精度为%s' % score)


            tes_time1 = time.time()
            pred = clf.predict(data)
            tes_time2 = time.time()

            pred_global = clf.predict(data_global)

            ac = get_accuracy(pred, label)

            Experiment_result[0, iter_num] = ac[-1] * 100  # OA
            Experiment_result[1, iter_num] = np.mean(ac[:-2]) * 100  # AA
            Experiment_result[2, iter_num] = ac[-2] * 100  # Kappa
            Experiment_result[3, iter_num] = train_time2 - train_time1
            Experiment_result[4, iter_num] = tes_time2 - tes_time1
            Experiment_result[5:, iter_num] = ac[:-2] * 100

            print('########### Experiment {}，Model assessment Finished！ ###########'.format(iter_num))

            ########## mean value & standard deviation #############

        Experiment_result[:, -2] = np.mean(Experiment_result[:, 0:-2], axis=1)  # 计算均值
        Experiment_result[:, -1] = np.std(Experiment_result[:, 0:-2], axis=1)  # 计算平均差

        print('OA_std={}'.format(Experiment_result[0, -1]))
        print('AA_std={}'.format(Experiment_result[1, -1]))
        print('Kappa_std={}'.format(Experiment_result[2, -1]))
        print('time training cost_std{:.4f} secs'.format(Experiment_result[3, -1]))
        print('time testing cost_std{:.4f} secs'.format(Experiment_result[4, -1]))
        for i in range(Experiment_result.shape[0]):
            if i > 4:
                print('Class_{}: accuracy_std {:.4f}.'.format(i - 4, Experiment_result[i, -1]))  # 均差

        day = datetime.datetime.now()
        day_str = day.strftime('%m_%d_%H_%M')

        f = open('./record/' + str(day_str) + '_' + dataname + '_' + model_name +'_'+str(num)+ 'num.txt', 'w')
        for i in range(Experiment_result.shape[0]):
            f.write(str(i + 1) + ':' + str(round(Experiment_result[i, -2],2)) + '+/-' + str(round(Experiment_result[i, -1],2)) + '\n')
        for i in range(Experiment_result.shape[1] - 2):
            f.write('Experiment_num' + str(i) + '_OA:' + str(Experiment_result[0, i]) + '\n')
        f.close()