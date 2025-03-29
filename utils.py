import os.path

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,BoundaryNorm
from PIL import Image
import io
import json
import torch
def mse_evaluation(a, b):
    return np.mean(np.square((a - b)))

def mae_evaluation(a, b):
    return np.mean(np.abs((a - b)))

def tp(pre, gt):
    return np.sum(pre * gt)

def fn(pre, gt):
    a = pre + gt
    flag = (gt == 1) & (a == 1)
    return np.sum(flag)

def fp(pre, gt):
    a = pre + gt
    flag = (pre == 1) & (a == 1)
    return np.sum(flag)

def tn(pre, gt):
    a = pre + gt
    flag = a == 0
    return np.sum(flag)

def csi_single(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    return TP / (TP + FN + FP + eps)

def ETS_single(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    N = TP + FN + FP + TN

    TPr = (TP+FP)*(TP+FN)/N

    return (TP-TPr) / (TP + FN + FP + eps-TPr)

def hss_single(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)

    n = TP + FN + FP + TN
    aref = (TP + FN) / n * (TP + FP)
    gss = (TP - aref) / (TP + FN + FP - aref + eps)
    hss = 2 * gss / (gss + 1)

    # hss = ((TP * TN) - (FN * FP)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + eps)

    return hss

def crps_multivariate(observation, forecast_samples):
    print(observation.shape)
    if len(observation.shape) == 4:
        n_samples, n_variables, height, width = observation.shape
    else:
        n_samples, c,n_variables, height, width = observation.shape

    obs_flat = observation.reshape((n_samples, -1))
    forecast_flat = forecast_samples.reshape((n_samples, -1))

    obs_ranked = np.argsort(obs_flat, axis=1)
    forecast_ranked = np.argsort(forecast_flat, axis=1)

    cdf_observation = (np.arange(n_variables * height * width) < obs_ranked).mean(axis=1)
    cdf_forecast = (np.arange(n_variables * height * width) < forecast_ranked).mean(axis=1)

    crps_score = np.mean((cdf_forecast - cdf_observation)**2)

    return crps_score

def draw_distribustion(data_dic):

    label = data_dic.keys()
    colors = {}
    for l in label:
        if 'gt' in l:
            colors[l] = 'gray'
        else:
            colors[l] = 'red'

    for key in data_dic:
        if 'gt' in key:
            plt.hist(data_dic[key].flatten(), bins=100, alpha=1, label=key, color=colors[key])
            percentile_99 = np.percentile(data_dic[key].flatten(), 99)
            plt.axvline(percentile_99, color='red', linestyle='dashed', linewidth=1, label='99% Percentile')
        else:
            plt.hist(data_dic[key].flatten(), bins=100, alpha=1, label=key, color=colors[key],histtype='step', linewidth=1)




    plt.yscale('symlog')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Multiple Datasets')

    plt.legend()

    plt.show()
    plt.close()

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('log', help='')
    parser.add_argument('--save', default='', type=str)
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--test_epoch', default=33, type=int)
    parser.add_argument('--load_checkpoint',action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', action='store_true')


    #data
    parser.add_argument('--input_frames', default=4, type=int)
    parser.add_argument('--output_frames', default=4, type=int)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--multi_modals', default='', type=str)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--env_frame', default=4, type=int)
    parser.add_argument('--input_transform_key', default='loge', type=str)
    parser.add_argument('--new_split', action='store_true')
    # train_batch_size

    #model
    parser.add_argument('--cond_dim', default=256, type=int)
    parser.add_argument('--timesteps', default=200, type=int)
    parser.add_argument('--loss_type', default='l2', type=str)
    parser.add_argument('--obs_channels', default=1, type=int)

    save_parser(parser.parse_args())

    return parser

def save_parser(args):
    save_path = args.save
    args_dict = vars(args)

    os.makedirs(save_path,exist_ok=True)
    with open(os.path.join(save_path,args.log+'args.json'), 'w') as file:
        json.dump(args_dict, file, indent=4)

def colormap():
    colors = ['#fefffe', '#b2d0dd', '#97bcd6', '#799ec2', '#6ca45e',
              '#89b95e','#a7c17f','#e5f35b','#e7bd60','#e07066',
              '#e16a94','#dc5f9b','#b65dbf','#5743ec','#1c125b']

    bounds = [0, 1, 2, 3, 5,
              7, 10, 15, 20, 25,
              30,40,50,70,100,150]

    #


    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap,norm

def video_tensor_to_gif(tensor, path, duration = 1000, loop = 0, optimize = True):
    # tensor = tensor*100
    # print(tensor.min(), tensor.max())
    # print(tensor.shape)
    tensor_np = tensor.detach().squeeze().cpu().numpy()
    lll = tensor_np.shape[0]
    images = [tensor_np[i] for i in range(lll) ]
    # images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    cmap, norm = colormap()
    images_np = []
    for img in images:
        numpy_array = np.array(img)
        # print(numpy_array.min(),numpy_array.max())
        plt.imshow(numpy_array, cmap=cmap, norm=norm)
        # plt.colorbar()  #
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        #
        image = Image.open(buf)
        images_np.append(image)
        plt.close()

    images_np = tuple(images_np)

    first_img, *rest_imgs = images_np
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

def diff_test_unnormalize(ds,samples_torch,data_traget,data_obs_real):
    pre_diff_real = ds.un_normalize_diff(samples_torch)
    target_batch = ds.un_normalize(data_traget)
    data_obs_real = ds.un_normalize(data_obs_real)

    np_samples = torch.zeros_like(data_traget)
    # b,c,f,h,w
    print(pre_diff_real.shape, data_obs_real.shape)
    for i in range(np_samples.shape[2]):
        np_samples[:, :, i] = torch.sum(pre_diff_real[:, :, :i + 1], dim=2) + data_obs_real[:, :, -1]

    np_samples = ds.data_untransform(np_samples)
    np_samples = torch.clamp(np_samples, min=0)

    target_batch = ds.data_untransform(target_batch)
    return target_batch,np_samples

def diff_val_unnormalize(ds,all_videos_list,data_traget,data_obs_real):
    all_videos_list = ds.un_normalize_diff(all_videos_list)
    data_obs_real = ds.un_normalize(data_obs_real)
    data_traget = ds.un_normalize(data_traget)

    data_pre_real = torch.zeros_like(data_traget)
    for i in range(all_videos_list.shape[2]):
        data_pre_real[:, :, i] = torch.sum(all_videos_list[:, :, :i + 1], dim=2) + data_obs_real[:, :, -1]

    data_pre_real = ds.data_untransform(data_pre_real)
    data_traget = ds.data_untransform(data_traget)

    return data_pre_real, data_traget



def getdataset_name(args):
    print('loss_type:'+args.loss_type)
    if args.output_frames > 4:
        print('train_MSWEP_in4_out8', 'val_MSWEP_in4_out8')
        rainfall_dic_train = 'train_MSWEP_in4_out8'
        rainfall_dic_val = 'val_MSWEP_in4_out8'
    elif args.new_split:
        print('train_MSWEP_in4_out4_95', 'val_MSWEP_in4_out4_05')
        rainfall_dic_train = 'train_MSWEP_in4_out4_95'
        rainfall_dic_val = 'val_MSWEP_in4_out4_05'
    elif not args.new_split:
        print('train_MSWEP_in4_out4', 'val_MSWEP_in4_out4')
        rainfall_dic_train = 'train_MSWEP_in4_out4'
        rainfall_dic_val = 'val_MSWEP_in4_out4'

    return rainfall_dic_train,rainfall_dic_val



def seed():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)