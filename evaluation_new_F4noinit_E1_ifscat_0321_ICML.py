import torch

from video_diffusion_pytorch.rainfall_diffusion_F4_E1ifs_0316 import Unet3D, GaussianDiffusion, Trainer
import argparse
from video_diffusion_pytorch.Env_transformer import Env_net
from video_diffusion_pytorch.rainfall_dataset_ICML import rainfall_data_multi,load_data_once
from torch.utils.data import Dataset,DataLoader
from utils import *
from matplotlib.colors import LinearSegmentedColormap,BoundaryNorm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


parser = get_parser()
args = parser.parse_args()
file_path = os.path.join('modal_txt',args.multi_modals+'.txt')

multi_modal = []
modal_channle = 0
ifs_channle = 0
ifs_out_channel=3

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        for M in str.split(line,' '):
            if 'ifs' in M:
                ifs_channle = ifs_channle+1
            else:
                modal_channle = modal_channle+1
            multi_modal.append(M)


model = Unet3D(
    dim=64,
    channels=2+modal_channle+1,
    # channels=4,
    out_dim=1,
    dim_mults=(1, 2, 4, 8),
    cond_dim=args.cond_dim,
    ifs_channels=ifs_channle,
    ifs_out_channel=ifs_out_channel
).cuda()

# x -> b c f h w

env_list = ['wind','intensity_class','move_velocity','month',
            'location_long','location_lat',
            # 'history_direction6', 'history_direction12','history_inte_change12'
            ]

cond_encoder = Env_net(env_list=env_list).cuda()
# x -> b c f h w
diffusion = GaussianDiffusion(
    model,
    cond_encoder=cond_encoder,   #  if no condition -> #
    image_size = args.img_size,
    num_frames = args.output_frames+args.input_frames,
    input_frames=args.input_frames,
    output_frames=args.output_frames,
    channels =1,
    timesteps = args.timesteps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    obs_channels=args.obs_channels
).cuda()




trainer = Trainer(
    diffusion,
    './data',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = args.train_batch_size,
    train_lr = 1e-4,
    save_and_sample_every = 5000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                        # turn on mixed precision
    # cond_encoder=cond_encoder
    results_folder=args.save

)


def save_sample(root_path,data_p,data_t,count_i,epoch):
    cmap, norm = colormap()
    save_path = os.path.join(root_path,epoch_name+str(epoch),'predictions_'+str(epoch))
    os.makedirs(save_path,exist_ok=True)
    # os.makedirs(os.path.join(root_path,'target'))
    batch_shape = data_p.shape
    # print(batch_shape,len(batch_shape))
    batch = batch_shape[0]
    if len(batch_shape) == 4:
        c = batch_shape[1]
    else:
        c = 1
    for batch_i in range(batch):
        for c_i in range(c):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            if len(batch_shape) == 4:
                img_p = data_p[batch_i, c_i]
                img_t = data_t[batch_i, c_i]
            else:
                img_p = data_p[batch_i]
                img_t = data_t[batch_i]

            img1 = ax1.imshow(img_t, cmap=cmap, norm=norm)
            ax1.set_title('groud truth')
            plt.colorbar(img1, ax=ax1)

            img2 = ax2.imshow(img_p, cmap=cmap, norm=norm)
            plt.colorbar(img2, ax=ax2)
            ax2.set_title('prediction')

            plt.savefig(save_path + '/' + str(batch_i+count_i)+'_'+str(c_i)+'.png')
            plt.close()

def save_all_step(root_path,data_t,data_p,count_i):
    cmap, norm = colormap()
    save_path = os.path.join(args.save,root_path,'process')
    os.makedirs(save_path,exist_ok=True)
    # os.makedirs(os.path.join(root_path,'target'))
    batch_shape = data_p.shape
    step = batch_shape[0]
    if len(batch_shape) == 4:
        c = batch_shape[1]
    else:
        c = 1
    for step_i in range(step):
        for c_i in range(c):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            if len(batch_shape) == 4:
                img_p = data_p[step_i, c_i]
                img_t = data_t[step_i, c_i]
            else:
                img_p = data_p[step_i]
                img_t = data_t[0]

            img1 = ax1.imshow(img_t, cmap=cmap, norm=norm)
            ax1.set_title('groud truth')
            plt.colorbar(img1, ax=ax1)

            img2 = ax2.imshow(img_p, cmap=cmap, norm=norm)
            plt.colorbar(img2, ax=ax2)
            ax2.set_title('prediction')

            plt.savefig(save_path + '/' + str(step_i)+'_'+str(c_i)+'.png')
            plt.close()

def save_result(CSI, HSS,ETS,preds,target,mse,mae,threshold,step,epoch):
    print('Saving some results on step ->'+step)
    ps = np.concatenate(preds, axis=0)

    ts = np.concatenate(target, axis=0)

    np.save(os.path.join(args.save,epoch_name+str(epoch),'prediction_all.npy'),ps)
    np.save(os.path.join(args.save, epoch_name + str(epoch), 'gt_all.npy'), ts)

    for i in range(3):
        CSI[i] = np.array(CSI[i]).mean()
        HSS[i] = np.array(HSS[i]).mean()
        ETS[i] = np.array(ETS[i]).mean()
    mse = np.array(mse).mean()
    mae = np.array(mae).mean()
    f = open(os.path.join(args.save, epoch_name + str(epoch), 'result_' + str(epoch) + '.txt'), 'a+')

    f.write('CSI: ')
    print('CSI: ')
    for i in range(len(threshold)):
        f.write('r >= ' + str(threshold[i]) + ':' + str(CSI[i]) + ' ')
        print('r >=', threshold[i], ':', CSI[i], end=' ')
    print()

    f.write('HSS:')
    print('HSS:')
    for i in range(len(threshold)):
        f.write('r >= ' + str(threshold[i]) + ':' + str(HSS[i]) + ' ')
        print('r >=', threshold[i], ':', HSS[i], end=' ')
    print()

    f.write('ETS:')
    print('ETS:')
    for i in range(len(threshold)):
        f.write('r >= ' + str(threshold[i]) + ':' + str(ETS[i]) + ' ')
        print('r >=', threshold[i], ':', ETS[i], end=' ')
    print()

    f.write('MSE:' + str(mse) + 'MAE:' + str(mae))
    print('MSE:', mse, 'MAE:', mae)


def sample_test(epoch=100,pre_len=1,output_dirpath='',break_time=0):

    preds=[]
    target=[]
    Alldata = load_data_once(multi_modal)
    rainfall = Alldata['rainfall']
    modals = Alldata['modals']
    env_data = Alldata['env']

    rainfall_dic = 'test_MSWEP_in4_out4_2020'
    ds = rainfall_data_multi(rainfall_dic, rainfall=rainfall, modals=modals,env_data=env_data,
                                       multi_modal=multi_modal,img_size=args.img_size, pre_num=pre_len,
                             input_transform_key=args.input_transform_key,data_augmentation=False)

    # print(ds.__len__())
    epoch = trainer.load(epoch)
    save_path = os.path.join(output_dirpath, epoch_name + str(epoch), 'predictions_' + str(epoch))
    os.makedirs(save_path, exist_ok=True)

    dl = DataLoader(ds, batch_size = args.train_batch_size, shuffle=False, pin_memory=True,collate_fn=ds.collate_data)
    CSI, HSS,ETS, mse, mae = [], [], [], [],[]
    for i in range(3):
        CSI.append([])
        HSS.append([])
        ETS.append([])
    threshold = [6, 24, 60]
    dl_t = tqdm(dl)

    for i,batch in enumerate(dl_t):
        if i >break_time:
            break
        data_obs_real = batch['obs_rain'].cuda()
        data_traget = batch['pre_rain'].cuda()
        data_obs_diff = batch['obs_diff'].cuda()
        data_traget_diff = batch['pre_diff'].cuda()
        ERA5_obs = batch['modal_env'].cuda()

        Env_obs = batch['env_data']
        for key in Env_obs:
            Env_obs[key] = Env_obs[key].cuda()

        data_obs = torch.cat([data_obs_diff, data_obs_real, ERA5_obs], dim=1)


        data_condition = Env_obs

        samples_torch = trainer.model.sample(batch_size=args.train_batch_size,
                                                           data_condition=data_condition,
                                                           obs_data=data_obs
                                                           )

        target_batch,np_samples = diff_test_unnormalize(ds=ds,samples_torch=samples_torch,data_traget=data_traget,data_obs_real=data_obs_real)
        target_batch = target_batch.squeeze().cpu().numpy()
        np_samples = np_samples.squeeze().cpu().numpy()
        # print(target_batch.shape, np_samples.shape)
        # print(target_batch.shape,np_samples.shape)

        count_i = args.train_batch_size*i

        if i < 10:
            save_sample(output_dirpath, np_samples, target_batch, count_i,epoch)

        mse.append(mse_evaluation(target_batch, np_samples))
        mae.append(mae_evaluation(target_batch, np_samples))

        for t in range(3):
            a = np_samples.copy()
            b = target_batch.copy()
            thre = threshold[t]
            a[a < thre] = 0
            a[a >= thre] = 1
            b[b < thre] = 0
            b[b >= thre] = 1
            CSI[t].append(csi_single(a, b))
            HSS[t].append(hss_single(a, b))
            ETS[t].append(ETS_single(a, b))


        preds.append(np_samples)
        target.append(target_batch)

        if i % 30 == 29:
            save_result(CSI.copy(), HSS.copy(), ETS.copy(),
                        preds.copy(), target.copy(),
                        mse.copy(), mae.copy(), threshold.copy(),step=str(i),epoch=epoch)


    save_result(CSI.copy(), HSS.copy(), ETS.copy(),
                preds.copy(), target.copy(),
                mse.copy(), mae.copy(), threshold.copy(),step='last',epoch=epoch)

    # return ps, ts

epoch_name = 'ICML_under_review_'
if __name__ == '__main__':

    epoch = args.test_epoch
    if args.debug:
        break_time = 0
    else:
        break_time=6666
    # break_time for debug if it is set as 0.
    sample_test(epoch=epoch,pre_len=args.output_frames,output_dirpath=args.save,break_time=break_time)
# python evaluation_new_F4noinit_E1_ifscat_0321_ICML.py ICMLtest --save results/TCP_ICML_Test --output_frames 4 --train_batch_size 16 --test_epoch 55 --timesteps 200 --new_split --multi_modals tquvz_t2m_sst_msl_topo_ifs --input_transform_key loge --loss_type l2