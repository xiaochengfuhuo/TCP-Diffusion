import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import cv2
from datetime import datetime,timezone
import netCDF4 as nc
from tqdm import tqdm
import torch.nn.functional as F
# from .train import load_data_once
from einops import rearrange
import torch
import random
torch.random.seed()
np.random.seed(0)
import matplotlib.pyplot as plt
dataset_path_2020 = 'J:\BP1backup\ICML_subset2020'  # The path of subset



def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds

class rainfall_data_multi(Dataset):
    def __init__(self,data_type,rainfall=None,modals=None,env_data=None,multi_modal='',
                 obs_num = 4,pre_num = 2,img_size=100,input_transform_key='01',
                 data_augmentation=False,env_frame=4):

        print('loading the rainfall data : '+data_type)
        self.rain_fall=rainfall
        self.env_data = env_data
        # self.data_path = data_path
        self.obs_num = obs_num
        self.pre_num = pre_num
        self.img_size = img_size
        self.data_list = np.load(os.path.join(dataset_path_2020,data_type+'.npy'),allow_pickle=True)
        self.datalist_filter()
        # self.data_list = [sample for sample in self.data_list if sample[0]['year']!=2020]
        self.multi_modal = multi_modal
        self.data_augmentation = data_augmentation
        self.env_frame = env_frame
        # self.just_one = just_one
        # self.multi_time = multi_time
        assert input_transform_key in ['01','loge','sqrt'], 'check the normalization method!'
        self.input_transform_key = input_transform_key
        print('--normalization method is ' + input_transform_key)
        print('--obs_time ' + str(self.obs_num))
        print('--pre_time ' + str(self.pre_num))
        print('--multi_modal ' + str(self.multi_modal))
        print('--data_augmentation ' + str(self.data_augmentation))


        # for standard normal
        self.mean_value = np.mean(self.rain_fall)
        self.std_dev = np.std(self.rain_fall)
        print('mean is '+ str(self.mean_value))
        print('std is ' + str(self.std_dev))
        self.data_transform()



        print('loading the multi_modal data')
        self.dataset_roots = modals


        self.dataset_normalize = {
            'sst': (246.9372599, 310.29638671875),
            't2m': (246.9372599, 319.1227845),
            'msl': (93137.3759, 103474.4268),
            'z_200_': (114269.5519, 123746.875),
            'z_600_': (37223.24629, 44474.36399),
            'z_850_': (7864.913365, 16159.62581),
            'z_925_': (528.1218262, 9061.944304),
            't_200_': (205.4924104, 236.9169874),
            't_600_': (249.9107088, 292.5276622),
            't_850_': (265.3704041, 307.8244362),
            't_925_': (266.5474339, 313.981372),
            'q_200_': (-0.000155115, 0.000702596),
            'q_600_': (3.01590769944543E-07, 0.014554153),
            'q_850_': (9.12206086884151E-07, 0.0223369046048584),
            'q_925_': (2.28928862341937E-06, 0.024116571599734),
            'u_200_': (-50.3288295, 73.14137885),
            'u_600_': (-63.26103189, 65.79474437),
            'u_850_': (-70.79315186, 69.08036606),
            'u_925_': (-68.18152412, 64.11427186),
            'v_200_': (-68.61412644, 74.65857924),
            'v_600_': (-62.02254669, 67.49864796),
            'v_850_': (-68.00113911, 76.64611816),
            'v_925_': (-66.44315571, 68.26896275),
            'topo': (-7906, 4500),

            'tp_ifs': (0, 0.06988716125488281),
            't2m_ifs': (242.53342598392996, 319.82794291554035),
            'msl_ifs': (94521.02333366396, 104166.30707048357),
            't_200_ifs': (204.02873631445442, 235.28852520565587),
            't_850_ifs': (260.0148344765363, 308.69361966130657),
            'q_200_ifs': (-4.0282640838995576e-05, 0.000435580859391594),
            'q_850_ifs': (1.5564412766243957e-06, 0.02220068623410043),
            'u_200_ifs': (-46.498877842765054, 83.27519730501616),
            'u_850_ifs': (-66.13909912109375, 53.6465625766141),
            'v_200_ifs': (-73.85488265328311, 83.86069117690691),
            'v_850_ifs': (-59.1883486895494, 62.60627350332703),


            'diff_loge':(-5.370121099,	5.3104655),
            'diff_sqrt': (-24.36050237,	20.13841541),
            # 'diff_original':(-693.875,	580.25),
            'diff_original': (-100, 100),
            'loge':(0,	6.549829553),
            'sqrt': (0, 26.42205518),
            'original': (0, 698.125),
            'tp_ifs':(0,69.88716125488281)


        }

    def data_transform(self):
        if self.input_transform_key == '01':
            self.rain_fall = self.rain_fall
        elif self.input_transform_key == 'loge':
            self.rain_fall = np.log(self.rain_fall+1)
        elif self.input_transform_key == 'sqrt':
            self.rain_fall = np.sqrt(self.rain_fall)

    def data_untransform(self,data):
        if isinstance(data, np.ndarray):
            if self.input_transform_key == '01':
                data_real = data
            elif self.input_transform_key == 'loge':
                data_real = np.exp(data) - 1
            elif self.input_transform_key == 'sqrt':
                data_real = data ** 2
        elif isinstance(data, torch.Tensor):
            if self.input_transform_key == '01':
                data_real = data
            elif self.input_transform_key == 'loge':
                data_real = torch.exp(data) - 1
            elif self.input_transform_key == 'sqrt':
                data_real = data ** 2
        return data_real


    def normalize(self,data):
        if self.input_transform_key == '01':
            min_data, max_data = (0,100)
        elif self.input_transform_key == 'loge':
            min_data, max_data = self.dataset_normalize['loge']
        elif self.input_transform_key == 'sqrt':
            min_data, max_data = self.dataset_normalize['sqrt']
        data_normal = (data - min_data) / (max_data - min_data)

        return data_normal

    def un_normalize(self,data):
        if self.input_transform_key == '01':
            min_data, max_data = (0,100)
        elif self.input_transform_key == 'loge':
            min_data, max_data = self.dataset_normalize['loge']
        elif self.input_transform_key == 'sqrt':
            min_data, max_data = self.dataset_normalize['sqrt']
        data_unnormal = data * (max_data - min_data) + min_data
        return data_unnormal

    def normalize_diff(self,diff):
        if self.input_transform_key == '01':
            min_diff, max_diff = self.dataset_normalize['diff_original']
        elif self.input_transform_key == 'loge':
            min_diff, max_diff = self.dataset_normalize['diff_loge']
        elif self.input_transform_key == 'sqrt':
            min_diff, max_diff = self.dataset_normalize['diff_sqrt']

        diff_normal = (diff - min_diff)/(max_diff - min_diff)
        return diff_normal

    def un_normalize_diff(self,diff):
        if self.input_transform_key == '01':
            min_diff, max_diff = self.dataset_normalize['diff_original']
        elif self.input_transform_key == 'loge':
            min_diff, max_diff = self.dataset_normalize['diff_loge']
        elif self.input_transform_key == 'sqrt':
            min_diff, max_diff = self.dataset_normalize['diff_sqrt']
        diff_real = diff * (max_diff - min_diff) + min_diff
        return diff_real

    def load_multimoda(self,modal):
        file_path = '/user/work/eg23371/ERA5_center/'+modal+'_merge.npy'
        return np.load(file_path,allow_pickle=True).item()

    def datalist_filter(self,year_filter_list=[1979]):
        print('filter--no '+str(year_filter_list))
        new_list = []
        for tc_dic in self.data_list:
            year = tc_dic[0]['year']
            if year not in year_filter_list:
                new_list.append(tc_dic)
        self.data_list = new_list


    def get_multi_modal_data(self,tc_dic):
        year = tc_dic['year']
        month = tc_dic['month']
        day = tc_dic['day']
        hour = tc_dic['hour']
        lat = tc_dic['centre_lat']
        lon = tc_dic['centre_lon']

        tc_modals = {}


        for modal in self.multi_modal:

            file_name = (str(year) + str('{:02d}'.format(month)) +
                        str('{:02d}'.format(day)) + str('{:02d}'.format(hour)))+ '_' + str(float(lat)) + '_' + str(float(lon))

            # rainfall_test =self.rain_fall[tc_dic['data_i']]
            # plt.imshow(rainfall_test)
            # plt.show()

            nc_file = self.dataset_roots[modal][file_name]
            # if len(nc_file.shape) == 2:
            #     plt.imshow(nc_file)
            #     plt.show()
            # else:
            #     plt.imshow(nc_file[0])
            #     plt.show()
            # nc_file = np.fliplr(nc_file)
            if len(nc_file.shape) == 2:
            #     # I suddenly found some data is 1,40,40  and  some of them are 40,40
                nc_file = np.flipud(nc_file)
            elif len(nc_file.shape) == 3:
                ###################
                nc_file = np.fliplr(nc_file)
            else:
                print('ERA5 data error!')
                exit()
            # if len(nc_file.shape) == 2:
            #     plt.imshow(nc_file)
            #     plt.show()
            # else:
            #     plt.imshow(nc_file[0])
            #     plt.show()
            tc_modals[modal] = torch.tensor(nc_file.copy())
            # plt.imshow(tc_modals[modal])
            # plt.show()

        return tc_modals

    def get_env_data(self,tc_dic):
        year = tc_dic['year']
        month = tc_dic['month']
        day = tc_dic['day']
        hour = tc_dic['hour']
        lat = tc_dic['centre_lat']
        lon = tc_dic['centre_lon']

        file_name = (str(year) + str('{:02d}'.format(month)) +
                    str('{:02d}'.format(day)) + str('{:02d}'.format(hour)))+ '_' + str(float(lat)) + '_' + str(float(lon))

        env_oneframe = self.env_data[file_name].copy()
        filter_list = ['location','history_direction6','history_direction12',
                       'future_direction24','history_inte_change12','future_inte_change12']
        env_oneframe_filter = {}
        for key in env_oneframe:
            if key in filter_list:
                continue
            env_oneframe_filter[key] = torch.tensor(env_oneframe[key])
        return env_oneframe_filter

    def input_merge(self,modals_env):
        input_list = []
        input_list_tptest = []
        ifs_flag = 0
        for key in modals_env:
            if '4test' in key:
                ifs_flag=1
                modal_data = modals_env[key]['pre'].squeeze(1).permute(3, 0, 1, 2)
                # variable_num t h w
                modal_data_resize = F.interpolate(modal_data, size=(64, 64), mode='bilinear', align_corners=False)
                input_list_tptest.append(modal_data_resize)
            else:
                # print(modals_env[key]['obs'].shape)
                # 4, 50, 50, 1
                # modal_data = modals_env[key]['obs'][-1,20:20+10,20:20+10,0].unsqueeze(0).unsqueeze(0)
                # print(modals_env[key]['obs'].shape)
                modal_data = modals_env[key]['obs'].squeeze(1).permute(3,0,1,2)
                modal_data = modal_data[:,-self.env_frame:]
                # variable_num t h w
                modal_data_resize = F.interpolate(modal_data, size=(64,64), mode='bilinear', align_corners=False)
                # modal_data_resize = modal_data_resize.squeeze(0)

                input_list.append(modal_data_resize)

        if len(input_list)>0:
            input = torch.cat(input_list,dim=0)
        else:
            input = torch.tensor(np.array([0,0]))
        if len(input_list_tptest)>0:
            input_ifs = torch.cat(input_list_tptest, dim=0)
        else:
            input_ifs = torch.tensor(np.array([0,0]))
        # variable_num t h w
        return {'modal':input,'ifs':input_ifs,'ifs_flag':ifs_flag}

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        # idx = idx+1
        normalizer_index = 100
        TC_dic = self.data_list[idx]
        rain_list = []
        modals_list = []
        rain_diff_list = []
        env_list = []
        modals_tensor = {modal:[] for modal in self.multi_modal}
        # pre_rain_list = []

        # get source data
        # source_data, source_data2 = self.read_source_data(TC_dic)


        for frame_i, one_frame in enumerate(TC_dic):
            #-98000 just for review
            rain_data = self.rain_fall[one_frame['data_i']-98000]
            if rain_data.shape[-1] != self.img_size:
                rain_data = cv2.resize(rain_data, (self.img_size, self.img_size))
            rain_list.append(torch.tensor(rain_data))

            if frame_i == 0:
                rain_diff_list.append(torch.tensor(np.zeros_like(rain_data)))
            else:
                rain_pre = self.rain_fall[TC_dic[frame_i-1]['data_i']-98000]
                if rain_pre.shape[-1] != self.img_size:
                    rain_pre = cv2.resize(rain_pre, (self.img_size, self.img_size))
                rain_diff_list.append(torch.tensor(rain_data-rain_pre))

            #get other data


            tc_modals = self.get_multi_modal_data(one_frame)
            modals_list.append(tc_modals)

            if self.env_data is not None:
                env_list.append(self.get_env_data(one_frame))
                #********************************************



        # normalization
        # if self.input_transform_key == '01':
        rain_tensor = torch.nan_to_num(torch.stack(rain_list,dim=0).unsqueeze(-1)).to(dtype=torch.float32)
        rain_tensor = self.normalize(rain_tensor)
        rain_diff_tensor = torch.nan_to_num(torch.stack(rain_diff_list, dim=0).unsqueeze(-1)).to(dtype=torch.float32)
        rain_diff_tensor = self.normalize_diff(rain_diff_tensor)



        if self.env_data is not None:
            env_tensor = {env_name: [] for env_name in env_list[0]}
        # list[dic{x:np}]  -->  dic{x:list[np]}
            for env_one in env_list:
                for key_env_one in env_one:
                    env_tensor[key_env_one].append(env_one[key_env_one])

            for key_env_tensor in env_tensor:
                env_tensor[key_env_tensor] = torch.stack(env_tensor[key_env_tensor], dim=0)
                env_tensor[key_env_tensor] = env_tensor[key_env_tensor].to(dtype=torch.float32)
                # env_tensor[key_env_tensor]['obs'] = env_tensor[key_env_tensor][:self.obs_num]
                # env_tensor[key_env_tensor]['pre'] = env_tensor[key_env_tensor][self.obs_num:]
        else:
            env_tensor = 0






        if len(self.multi_modal)>0:
            # list[dic{x:np}]  -->  dic{x:list[np]}
            for modal_data in modals_list:
                for key in modal_data:
                    modals_tensor[key].append(modal_data[key])

            # dic{x:list[np]}  -->  dic{x:{obs:tensor[t,h,w,c],pre:tensor[t,h,w,c]}}
            modals_ear5 = {modal: {} for modal in self.multi_modal}
            for key in modals_tensor:
                modals_tensor[key] = torch.nan_to_num(torch.stack(modals_tensor[key], dim=0).unsqueeze(-1))
                modals_tensor[key] = modals_tensor[key].to(dtype=torch.float32)
                # print(modals_tensor[key].shape)
                modals_tensor[key] = (modals_tensor[key]-self.dataset_normalize[key][0])/\
                                     (self.dataset_normalize[key][1]-self.dataset_normalize[key][0])
                # print(modals_tensor.keys())
                modals_ear5[key]['obs'] = modals_tensor[key][:self.obs_num]
                modals_ear5[key]['pre'] = modals_tensor[key][self.obs_num:]



        obs_rain = rain_tensor[:self.obs_num]
        pre_rain = rain_tensor[self.obs_num:]

        obs_diff = rain_diff_tensor[:self.obs_num]
        pre_diff = rain_diff_tensor[self.obs_num:]


        obs_TCinfor = TC_dic[:self.obs_num]
        pre_TCinfor = TC_dic[self.obs_num:]


        if len(self.multi_modal)==0:
            # x -> b c f h w
            return (obs_rain[-self.obs_num:].permute(3, 0, 1,2),
                    pre_rain[:self.pre_num].permute(3, 0, 1,2),
                    obs_diff[-self.obs_num:].permute(3, 0, 1, 2),
                    pre_diff[:self.pre_num].permute(3, 0, 1, 2),
                    torch.tensor(np.array([0,0])),   #  for modal
                    obs_TCinfor,pre_TCinfor,
                    env_tensor,
                    torch.tensor(np.array([0,0])))
        else:
            all_data = self.input_merge(modals_ear5)   # c modals_num h w
            modals_data = all_data['modal']
            ifs = all_data['ifs']
            return (obs_rain[-self.obs_num:].permute(3, 0, 1,2),
                    pre_rain[:self.pre_num].permute(3, 0, 1,2),
                    obs_diff[-self.obs_num:].permute(3, 0, 1, 2),
                    pre_diff[:self.pre_num].permute(3, 0, 1, 2),
                    modals_data,
                    obs_TCinfor,pre_TCinfor,
                    env_tensor,
                    ifs)


    def random_crop(self,tensor):
        b, c, f, h, w = tensor.shape
        crop_size = 60
        top = torch.randint(0, tensor.shape[3] - crop_size + 1, (1,)).item()
        left = torch.randint(0, tensor.shape[4] - crop_size + 1, (1,)).item()

        cropped = tensor[:, :, :, top:top + crop_size, left:left + crop_size]
        cropped_orginal = F.interpolate(cropped, size=(64, 64), mode='bilinear', align_corners=False)

        return cropped_orginal

    # def flip(self,tensor):
    #     if random.random() < 0.5:
    #         torch.flip(tensor, [4])
    #     else:
    #         torch.flip(tensor, [3])

    def collate_data(self,data_list):
        # print(type(data_list[0]))
        obs_rain, pre_rain,obs_diff,pre_diff, modals_data , obs_TCinfor, pre_TCinfor = [], [], [], [], [],[], []
        ifs_pre = []
        if self.env_data is None:
            env_data = 0
        else:
            env_data = {env_name: [] for env_name in data_list[0][7]}
        # env_tensor = {env_name: [] for env_name in env_list[0]}

        for data_one in data_list:
            # print(data_one[0])
            obs_rain.append(data_one[0])
            pre_rain.append(data_one[1])
            obs_diff.append(data_one[2])
            pre_diff.append(data_one[3])
            modals_data.append(data_one[4])
            obs_TCinfor.append(data_one[5])
            pre_TCinfor.append(data_one[6])
            if self.env_data is not None:
                for env_one in data_one[7]:
                    env_data[env_one].append(data_one[7][env_one])
            ifs_pre.append(data_one[8])



        obs_rain = torch.stack(obs_rain, dim=0)
        pre_rain = torch.stack(pre_rain, dim=0)
        obs_diff = torch.stack(obs_diff, dim=0)
        pre_diff = torch.stack(pre_diff, dim=0)
        modals_data = torch.stack(modals_data, dim=0)
        ifs_pre = torch.stack(ifs_pre, dim=0)
        if self.env_data is not None:
            for key_env_data in env_data:
                env_data[key_env_data] = torch.stack(env_data[key_env_data], dim=0)
                if len(env_data[key_env_data].shape) == 2:
                    env_data[key_env_data] = env_data[key_env_data].unsqueeze(-1)
                # env_data[key_env_data]['obs'] = env_data[key_env_data][:,:self.obs_num]
                # env_data[key_env_data]['pre'] = env_data[key_env_data][:,self.obs_num:]

        if len(modals_data.shape) != len(obs_rain.shape):
            modals_data = torch.zeros_like(obs_rain)
        if len(ifs_pre.shape) != len(obs_rain.shape):
            ifs_pre = torch.zeros_like(obs_rain)
        # print(obs_rain.shape)

        if self.data_augmentation:
            data_aug_list = [obs_rain, pre_rain, obs_diff, pre_diff, modals_data,ifs_pre]
            if random.random() < 0.5:
                # 随机选择旋转角度
                rotation_angle = torch.randint(0, 4, (1,)).item() * 90
                for data_aug_i,_ in enumerate(data_aug_list):
                    data_aug_list[data_aug_i] = torch.rot90(data_aug_list[data_aug_i], k=rotation_angle // 90,dims=(-2, -1))


            if random.random() < 0.5:
                for data_aug_i,_ in enumerate(data_aug_list):
                    # data_aug_list[data_aug_i] = torch.flip(data_aug_list[data_aug_i], [-1])
                    b, c, f, h, w = data_aug_list[data_aug_i].shape
                    crop_size = 60
                    top = torch.randint(0, data_aug_list[data_aug_i].shape[-2] - crop_size + 1, (1,)).item()
                    left = torch.randint(0, data_aug_list[data_aug_i].shape[-1] - crop_size + 1, (1,)).item()

                    cropped = data_aug_list[data_aug_i][:, :, :, top:top + crop_size, left:left + crop_size]
                    cropped = rearrange(cropped,'b c f h w -> b (c f) h w')
                    cropped_hw = F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
                    cropped_hw = rearrange(cropped_hw, 'b (c f) h w -> b c f h w', c=c, f=f)
                    data_aug_list[data_aug_i] = cropped_hw

            obs_rain,pre_rain,obs_diff,pre_diff,modals_data,ifs_pre = tuple(data_aug_list)



        # x -> b c f h w
        return {'obs_rain': obs_rain, 'pre_rain': pre_rain,'obs_diff': obs_diff, 'pre_diff': pre_diff,
                'obs_TCinfor': obs_TCinfor, 'pre_TCinfor': pre_TCinfor, 'modal_env':modals_data,
                'env_data':env_data,'ifs':ifs_pre}





def load_data_once(multi_modal):
    # root = 'J:\BP1backup\ICML_subset2020'
    def load_multimoda(modal):
        if 'ifs' in modal:
            if '200' in modal or '850' in modal:
                file_path = os.path.join(dataset_path_2020,'Future_Prediction_data','ERA5IFS_pressure_level_crop/' + modal[0] + '/' + modal + '.npy')
            else:
                file_path = os.path.join(dataset_path_2020,'Future_Prediction_data','ERA5IFS_sst_tp_msl_2mt_crop/' + modal + '/' + modal + '.npy')
        elif '200' in modal or '600' in modal or '850' in modal or '925' in modal:
            file_path = os.path.join(dataset_path_2020,'Historical_Data','Environment_Data','X_PlEnv','gph_tquv_crop/' +modal[0]+'/'+modal+'.npy')
        elif modal == 'topo':
            file_path = os.path.join(dataset_path_2020,'Historical_Data','Environment_Data','X_SfEnv','topography_crop/topography_crop.npy')
        elif modal == 'tp4test':
            file_path = os.path.join(dataset_path_2020,'Future_Prediction_data','ERA5IFS_sst_tp_msl_2mt_crop/tp/tp.npy')
        else:
            file_path = os.path.join(dataset_path_2020,'Historical_Data','Environment_Data','X_SfEnv','sst_tp_msl_2mt_crop/'+modal+'/'+modal+'_nonan.npy')
        # file_path = '/user/work/eg23371/ERA5_center/'+modal+'_merge.npy'
        return np.load(file_path,allow_pickle=True).item()
    print('loading the multi_modal data')
    modals = {}
    for modal in multi_modal:
        modals[modal] = load_multimoda(modal)

    rain_fall = np.load(os.path.join(dataset_path_2020,'Historical_Data','Rainfall_Data', 'storm_rain_extended_2020.npy'), allow_pickle=True)

    env_TCrainfall = np.load(os.path.join(dataset_path_2020,'Historical_Data','Environment_Data','X_Sc','env_rainfall_2020.npy'), allow_pickle=True).item()

    return {'rainfall':rain_fall,'modals': modals,'env':env_TCrainfall}



if __name__ == '__main__':

    file_path = os.path.join('..','modal_txt', 'tquvz_t2m_sst_msl_topo_ifs.txt')

    multi_modal = []
    modal_channle = 0
    ifs_channle = 0


    with open(file_path, 'r') as file:
        for line in file:

            line = line.strip()
            for M in str.split(line, ' '):
                if 'ifs' in M:
                    ifs_channle = ifs_channle + 1
                else:
                    modal_channle = modal_channle + 1
                multi_modal.append(M)


    Alldata = load_data_once(multi_modal)
    print('multi_modal loading completed!')
    rainfall = Alldata['rainfall']
    modals = Alldata['modals']
    env_data = Alldata['env']
    dataset = rainfall_data_multi('test_MSWEP_in4_out4_2020', rainfall=rainfall, modals=modals,env_data=env_data,
                                   multi_modal=multi_modal,
                                   img_size=64, pre_num=4,env_frame=4,data_augmentation=False)

    loader = DataLoader(dataset, batch_size=16,
                        shuffle=False,
                        num_workers=1,collate_fn=dataset.collate_data)
    pbar = tqdm(loader)

    ifs_list = []
    # x -> b c f h w
    information_list = []
    for batch in pbar:

        obs = batch['obs_rain']
        pre = batch['pre_rain']
        # ifs = batch['modal_env']
        sst = batch['modal_env']
        env_data = batch['env_data']
        # ifs_list.append(ifs)
        pbar.set_description(str(env_data['wind'].shape)+'_'+str(sst.shape))
        information_list = information_list+batch['obs_TCinfor']

