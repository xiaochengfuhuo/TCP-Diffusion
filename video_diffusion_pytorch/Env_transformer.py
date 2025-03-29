import torch
from torch import nn

class Env_net(nn.Module):
    def __init__(self,env_list=[]):
        super(Env_net, self).__init__()

        embed_dim = 16
        out_dim = 256
        data_length = {'wind':1,'intensity_class':6,'move_velocity':1,'month':12,
                       'location_long':36,'location_lat':12,'history_direction6':8,
                       'history_direction12':8,'history_inte_change12':4}
        self.data_embed = nn.ModuleDict()
        for key in env_list:
            self.data_embed[key] =  nn.Linear(data_length[key], embed_dim)


        self.env_list = env_list

        env_f_in = len(self.data_embed)*16
        self.evn_extract = nn.Sequential(
            nn.Linear(env_f_in,env_f_in*2),
            nn.GELU(),
            nn.Linear(env_f_in*2, env_f_in * 2),
            nn.GELU(),
            nn.Linear(env_f_in * 2, out_dim)
        )


        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)


    def init_hidden(self, batch):
        return (
            torch.zeros(2, batch, 64).cuda(),
            torch.zeros(2, batch, 64).cuda()
        )

    def init_weights(self):
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in")
                m.bias.data.fill_(0.01)

        self.apply(init_kaiming)


    def forward(self,env_data,gph=None):
        '''

        :param env_data: b,pre_len,x_len
        :param gph: b,1,obs_len,h,w
        :return:
        '''
        # gph = gph.permute(0,2,1,3,4)
        # batch,pre_len,channel,h,w = gph.shape
        embed_list = []
        gph_list = []
        for key in self.env_list:
            # print(env_data[key].shape)
            now_embed = self.data_embed[key](env_data[key])
            embed_list.append(now_embed)
        # for i_len in range(pre_len):
        #     gph_list.append(self.GPH_embed(gph[:,i_len]).reshape(batch,-1))
        # gph_feature = torch.stack(gph_list,dim=1)
        # embed_list.append(gph_feature)
        embed = torch.cat(embed_list, dim=2)

        # embed_list.append(self.GPH_embed(gph.reshape(-1, channel, h, w)).reshape(batch, pre_len, -1))
#       batch,env_f_in

        feature_in = self.evn_extract(embed).permute(1,0,2)
        # time_weight = self.time_weight_emb(feature_in.permute(1,0,2).reshape(batch,-1)) #++
        # time_weight_0_1 = self.softmax(time_weight).unsqueeze(dim=-1)  # batch  obs_len++
        output = self.encoder(feature_in)
        feature = output[-1]

        return feature



if __name__ == '__main__':
    env_data = {}
    env_data['wind'] = torch.randn((4,8,1)).cuda()
    env_data['intensity_class'] = torch.randn((4,8,6)).cuda()
    env_data['move_velocity'] = torch.randn((4,8,1)).cuda()
    env_data['month'] = torch.randn((4,8,12)).cuda()
    env_data['location_long'] = torch.randn((4,8,36)).cuda()
    env_data['location_lat'] = torch.randn((4,8,12)).cuda()
    env_data['history_direction12'] = torch.randn((4,8,8)).cuda()
    env_data['history_direction24'] = torch.randn((4,8,8)).cuda()
    env_data['history_inte_change24'] = torch.randn((4,8,4)).cuda()



    gph = torch.randn((4,1,8,64,64)).cuda()

    env_net = Env_net().cuda()
    f,_,_  =  env_net(env_data,gph)
    print(f.shape)

