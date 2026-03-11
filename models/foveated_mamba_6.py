import config
import torch.nn as nn
import torch


class FoveatedMamba6(nn.Module):

    def __init__(self, mamba, heads: nn.ModuleList, action_converter, backbones: nn.ModuleList, joiner,
                 foveation_properties, use_pos_emb=True, pos_emb_type='concat'):
        super().__init__()
        self.mamba = mamba
        self.heads = heads
        self.use_pos_emb = use_pos_emb
        self.action_converter = action_converter
        self.backbones = backbones
        self.joiner = joiner
        self.foveation_properties = foveation_properties
        # agent state
        self.pos_x, self.pos_y = None, None
        self.pos_emb_type = pos_emb_type
        self.delimiter_token = nn.Parameter(torch.randn(mamba.config.hidden_size) * .02)
        if pos_emb_type == 'concat':
            self.query_token = nn.Parameter(torch.randn(mamba.config.hidden_size - 2) * .02)
        else:
            self.query_token = nn.Parameter(torch.randn(mamba.config.hidden_size) * .02)

        # self.init_fix_logits = nn.Parameter(torch.ones(foveation_grid.num_actions) * 0.1)


    def pos_embed(self, token, pos_x, pos_y):
        if self.pos_emb_type == 'concat':
            if len(token.shape) == 3:
                token =  torch.cat([token, pos_x.unsqueeze(2) / 100, pos_y.unsqueeze(2) / 100], dim=2)
            else:
                token = torch.cat([token, pos_x.unsqueeze(1) / 100, pos_y.unsqueeze(1) / 100], dim=1)  # token += get_pos_emb(pos_x=self.pos_x, pos_y=self.pos_y, dim=token.shape[-1])
        elif self.pos_emb_type == 'add':
            pos_emb = get_pos_emb(pos_x, pos_y, dim=token.shape[-1])
            token = token + pos_emb
        return token


    def forward_seq(self, seq, pos_x, pos_y, query_pos_x, query_pos_y, weird_forward=False, cache_params=None, time_scalers=None):
        patches = self.forward_through_backbones(seq)
        img_seq = self.joiner(patches)
        img_seq = self.pos_embed(img_seq, pos_x, pos_y)


        query_token = self.query_token.expand(img_seq.shape[0], query_pos_x.shape[1], -1)
        query_token = self.pos_embed(query_token, query_pos_x, query_pos_y)
        delimiter_token = self.delimiter_token.expand(img_seq.shape[0], 1, -1)

        if weird_forward:
            # torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            if cache_params is not None:
                cache_position = torch.arange(0, 4, device=config.DEVICE)
            else:
                cache_position = None
            y = self.mamba(inputs_embeds=img_seq, use_cache=True, cache_params=cache_params, output_hidden_states=True, cache_position=cache_position)
            query_seq = torch.concat([delimiter_token, query_token], dim=1)
            g = self.mamba(inputs_embeds=query_seq, use_cache=False, cache_params=y.cache_params)
            outputs = self.forward_through_head(g.last_hidden_state[:, 1:])
            return outputs, y.cache_params
        else:
            seq = torch.concat([img_seq, delimiter_token, query_token], dim=1)
            y = self.mamba(inputs_embeds=seq, use_cache=False, cache_params=None, time_scalers=time_scalers)
            outputs = self.forward_through_head(y.last_hidden_state[:, img_seq.shape[1] + 1:])
            return outputs


    def forward_through_backbones(self, x):
        patches = [self.backbones[i](x[i]) for i in range(len(x))]
        return patches

    def forward_through_head(self, x):
        return self.heads[0](x)


    def step(self, patches, cache, pos_x, pos_y, output_pos_x, output_pos_y):
        # print('pos_x_0', self.pos_x[0].item(), 'move_x_0', move_x[0].item())
        # self.update_pos(move_x, move_y)
        patches = self.forward_through_backbones(patches)
        token = self.joiner(patches)
        token = self.pos_embed(token, pos_x, pos_y)
        cache, outputs = self.mamba_step(token, cache, output_pos_x, output_pos_y)
        return cache, outputs

    def mamba_step(self, token, cache, output_pos_x, output_pos_y):
        y, cache = self.mamba.step(token, cache)
        outputs = self.forward_through_heads(y)
        return cache, outputs

    def update_pos(self, move_x, move_y):
        self.pos_x += move_x
        self.pos_y += move_y


    def get_action(self, action_logits, max_action=False):
        probs = torch.softmax(input=action_logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        if max_action:
            actions = torch.argmax(dist.probs, dim=1)
        else:
            actions = dist.sample()
        shift_x, shift_y = self.action_converter.convert(actions)
        return actions, shift_x, shift_y, dist.log_prob(actions), dist.entropy()

    def initialize(self, img):
        cache = self.get_init_cache(img)
        self.pos_x = torch.zeros(size=(img.shape[0],), device=config.DEVICE)
        self.pos_y = torch.zeros(size=(img.shape[0],), device=config.DEVICE)
        return cache

    def get_init_cache(self, tensor):
        cache = [(None, torch.zeros(tensor.shape[0], self.mamba.config.d_model * self.mamba.config.expand_factor,
                                    self.mamba.config.d_conv - 1, device=config.DEVICE)) for _ in
                 range(self.mamba.config.n_layers)]
        return cache

#
def get_pos_emb(pos_x, pos_y, dim, temperature=10000):
    omega = torch.arange(dim // 4, dtype=torch.float32, device=config.DEVICE)
    omega = 1. / (temperature ** (omega / (dim / 4)))
    out_x = pos_x[..., None] * omega[None, :]
    sin_x, cos_x = torch.sin(out_x), torch.cos(out_x)
    out_y = pos_y[..., None] * omega[None, :]
    sin_y, cos_y = torch.sin(out_y), torch.cos(out_y)
    pos_emb = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
    return pos_emb


if __name__ == '__main__':
    pass
