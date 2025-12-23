

import math
import copy
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)
import torchvision.ops.roi_align as ROIalign

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor,hidden_dim):
    '''
    pos_tensor: [num_queries, b, 1]
    '''
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = pos_x # [num_queries, b, hidden_dim]
    return pos

def _to_roi_align_format(rois, truely_length, scale_factor=1):
    '''Convert RoIs to RoIAlign format.
    Params:
        RoIs: normalized segments coordinates, shape (batch_size, num_segments, 2)
        T: length of the video feature sequence
    '''
    # transform to absolute axis
    B, N = rois.shape[:2]
    rois_center = rois[:, :, 0:1] # [B,N,1]
    rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
    truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
    rois_abs = torch.cat(
        (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * truely_length # [B,N,2]->"start,end"
    # expand the RoIs
    _max = truely_length.repeat(1,N,2)
    _min = torch.zeros_like(_max)
    rois_abs = torch.clamp(rois_abs, min=_min, max=_max)  # (B, N, 2)
    # transfer to 4 dimension coordination
    rois_abs_4d = torch.zeros((B,N,4),dtype=rois_abs.dtype,device=rois_abs.device)
    rois_abs_4d[:,:,0], rois_abs_4d[:,:,2] = rois_abs[:,:,0], rois_abs[:,:,1] # x1,0,x2,0

    # add batch index
    batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device) # [B,1,1]
    batch_ind = batch_ind.repeat(1, N, 1) # [B,N,1]
    rois_abs_4d = torch.cat((batch_ind, rois_abs_4d), dim=2) # [B,N,1+4]->"batch_id,x1,0,x2,0"
    # NOTE: stop gradient here to stablize training
    return rois_abs_4d.view((B*N, 5)).detach()


def _roi_align(rois, origin_feat, mask, ROIalign_size, scale_factor=1):
    B,Q,_ = rois.shape
    B,T,C = origin_feat.shape
    truely_length = T-torch.sum(mask,dim=1) # [B]
    rois_abs_4d = _to_roi_align_format(rois,truely_length,scale_factor)
    feat = origin_feat.permute(0,2,1) # [B,dim,T]
    feat = feat.reshape(B,C,1,T)
    roi_feat = ROIalign(feat, rois_abs_4d, output_size=(1,ROIalign_size))
    roi_feat = roi_feat.reshape(B,Q,C,-1) # [B,Q,dim,output_width]
    roi_feat = roi_feat.permute(0,1,3,2) # [B,Q,output_width,dim]
    return roi_feat


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_enc=False,
                 return_intermediate_dec=False,
                 args = None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, return_intermediate=return_intermediate_enc,args=args)


        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          nhead=nhead,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation,
                                          normalize_before=normalize_before,
                                          args=args)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.enable_posPrior = args.enable_posPrior
        self.num_queries = args.num_queries
        self.inst_dim = d_model // 2
        self.boundary_dim = d_model // 4

        # dense head for query initialization
        self.enc_dense_cls_head = nn.Linear(d_model, 1)
        self.enc_dense_bbox_head = MLP(d_model, d_model, 2, 3)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed_tuple: Tuple[Tensor, Tensor, Tensor], query_pos,
                pos_embed, clip_feat=None, bbox_function=None, refine_decoder=None):
        '''
        input:
            src: [b,t,c]
            mask: [b,t]
            query_embed: [num_queries,c]
            pos_embed: [b,t,c]
        '''
        # permute NxTxC to TxNxC
        bs, t, c = src.shape
        src = src.permute(1, 0, 2) # [t,b,c]
        pos_embed = pos_embed.permute(1, 0, 2) # [t,b,c]

        inst_query_embed, start_query_embed, end_query_embed = query_embed_tuple
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # [layers,t,b,c]

        # ===== 使用 Encoder 最后一层输出做密集预测，用于初始化 decoder queries =====
        last_memory = memory[-1]  # [t,b,c]
        enc_feat_for_head = last_memory.permute(1, 0, 2)  # [b,t,c]

        enc_dense_logits = self.enc_dense_cls_head(enc_feat_for_head)  # [b,t,1]
        enc_dense_boxes = self.enc_dense_bbox_head(enc_feat_for_head).sigmoid()  # [b,t,2]

        scores = enc_dense_logits.squeeze(-1)  # [b,t]
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        K = min(self.num_queries, t)
        topk_scores, topk_indices = scores.topk(K, dim=1)  # [b,K]

        B, T, C = enc_feat_for_head.shape
        idx_feat = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        selected_feats = enc_feat_for_head.gather(1, idx_feat)  # [b,K,c]

        idx_box = topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        selected_boxes = enc_dense_boxes.gather(1, idx_box)  # [b,K,2]

        if K < self.num_queries:
            pad_num = self.num_queries - K
            selected_feats = torch.cat([selected_feats, selected_feats[:, -1:, :].expand(B, pad_num, C)], dim=1)
            selected_boxes = torch.cat([selected_boxes, selected_boxes[:, -1:, :].expand(B, pad_num, 2)], dim=1)
            K = self.num_queries
        inst_query_embed = inst_query_embed.unsqueeze(1).repeat(1, bs, 1)
        start_query_embed = start_query_embed.unsqueeze(1).repeat(1, bs, 1)
        end_query_embed = end_query_embed.unsqueeze(1).repeat(1, bs, 1)

        if self.enable_posPrior:
            query_pos = selected_boxes.permute(1, 0, 2)  # [num_queries,b,2]
        else:
            query_pos = query_pos.unsqueeze(1).repeat(1, bs, 1)
            if query_pos.shape[0] != selected_boxes.shape[1]:
                if query_pos.shape[0] < selected_boxes.shape[1]:
                    pad_num = selected_boxes.shape[1] - query_pos.shape[0]
                    query_pos = torch.cat([query_pos, query_pos[-1:, :, :].expand(pad_num, bs, -1)], dim=0)
                else:
                    query_pos = query_pos[: selected_boxes.shape[1]]
        mask = mask # [b,t]

        inst_feats = selected_feats[..., : self.inst_dim]
        start_feats = selected_feats[..., self.inst_dim: self.inst_dim + self.boundary_dim]
        end_feats = selected_feats[..., self.inst_dim + self.boundary_dim: self.inst_dim + 2 * self.boundary_dim]

        inst_tgt = inst_feats.permute(1, 0, 2)
        start_tgt = start_feats.permute(1, 0, 2)
        end_tgt = end_feats.permute(1, 0, 2)

        inst_hs, start_hs, end_hs, references = self.decoder(inst_tgt, start_tgt, end_tgt, memory[-1],
                                                             memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_pos,
                                                             clip_feat=clip_feat, bbox_function=bbox_function,
                                                             refine_decoder=refine_decoder)
        # permute TxNxC to NxTxC
        memory = memory.permute(0,2,1,3)
        return memory, inst_hs, start_hs, end_hs, references, enc_dense_logits, enc_dense_boxes


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False, args=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate,dim=0) # [layers,t,b,c]

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, norm=None, return_intermediate=False, d_model=256, nhead=8,
                 dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False, args=None):
        super().__init__()
        self.enable_posPrior = args.enable_posPrior
        self.inst_dim = d_model // 2
        self.boundary_dim = d_model // 4

        inst_layer = InstanceDecoderLayer(self.inst_dim, nhead, dim_feedforward, dropout, activation, normalize_before)
        boundary_layer = BoundaryDecoderLayer(self.boundary_dim, nhead, dim_feedforward, dropout, activation,
                                              normalize_before)

        self.inst_layers = _get_clones(inst_layer, num_layers)
        self.boundary_layers = _get_clones(boundary_layer, num_layers)
        self.num_layers = num_layers
        self.norm_inst = nn.LayerNorm(self.inst_dim) if norm is not None else None
        self.norm_boundary = nn.LayerNorm(self.boundary_dim) if norm is not None else None
        self.return_intermediate = return_intermediate
        self.inst_query_scale = MLP(self.inst_dim, self.inst_dim, self.inst_dim, 2)
        self.boundary_query_scale = MLP(self.boundary_dim, self.boundary_dim, self.boundary_dim, 2)

        self.inst_ref_point_head = MLP(self.inst_dim, self.inst_dim, self.inst_dim, 2)
        self.boundary_ref_point_head = MLP(self.boundary_dim, self.boundary_dim, self.boundary_dim, 2)

    def _build_instance_pos(self, reference_points: Tensor) -> Tensor:
        center = reference_points[..., :1]
        duration = reference_points[..., 1:2]
        center_embed = gen_sineembed_for_position(center.transpose(0, 1), self.inst_dim // 2)
        duration_embed = gen_sineembed_for_position(duration.transpose(0, 1), self.inst_dim - self.inst_dim // 2)
        return torch.cat([center_embed, duration_embed], dim=2)

    def _build_boundary_pos(self, boundary_points: Tensor) -> Tensor:
        return gen_sineembed_for_position(boundary_points.transpose(0, 1), self.boundary_dim)

    def forward(self, inst_tgt, start_tgt, end_tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                clip_feat=None, bbox_function=None, refine_decoder=None):
        '''
            inst_tgt/start_tgt/end_tgt: [num_queries,b,dim]
            memory: [t,b,c]
            query_pos: [num_queries,b,2] representing (center,width)
        '''

        reference_points = query_pos.sigmoid().transpose(0, 1)  # [b,num_queries,2]

        memory_start = memory[:, :, :self.boundary_dim]
        memory_end = memory[:, :, self.boundary_dim: self.boundary_dim * 2]
        memory_inst = memory[:, :, self.boundary_dim * 2:]

        pos_start = pos[:, :, :self.boundary_dim] if pos is not None else None
        pos_end = pos[:, :, self.boundary_dim: self.boundary_dim * 2] if pos is not None else None
        pos_inst = pos[:, :, self.boundary_dim * 2:] if pos is not None else None

        inst_output = inst_tgt
        start_output = start_tgt
        end_output = end_tgt

        intermediate_inst = []
        intermediate_start = []
        intermediate_end = []

        for layer_id in range(self.num_layers):
            inst_pos_embed = self._build_instance_pos(reference_points)
            start_ref = reference_points[..., :1] - reference_points[..., 1:2] / 2
            end_ref = reference_points[..., :1] + reference_points[..., 1:2] / 2

            start_pos_embed = self._build_boundary_pos(start_ref)
            end_pos_embed = self._build_boundary_pos(end_ref)

            if layer_id == 0:
                inst_pos_trans = 1
                boundary_pos_trans = 1
            else:
                inst_pos_trans = self.inst_query_scale(inst_output)
                boundary_pos_trans = self.boundary_query_scale(start_output)

            inst_pos_proj = self.inst_ref_point_head(inst_pos_embed)
            start_pos_proj = self.boundary_ref_point_head(start_pos_embed)
            end_pos_proj = self.boundary_ref_point_head(end_pos_embed)

            inst_sine = inst_pos_embed * inst_pos_trans
            start_sine = start_pos_embed * boundary_pos_trans
            end_sine = end_pos_embed * boundary_pos_trans

            inst_output = self.inst_layers[layer_id](inst_output, memory_inst, tgt_mask=tgt_mask,
                                                     memory_mask=memory_mask,
                                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                                     memory_key_padding_mask=memory_key_padding_mask,
                                                     pos=pos_inst, query_pos=inst_pos_proj, query_sine_embed=inst_sine,
                                                     is_first=(layer_id == 0))

            start_output, end_output = self.boundary_layers[layer_id](start_output, end_output, memory_start,
                                                                      memory_end,
                                                                      tgt_mask=tgt_mask,
                                                                      memory_mask=memory_mask,
                                                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                                                      memory_key_padding_mask=memory_key_padding_mask,
                                                                      pos=(pos_start, pos_end),
                                                                      query_pos=(start_pos_proj, end_pos_proj),
                                                                      query_sine_embed=(start_sine, end_sine),
                                                                      is_first=(layer_id == 0))



            if self.return_intermediate:
                intermediate_inst.append(self.norm_inst(inst_output) if self.norm_inst is not None else inst_output)
                intermediate_start.append(
                    self.norm_boundary(start_output) if self.norm_boundary is not None else start_output)
                intermediate_end.append(
                    self.norm_boundary(end_output) if self.norm_boundary is not None else end_output)

        if self.norm_inst is not None:
            inst_output = self.norm_inst(inst_output)
            start_output = self.norm_boundary(start_output)
            end_output = self.norm_boundary(end_output)

            if self.return_intermediate:
                intermediate_inst.pop()
                intermediate_start.pop()
                intermediate_end.pop()
                intermediate_inst.append(inst_output)
                intermediate_start.append(start_output)
                intermediate_end.append(end_output)

        if self.return_intermediate:
            return [torch.stack(intermediate_inst).transpose(1, 2),
                    torch.stack(intermediate_start).transpose(1, 2),
                    torch.stack(intermediate_end).transpose(1, 2), reference_points]

        return inst_output.unsqueeze(0), start_output.unsqueeze(0), end_output.unsqueeze(0), reference_points


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class InstanceDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        if self.normalize_before:
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)

class BoundaryDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # shared self-attention for start/end sequence
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Cross attention for start and end independently
        self.ca_qcontent_proj_start = nn.Linear(d_model, d_model)
        self.ca_qpos_proj_start = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj_start = nn.Linear(d_model, d_model)
        self.ca_kpos_proj_start = nn.Linear(d_model, d_model)
        self.ca_v_proj_start = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj_start = nn.Linear(d_model, d_model)
        self.cross_attn_start = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.ca_qcontent_proj_end = nn.Linear(d_model, d_model)
        self.ca_qpos_proj_end = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj_end = nn.Linear(d_model, d_model)
        self.ca_kpos_proj_end = nn.Linear(d_model, d_model)
        self.ca_v_proj_end = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj_end = nn.Linear(d_model, d_model)
        self.cross_attn_end = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, start_tgt, end_tgt, memory_start, memory_end,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tuple[Tensor, Tensor]] = None,
                query_pos: Optional[Tuple[Tensor, Tensor]] = None,
                query_sine_embed: Optional[Tuple[Tensor, Tensor]] = None,
                is_first: bool = False):

        pos_start, pos_end = pos if pos is not None else (None, None)
        query_pos_start, query_pos_end = query_pos
        query_sine_start, query_sine_end = query_sine_embed

        # concatenate for shared self-attention
        boundary = torch.cat([start_tgt, end_tgt], dim=0)
        boundary_pos = torch.cat([query_pos_start, query_pos_end], dim=0)

        q_content = self.sa_qcontent_proj(boundary)
        q_pos = self.sa_qpos_proj(boundary_pos)
        k_content = self.sa_kcontent_proj(boundary)
        k_pos = self.sa_kpos_proj(boundary_pos)
        v = self.sa_v_proj(boundary)

        q = q_content + q_pos
        k = k_content + k_pos

        boundary2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                   key_padding_mask=tgt_key_padding_mask)[0]
        boundary = boundary + self.dropout(boundary2)
        boundary = self.norm(boundary)

        start_tgt, end_tgt = boundary.chunk(2, dim=0)

        # start cross attention
        q_content_start = self.ca_qcontent_proj_start(start_tgt)
        k_content_start = self.ca_kcontent_proj_start(memory_start)
        v_start = self.ca_v_proj_start(memory_start)
        if pos_start is not None:
            k_pos_start = self.ca_kpos_proj_start(pos_start)
        else:
            k_pos_start = torch.zeros_like(k_content_start)

        if is_first:
            q_pos_start_proj = self.ca_qpos_proj_start(query_pos_start)
            q_start = q_content_start + q_pos_start_proj
            k_start = k_content_start + k_pos_start
        else:
            q_start = q_content_start
            k_start = k_content_start

        num_queries, bs, n_model = q_content_start.shape
        t, _, _ = k_content_start.shape

        q_start = q_start.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_start = self.ca_qpos_sine_proj_start(query_sine_start)
        query_sine_start = query_sine_start.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q_start = torch.cat([q_start, query_sine_start], dim=3).view(num_queries, bs, n_model * 2)

        k_start = k_start.view(t, bs, self.nhead, n_model // self.nhead)
        k_pos_start = k_pos_start.view(t, bs, self.nhead, n_model // self.nhead)
        k_start = torch.cat([k_start, k_pos_start], dim=3).view(t, bs, n_model * 2)

        start_delta = self.cross_attn_start(query=q_start,
                                            key=k_start,
                                            value=v_start,
                                            attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)[0]
        start_tgt = start_tgt + self.dropout1(start_delta)
        start_tgt = self.norm1(start_tgt)
        start_ffn = self.linear2(self.dropout(self.activation(self.linear1(start_tgt))))
        start_tgt = start_tgt + self.dropout3(start_ffn)
        start_tgt = self.norm2(start_tgt)

        # end cross attention
        q_content_end = self.ca_qcontent_proj_end(end_tgt)
        k_content_end = self.ca_kcontent_proj_end(memory_end)
        v_end = self.ca_v_proj_end(memory_end)
        if pos_end is not None:
            k_pos_end = self.ca_kpos_proj_end(pos_end)
        else:
            k_pos_end = torch.zeros_like(k_content_end)

        if is_first:
            q_pos_end_proj = self.ca_qpos_proj_end(query_pos_end)
            q_end = q_content_end + q_pos_end_proj
            k_end = k_content_end + k_pos_end
        else:
            q_end = q_content_end
            k_end = k_content_end

        q_end = q_end.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_end = self.ca_qpos_sine_proj_end(query_sine_end)
        query_sine_end = query_sine_end.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q_end = torch.cat([q_end, query_sine_end], dim=3).view(num_queries, bs, n_model * 2)

        k_end = k_end.view(t, bs, self.nhead, n_model // self.nhead)
        k_pos_end = k_pos_end.view(t, bs, self.nhead, n_model // self.nhead)
        k_end = torch.cat([k_end, k_pos_end], dim=3).view(t, bs, n_model * 2)

        end_delta = self.cross_attn_end(query=q_end,
                                        key=k_end,
                                        value=v_end,
                                        attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        end_tgt = end_tgt + self.dropout1(end_delta)
        end_tgt = self.norm1(end_tgt)
        end_ffn = self.linear2(self.dropout(self.activation(self.linear1(end_tgt))))
        end_tgt = end_tgt + self.dropout3(end_ffn)
        end_tgt = self.norm2(end_tgt)

        return start_tgt, end_tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_enc=True,
        return_intermediate_dec=True,
        args=args
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == "__main__":
    import sys
    sys.path.append('../../')
    import options
    
    args = options.parser.parse_args()
    src = torch.randn((4,16,512))
    mask = torch.zeros((4,16),dtype=bool)
    query_embed = torch.randn((10,512))
    pos = torch.randn((4,16,512))
    trans = Transformer(d_model=args.hidden_dim,
                        dropout=args.dropout,
                        nhead=args.nheads,
                        dim_feedforward=args.dim_feedforward,
                        num_encoder_layers=args.enc_layers,
                        num_decoder_layers=args.dec_layers,
                        normalize_before=args.pre_norm,
                        return_intermediate_dec=True)
    memory, hs, reference  = trans(src,mask,query_embed,pos)
    print(f"memory.shape:{memory.shape}") # [4,16,512]->[bs,T,512]
    print(f"hs.shape:{hs.shape}") # [6, 4, 10, 512]->[dec_layers,bs,num_queries,dim]
    print(f"reference.shape:{reference.shape}") # [4, 10, 1]->[bs,num_queries,1]
