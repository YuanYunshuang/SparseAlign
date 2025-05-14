import torch
from torch.autograd import Function

import cuda_ops


class DotProduct(Function):
  @staticmethod
  def forward(ctx, query, pos_enc, out_F, kq_map):
    assert (query.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_map.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.kkk = pos_enc.shape[0]
    ctx.save_for_backward(query, pos_enc, kq_map)
    cuda_ops.dot_product_forward(ctx.m, ctx.h, ctx.kkk, ctx.c, query, pos_enc,
                                        out_F, kq_map)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, pos_enc, kq_map = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_ops.dot_product_backward(ctx.m, ctx.h, ctx.kkk, ctx.c, query, pos_enc,
                                         kq_map, grad_query, grad_pos, grad_out_F)
    return grad_query, grad_pos, None, None

dot_product_cuda = DotProduct.apply


class ScalarAttention(Function):
  @staticmethod
  def forward(ctx, weight, value, out_F, kq_indices):
    assert (weight.is_contiguous() and value.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_indices.shape[1]
    _, ctx.h, ctx.c = value.shape
    ctx.save_for_backward(weight, value, kq_indices)
    cuda_ops.scalar_attention_forward(ctx.m, ctx.h, ctx.c, weight, value, out_F,
                                             kq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    weight, value, kq_indices = ctx.saved_tensors
    grad_weight = torch.zeros_like(weight)
    grad_value = torch.zeros_like(value)
    cuda_ops.scalar_attention_backward(ctx.m, ctx.h, ctx.c, weight, value,
                                              kq_indices, grad_weight, grad_value,
                                              grad_out_F)
    return grad_weight, grad_value, None, None


scalar_attention_cuda = ScalarAttention.apply


class IndexPooling(Function):
  @staticmethod
  def forward(ctx, x, c_indices, out, out_indices):
    assert (x.is_contiguous() and c_indices.is_contiguous() and out.is_contiguous()), 'inputs should be contiguous.'
    assert len(x.shape)==1 and len(c_indices.shape)==1 and len(c_indices.shape)==1, 'input tensors dim error.'
    assert len(out.shape) == 2, 'out tensor dim error.'
    assert x.shape[0] == out_indices.shape[0] and x.shape[0] == out_indices.shape[0], 'shape doesn\'t match.'
    ctx.m = x.shape[0]
    assert c_indices.max() < ctx.m, 'c_indices max value larger than out dim.'
    assert c_indices.min() >= 0, 'indices should >= 0'
    assert out_indices.min() >= 0, 'indices should >= 0'
    _, ctx.c = out.shape
    ctx.save_for_backward(x, c_indices, out_indices)
    cuda_ops.index_pooling_forward(ctx.m, ctx.c, x, c_indices, out, out_indices)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    # print(torch.isnan(grad_out).sum())
    # print(grad_out.type())
    # print(grad_out.shape)
    x, c_indices, out_indices = ctx.saved_tensors
    assert c_indices.min() >= 0, 'indices should >= 0'
    assert out_indices.min() >= 0, 'indices should >= 0'
    grad_x = torch.zeros_like(x)
    cuda_ops.index_pooling_backward(ctx.m, ctx.c, c_indices, out_indices,
                                              grad_x, grad_out)
    return grad_x, None, None, None

index_pooling_cuda = IndexPooling.apply
