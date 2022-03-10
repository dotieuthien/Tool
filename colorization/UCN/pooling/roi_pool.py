from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from pooling import cpp_module
from apex import amp


class _ROIPool(Function):
    @staticmethod
    def forward(ctx, inputs, roi, mask, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = inputs.size()
        output, argmax = cpp_module.roi_pool_forward(
            inputs, roi, mask, spatial_scale, output_size[0], output_size[1]
        )
        ctx.save_for_backward(inputs, roi, mask, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, rois, masks, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = cpp_module.roi_pool_backward(
            grad_output, inputs, rois, masks,
            argmax, spatial_scale,
            output_size[0], output_size[1],
            bs, ch, h, w,
        )
        return grad_input, None, None, None, None


roi_pool = _ROIPool.apply


class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    @amp.float_function
    def forward(self, inputs, rois, masks):
        return roi_pool(inputs, rois, masks, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
