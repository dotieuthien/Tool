from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from pooling import cpp_module
from apex import amp


class _ROIArea(Function):
    @staticmethod
    def forward(ctx, inputs, roi, mask, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = inputs.size()
        output = cpp_module.roi_area_forward(
            inputs, roi, mask, spatial_scale, output_size[0], output_size[1]
        )
        ctx.save_for_backward(inputs, roi, mask)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, rois, masks = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = cpp_module.roi_area_backward(
            grad_output, inputs, rois, masks,
            spatial_scale, output_size[0], output_size[1],
            bs, ch, h, w,
        )
        return grad_input, None, None, None, None


roi_area = _ROIArea.apply


class ROIArea(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIArea, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    @amp.float_function
    def forward(self, inputs, rois, masks):
        return roi_area(inputs, rois, masks, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
