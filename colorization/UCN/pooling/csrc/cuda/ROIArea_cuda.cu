// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)


template <typename T>
__global__ void RoIAreaFForward(
        const int nthreads,
        const T* bottom_data,
        const T spatial_scale,
        const int channels,
        const int height,
        const int width,
        const int pooled_height,
        const int pooled_width,
        const T* bottom_rois,
        const T* bottom_masks,
        T* top_data) {
    int roi_cols = 5;
    int mask_size = pooled_width * pooled_height;
    int total_channels = channels + 4;

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % total_channels;
        int n = index / pooled_width / pooled_height / total_channels;

        // Check segmentation mask value
        const T* offset_bottom_mask = bottom_masks + n * mask_size;
        int mask_index = ph * pooled_width + pw;
        const T mask_value = offset_bottom_mask[mask_index];

        if (mask_value == 0.0 && c < channels) {
            top_data[index] = 0.0;
            continue;
        }

        // Get current feature map
        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = offset_bottom_rois[0];
        int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
        int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
        int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
        int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);

        if (c >= channels) {
            T mean_loc = -1.0;
            // Compute mean-x
            if (c == total_channels - 4) {
                mean_loc = (static_cast<T>(hstart) + static_cast<T>(hend));
                mean_loc = mean_loc / static_cast<T>(height) - 1.0;
            }
            // Compute mean-y
            else if (c == total_channels - 3) {
                mean_loc = (static_cast<T>(wstart) + static_cast<T>(wend));
                mean_loc = mean_loc / static_cast<T>(width) - 1.0;
            }
            else if (c == total_channels - 2) {
                mean_loc = static_cast<T>(roi_height) / static_cast<T>(height);
            }
            else if (c == total_channels - 1) {
                mean_loc = static_cast<T>(roi_width) / static_cast<T>(width);
            }
            top_data[index] = mean_loc;
        }
        // Compute max pooling value
        else {
            // Define an empty pooling region to be zero
            T mean_value = 0.0;
            int num_sampled = max((hend - hstart) * (wend - wstart), 1);

            const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = h * width + w;
                    T value = offset_bottom_data[bottom_index];
                    mean_value = mean_value + value;
                }
            }
            top_data[index] = mean_value / static_cast<T>(num_sampled);
        }
    }
}

template <typename T>
__global__ void RoIAreaFBackward(
        const int nthreads,
        const T* top_diff,
        const int num_rois,
        const T spatial_scale,
        const int channels,
        const int height,
        const int width,
        const int pooled_height,
        const int pooled_width,
        T* bottom_diff,
        const T* bottom_rois,
        const T* bottom_masks) {
    int roi_cols = 5;
    int mask_size = pooled_width * pooled_height;
    int total_channels = channels + 4;

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % total_channels;
        int n = index / pooled_width / pooled_height / total_channels;

        // Check segmentation mask value
        const T* offset_bottom_mask = bottom_masks + n * mask_size;
        int mask_index = ph * pooled_width + pw;
        const T mask_value = offset_bottom_mask[mask_index];

        if (mask_value == 0.0 || c >= channels) {
            continue;
        }

        // Backward for pooling value
        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = offset_bottom_rois[0];
        int bottom_offset = (roi_batch_ind * channels + c) * height * width;
        T* offset_bottom_diff = bottom_diff + bottom_offset;

        int top_offset = (n * total_channels + c) * pooled_height * pooled_width;
        const T* offset_top_diff = top_diff + top_offset;
        T top_diff_value = static_cast<T>(offset_top_diff[ph * pooled_width + pw]);

        // Get box info
        int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
        int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
        int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
        int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);
        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
        int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        int num_sampled = max((hend - hstart) * (wend - wstart), 1);

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                atomicAdd(
                    offset_bottom_diff + bottom_index,
                    static_cast<T>(top_diff_value) / static_cast<T>(num_sampled));
            }
        }
    }
}

at::Tensor ROIArea_forward_cuda(
        const at::Tensor& input,
        const at::Tensor& rois,
        const at::Tensor& masks,
        const float spatial_scale,
        const int pooled_height,
        const int pooled_width) {
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

    AT_ASSERTM(masks.size(1) == pooled_height, "mask height must be equal to pooled_height");
    AT_ASSERTM(masks.size(2) == pooled_width, "mask width must be equal to pooled_width");

    auto num_rois = rois.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = at::empty({num_rois, channels + 4, pooled_height, pooled_width}, input.options());
    auto output_size = num_rois * pooled_height * pooled_width * (channels + 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
    dim3 block(512);

    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return output;
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIArea_forward", [&] {
        RoIAreaFForward<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input.contiguous().data<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois.contiguous().data<scalar_t>(),
            masks.contiguous().data<scalar_t>(),
            output.data<scalar_t>());
    });
    THCudaCheck(cudaGetLastError());
    return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor ROIArea_backward_cuda(
        const at::Tensor& grad,
        const at::Tensor& input,
        const at::Tensor& rois,
        const at::Tensor& masks,
        const float spatial_scale,
        const int pooled_height,
        const int pooled_width,
        const int batch_size,
        const int channels,
        const int height,
        const int width) {
    AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
    AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
    // TODO add more checks
    AT_ASSERTM(masks.size(1) == pooled_height, "mask height must be equal to pooled_height");
    AT_ASSERTM(masks.size(2) == pooled_width, "mask width must be equal to pooled_width");

    auto num_rois = rois.size(0);
    auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
    dim3 block(512);

    // handle possibly empty gradients
    if (grad.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return grad_input;
    }

    AT_DISPATCH_FLOATING_TYPES(grad.type(), "ROIArea_backward", [&] {
        RoIAreaFBackward<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.contiguous().data<scalar_t>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data<scalar_t>(),
            rois.contiguous().data<scalar_t>(),
            masks.contiguous().data<scalar_t>());
    });
    THCudaCheck(cudaGetLastError());
    return grad_input;
}
