// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
        const int height, const int width,
        T y, T x,
        const int index) {

    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        //empty
        return 0;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T) y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T) x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1.0 - ly, hx = 1.0 - lx;
    // do bilinear interpolation
    T v1 = bottom_data[y_low * width + x_low];
    T v2 = bottom_data[y_low * width + x_high];
    T v3 = bottom_data[y_high * width + x_low];
    T v4 = bottom_data[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <typename T>
__global__ void RoIAlignForward(const int nthreads, const T* bottom_data,
        const T spatial_scale, const int channels,
        const int height, const int width,
        const int pooled_height, const int pooled_width,
        const int sampling_ratio,
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

        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = offset_bottom_rois[0];
        const T* offset_bottom_mask = bottom_masks + n * mask_size;

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_bottom_rois[1] * spatial_scale;
        T roi_start_h = offset_bottom_rois[2] * spatial_scale;
        T roi_end_w = offset_bottom_rois[3] * spatial_scale;
        T roi_end_h = offset_bottom_rois[4] * spatial_scale;
        // printf("%f %f %f %f\n", roi_start_h, roi_end_h, roi_start_w, roi_end_w);

        // Force malformed ROIs to be 1x1
        T roi_width = max(roi_end_w - roi_start_w, (T)1.0);
        T roi_height = max(roi_end_h - roi_start_h, (T)1.0);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = max(static_cast<int>(ceil(roi_height / pooled_height)), sampling_ratio);
        int roi_bin_grid_w = max(static_cast<int>(ceil(roi_width / pooled_width)), sampling_ratio);
        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w;

        // Get mask value
        int mask_index = ph * pooled_width + pw;
        const T mask_value = offset_bottom_mask[mask_index];

        if (c >= channels) {
            T mean_loc = -1.0;
            // Compute mean-x
            if (c == total_channels - 4) {
                mean_loc = roi_start_h + (static_cast<T>(ph) + 0.5) * bin_size_h;
                mean_loc = 2.0 * mean_loc / static_cast<T>(height) - 1.0;
            }
            // Compute mean-y
            else if (c == total_channels - 3) {
                mean_loc = roi_start_w + (static_cast<T>(pw) + 0.5) * bin_size_w;
                mean_loc = 2.0 * mean_loc / static_cast<T>(width) - 1.0;
            }
            else if (c == total_channels - 2) {
                mean_loc = roi_height / static_cast<T>(height);
            }
            else if (c == total_channels - 1) {
                mean_loc = roi_width / static_cast<T>(width);
            }
            top_data[index] = mean_loc;
        }
        else {
            T output_val = 0.0;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + 0.5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
                // e.g., 0.5, 1.5

                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + 0.5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    T val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
                    output_val += val;
                }
            }
	    output_val = (output_val / count) * mask_value;
            top_data[index] = output_val;
        }
    }
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
        const int height, const int width,
        T y, T x,
        T & w1, T & w2, T & w3, T & w4,
        int & x_low, int & x_high, int & y_low, int & y_high,
        const int index /* index for debug only*/) {

    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        //empty
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    y_low = (int) y;
    x_low = (int) x;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T) y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T) x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1.0 - ly, hx = 1.0 - lx;

    // reference in forward
    // T v1 = bottom_data[y_low * width + x_low];
    // T v2 = bottom_data[y_low * width + x_high];
    // T v3 = bottom_data[y_high * width + x_low];
    // T v4 = bottom_data[y_high * width + x_high];
    // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    return;
}

template <typename T>
__global__ void RoIAlignBackwardFeature(const int nthreads, const T* top_diff,
        const int num_rois, const T spatial_scale,
        const int channels, const int height, const int width,
        const int pooled_height, const int pooled_width,
        const int sampling_ratio,
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

        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = offset_bottom_rois[0];
        const T* offset_bottom_mask = bottom_masks + n * mask_size;

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_bottom_rois[1] * spatial_scale;
        T roi_start_h = offset_bottom_rois[2] * spatial_scale;
        T roi_end_w = offset_bottom_rois[3] * spatial_scale;
        T roi_end_h = offset_bottom_rois[4] * spatial_scale;

        // Force malformed ROIs to be 1x1
        T roi_width = max(roi_end_w - roi_start_w, (T)1.0);
        T roi_height = max(roi_end_h - roi_start_h, (T)1.0);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
        T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

        int top_offset = (n * total_channels + c) * pooled_height * pooled_width;
        const T* offset_top_diff = top_diff + top_offset;
        const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = max(static_cast<int>(ceil(roi_height / pooled_height)), sampling_ratio);
        int roi_bin_grid_w = max(static_cast<int>(ceil(roi_width / pooled_width)), sampling_ratio);
        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w;

        // Get mask value
        int mask_index = ph * pooled_width + pw;
        const T mask_value = offset_bottom_mask[mask_index];

        if (mask_value == 0.0 || c >= channels) {
            continue;
        }

        for (int iy = 0; iy < roi_bin_grid_h; iy ++) {
            const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + 0.5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
            // e.g., 0.5, 1.5

            for (int ix = 0; ix < roi_bin_grid_w; ix ++) {
                const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + 0.5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                T w1, w2, w3, w4;
                int x_low, x_high, y_low, y_high;

                bilinear_interpolate_gradient(height, width, y, x,
                    w1, w2, w3, w4,
                    x_low, x_high, y_low, y_high,
                    index);

                T g1 = top_diff_this_bin * w1 / count;
                T g2 = top_diff_this_bin * w2 / count;
                T g3 = top_diff_this_bin * w3 / count;
                T g4 = top_diff_this_bin * w4 / count;

                if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                    atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
                    atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
                    atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
                    atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
                } // if
            } // ix
        } // iy
    } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward


at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
        const at::Tensor& rois,
        const at::Tensor& masks,
        const float spatial_scale,
        const int pooled_height,
        const int pooled_width,
        const int sampling_ratio) {
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

    AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlign_forward", [&] {
        RoIAlignForward<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input.contiguous().data<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            rois.contiguous().data<scalar_t>(),
            masks.contiguous().data<scalar_t>(),
            output.data<scalar_t>());
    });
    THCudaCheck(cudaGetLastError());
    return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
        const at::Tensor& rois,
        const at::Tensor& masks,
        const float spatial_scale,
        const int pooled_height,
        const int pooled_width,
        const int batch_size,
        const int channels,
        const int height,
        const int width,
        const int sampling_ratio) {
    AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
    AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

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

    AT_DISPATCH_FLOATING_TYPES(grad.type(), "ROIAlign_backward", [&] {
        RoIAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.contiguous().data<scalar_t>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            grad_input.data<scalar_t>(),
            rois.contiguous().data<scalar_t>(),
            masks.contiguous().data<scalar_t>());
    });
    THCudaCheck(cudaGetLastError());
    return grad_input;
}
