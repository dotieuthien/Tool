// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ROIAlign.h"
#include "ROIPool.h"
#include "ROIArea.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
    m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
    m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
    m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
    m.def("roi_area_forward", &ROIArea_forward, "ROIArea_forward");
    m.def("roi_area_backward", &ROIArea_backward, "ROIArea_backward");
    // dcn-v2
}