import torch
import torch.nn.functional as F


def _bottom_data_slice(bottom_data, y, x, height, width):
    batch = y.size(0)
    num_filters = bottom_data.size(1)
    bottom_data = bottom_data.permute([0, 2, 3, 1])
    bottom_data = bottom_data.view([-1, num_filters])
    output_height = y.size(2)
    output_width = x.size(2)
    y = y.unsqueeze(3)
    x = x.unsqueeze(2)
    batch_off = torch.arange(batch) * height * width
    batch_off = batch_off.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    linear_indices = (y * width + x + batch_off).view([-1])
    gathered = torch.index_select(bottom_data, 0, linear_indices)
    roi_count = y.size(1)
    gathered = gathered.view([batch, roi_count, output_height, output_width, num_filters])
    gathered = gathered.permute([0, 1, 4, 2, 3])
    return gathered


def _bilinear_interpolate(bottom_data, height, width, y, x):
    y_low = torch.where(y.long() >= height - 1, torch.full_like(y, height - 1),
                        y.long().float()).long()
    y_low = y_low.clamp(0, height - 1)
    x_low = torch.where(x.long() >= width - 1, torch.full_like(x, width - 1),
                        x.long().float()).long()
    x_low = x_low.clamp(0, width - 1)

    y_high = torch.where(y.long() >= height - 1,
                         torch.full_like(y_low, height - 1), y_low + 1)
    y_high = y_high.clamp(0, height - 1)
    x_high = torch.where(x.long() >= width - 1, torch.full_like(x_low, width - 1),
                         x_low + 1)
    x_high = x_high.clamp(0, width - 1)

    y = torch.where(y.long() >= height - 1, y_low.float(), y)
    x = torch.where(x.long() >= width - 1, x_low.float(), x)

    ly = (y - y_low.float()).unsqueeze(3)
    lx = (x - x_low.float()).unsqueeze(2)
    hy = 1. - ly
    hx = 1. - lx

    v1 = _bottom_data_slice(bottom_data, y_low, x_low, height, width)
    v2 = _bottom_data_slice(bottom_data, y_low, x_high, height, width)
    v3 = _bottom_data_slice(bottom_data, y_high, x_low, height, width)
    v4 = _bottom_data_slice(bottom_data, y_high, x_high, height, width)

    w1 = (hy * hx).unsqueeze(2)
    w2 = (hy * lx).unsqueeze(2)
    w3 = (ly * hx).unsqueeze(2)
    w4 = (ly * lx).unsqueeze(2)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    y = y.unsqueeze(2).unsqueeze(4).expand_as(val)
    x = x.unsqueeze(2).unsqueeze(3).expand_as(val)
    val = torch.where(y < -1, torch.zeros_like(val), val)
    val = torch.where(y > height, torch.zeros_like(val), val)
    val = torch.where(x < -1, torch.zeros_like(val), val)
    val = torch.where(x > height, torch.zeros_like(val), val)

    return val


def tensor_roi_align(bottom_data, bottom_rois, pooled_size, spatial_scale,
                     sampling_ratio):
    # Possible bug scenario arguments: 
    # torch.Size([1, 256, 200, 304]) torch.Size([1, 0, 4]) (14, 14) 0.25 2
    print(bottom_data.shape, bottom_rois.shape, pooled_size, spatial_scale, sampling_ratio) 
    pooled_height = pooled_size[0]
    pooled_width = pooled_size[1]

    roi_sizes = bottom_rois * spatial_scale
    roi_start_w = roi_sizes[:, :, 0]
    roi_start_h = roi_sizes[:, :, 1]
    roi_end_w = roi_sizes[:, :, 2]
    roi_end_h = roi_sizes[:, :, 3]

    roi_width = torch.max(roi_end_w - roi_start_w, torch.ones_like(roi_end_w))
    roi_height = torch.max(roi_end_h - roi_start_h, torch.ones_like(roi_end_h))
    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    pw = torch.tensor(
        [pw for pw in range(pooled_width) for _ in range(sampling_ratio)])
    ph = torch.tensor(
        [ph for ph in range(pooled_height) for _ in range(sampling_ratio)])
    x_neigh_offsets = torch.tensor([pw + .5 for pw in range(sampling_ratio)] *
                                   pooled_width)
    y_neigh_offsets = torch.tensor([ph + .5 for ph in range(sampling_ratio)] *
                                   pooled_height)

    ph = ph.unsqueeze(0).unsqueeze(0)
    pw = pw.unsqueeze(0).unsqueeze(0)
    bin_size_h = bin_size_h.unsqueeze(2)
    bin_size_w = bin_size_w.unsqueeze(2)
    batch = bottom_data.size(0)
    y_neigh_offsets = y_neigh_offsets.unsqueeze(0).unsqueeze(0)
    x_neigh_offsets = x_neigh_offsets.unsqueeze(0).unsqueeze(0)

    y_neigh_offsets = y_neigh_offsets.expand(batch, 1, y_neigh_offsets.size(2))
    x_neigh_offsets = x_neigh_offsets.expand(batch, 1, x_neigh_offsets.size(2))
    ph = ph.expand(batch, 1, ph.size(2))
    pw = pw.expand(batch, 1, pw.size(2))
    roi_start_h = roi_start_h.unsqueeze(2)
    roi_start_w = roi_start_w.unsqueeze(2)

#    if roi_start_w.shape == torch.Size([1, 4, 1]):
#      import pdb
#      pdb.set_trace()

    y = (roi_start_h + bin_size_h * ph.float()
         + bin_size_h * y_neigh_offsets / sampling_ratio)
    print("=debug=> roi_start_h.shape: {}, bin_size_h.shape: {}, ph.float().shape: {}, bin_size_h.shape: {}, y_neigh_offsets.shape: {}"
          .format(roi_start_h.shape, bin_size_h.shape, ph.float().shape, bin_size_h.shape, y_neigh_offsets.shape))
    x = (roi_start_w + bin_size_w * pw.float()
         + bin_size_w * x_neigh_offsets / sampling_ratio)
    print("=debug=> roi_start_w.shape: {}, bin_size_w.shape: {}, pw.float().shape: {}, bin_size_w.shape: {}, x_neigh_offsets.shape: {}"
          .format(roi_start_w.shape, bin_size_w.shape, pw.float().shape, bin_size_w.shape, x_neigh_offsets.shape))

    height = bottom_data.size(2)
    width = bottom_data.size(3)
    interpolated = _bilinear_interpolate(bottom_data, height, width, y, x)
    interpolated = interpolated.view(-1, interpolated.size(2), interpolated.size(3), interpolated.size(4))
    return F.avg_pool2d(interpolated, sampling_ratio)

