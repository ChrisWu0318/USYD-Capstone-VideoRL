"""
temporal_mask.py
================
帧级别的视觉 mask 工具, 用于 D1 反事实因果奖励。

Qwen2.5-VL 的 pixel_values_videos 布局:
  - video_grid_thw 每行 (t, h, w) 描述一个视频片段的 3D grid
  - pixel_values_videos 按 [时间组0的所有空间patch, 时间组1的..., ...] 铺平
  - temporal_patch_size=2, 所以 t 个时间组对应 2t 个原始帧

在 trainer 中, pixel_values_videos 已经 .repeat(num_gen, 1),
video_grid_thw 也已经 .repeat(num_gen, 1),
所以有 num_gen 份完全相同的拷贝首尾相接。
"""

import random
import torch
from typing import Optional


def build_temporal_frame_mask(
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
    mask_ratio: float,
    num_generations: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    对视频的时间组做随机 mask (置零), 保留空间结构完整。

    所有 num_generations 份拷贝使用同一组 mask indices,
    保证同一 prompt 下的因果对比是一致的。

    Args:
        pixel_values_videos: (total_patches, patch_dim)
            = num_gen × (t × h × w), patch_dim
        video_grid_thw: (num_gen, 3)
            每行 (t, h, w), 所有行相同 (因为 repeat)
        mask_ratio: 要 mask 掉的时间组比例, 如 0.5
        num_generations: G, generation 数量
        seed: 可选, 用于可复现性

    Returns:
        masked_pixel_values: 同 shape, 被选中时间组的 patch 置零
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # 从第一个 segment 读取 grid 结构 (所有 segment 相同)
    t, h, w = video_grid_thw[0].int().tolist()
    patches_per_frame = h * w            # 每个时间组的 patch 数
    patches_per_video = t * h * w        # 一份视频的总 patch 数

    # 决定要 mask 哪些时间组 (至少 mask 1 个, 最多 mask t-1 个保留至少一帧)
    num_mask = max(1, min(t - 1, int(t * mask_ratio)))
    mask_frame_indices = rng.sample(range(t), num_mask)

    # 构建单份视频的 patch-level boolean mask
    # True = 保留, False = 置零
    keep_mask_single = torch.ones(patches_per_video, dtype=torch.bool,
                                  device=pixel_values_videos.device)
    for fi in mask_frame_indices:
        start = fi * patches_per_frame
        end = start + patches_per_frame
        keep_mask_single[start:end] = False

    # 扩展到所有 num_generations 份拷贝
    keep_mask = keep_mask_single.repeat(num_generations)  # (total_patches,)

    # 应用 mask
    masked = pixel_values_videos.clone()
    masked[~keep_mask] = 0.0

    return masked


# ====================================================================
# 用于 grpo_trainer.py 中替换全量 mask 的代码片段
# ====================================================================
#
# 原始代码 (grpo_trainer.py L562-586):
#
#     if self.exp_config.enable_causal_reward and video_inputs:
#         with torch.no_grad():
#             masked_inputs = {}
#             for k, v in prompt_inputs.items():
#                 masked_inputs[k] = v.clone() if isinstance(v, torch.Tensor) else v
#             if "pixel_values_videos" in masked_inputs:
#                 masked_inputs["pixel_values_videos"] = torch.zeros_like(
#                     masked_inputs["pixel_values_videos"]
#                 )
#             ...
#
# 替换为:
#
#     from temporal_mask import build_temporal_frame_mask
#
#     if self.exp_config.enable_causal_reward and video_inputs:
#         with torch.no_grad():
#             masked_inputs = {}
#             for k, v in prompt_inputs.items():
#                 masked_inputs[k] = v.clone() if isinstance(v, torch.Tensor) else v
#
#             if "pixel_values_videos" in masked_inputs:
#                 masked_inputs["pixel_values_videos"] = build_temporal_frame_mask(
#                     pixel_values_videos=masked_inputs["pixel_values_videos"],
#                     video_grid_thw=masked_inputs["video_grid_thw"],
#                     mask_ratio=self.exp_config.causal_mask_ratio,
#                     num_generations=self.num_generations,
#                 )
#             elif "pixel_values" in masked_inputs:
#                 # Image fallback: 图片没有时间维, 全量 mask
#                 masked_inputs["pixel_values"] = torch.zeros_like(
#                     masked_inputs["pixel_values"]
#                 )
#
#             try:
#                 masked_per_token_logps = self._get_per_token_logps(
#                     model, prompt_completion_ids, **masked_inputs
#                 )
#                 masked_per_token_logps = masked_per_token_logps[:, prompt_length - 1 :]
#             except Exception as e:
#                 print(f"[D1] Error computing masked logps: {e}. Skipping.")
#                 masked_per_token_logps = None