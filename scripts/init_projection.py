"""
初始化投影层权重并保存。

Usage:
    python init_projection.py --dim 768 --output ./projection_768.pt
"""
import argparse
import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """简单的投影层，与 ColQwen3VLEmbedding 中的 custom_text_proj 一致"""
    def __init__(self, hidden_size: int = 2048, dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(hidden_size, dim)

    def forward(self, x):
        return self.proj(x)


def main():
    parser = argparse.ArgumentParser(description="初始化投影层权重")
    parser.add_argument("--dim", type=int, default=768, help="输出维度 (default: 768)")
    parser.add_argument("--hidden-size", type=int, default=2048, help="输入维度 (default: 2048)")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径 (.pt)")
    parser.add_argument("--std", type=float, default=0.02, help="初始化标准差 (default: 0.02)")
    args = parser.parse_args()

    # 创建投影层
    projection = ProjectionLayer(hidden_size=args.hidden_size, dim=args.dim)

    # 使用与 Qwen/LLaMA 相同的默认初始化方式
    # nn.Linear 默认使用 uniform initialization: -scale ~ scale
    # scale = init_std or 1/sqrt(fan_in)
    # 这里保持默认初始化即可，与 ColQwen3VLEmbedding 保持一致

    state_dict = {
        "weight": projection.proj.weight.data.clone(),
        "bias": projection.proj.bias.data.clone() if projection.proj.bias is not None else None,
    }

    torch.save(state_dict, args.output)
    print(f"Saved projection layer to {args.output}")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  dim: {args.dim}")
    print(f"  weight shape: {state_dict['weight'].shape}")
    print(f"  bias shape: {state_dict['bias'].shape if state_dict['bias'] is not None else None}")


if __name__ == "__main__":
    main()
