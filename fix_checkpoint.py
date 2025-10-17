# fix_checkpoint.py
import torch
import json
from dataset.mineral_dataset import MineralDataset  # 确保路径正确

# =========================
# 用户配置区
# =========================
CHECKPOINT_PATH = 'checkpoints/mineral_model_best/mineral_model_best.pth'
OUTPUT_PATH = 'checkpoints/best_fixed.pth'

DB_CONFIG = {
    'host': 'localhost',
    'database': 'mineral_db',
    'user': 'root',
    'password': 'sql2008',
    'charset': 'utf8mb4'
}

DATA_ROOT = 'C:/Users/Oliver/PycharmProjects/cvTest1/data'  # 修改为你的 data_root
SPLIT = 'train'  # 通常是 'train'

CLASS_MAPPING_JSON = 'data/classes2063.json'  # 可选：用于验证
# =========================

def main():
    print("🔍 加载原始 checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    # 检查是否已有 class_to_idx
    if 'class_to_idx' in checkpoint:
        print(f"✅ 已存在 class_to_idx（{len(checkpoint['class_to_idx'])} 类），无需修复")
        return

    print("📊 构建 MineralDataset 以获取 class_to_idx...")
    dataset = MineralDataset(
        split=SPLIT,
        transform=None,  # 不需要增强
        db_config=DB_CONFIG,
        data_root=DATA_ROOT
    )

    class_to_idx = dataset.class_to_idx
    num_classes = len(class_to_idx)
    print(f"✅ 成功提取 class_to_idx，共 {num_classes} 类")

    # 可选：打印前几个类别检查是否正确
    print("🔍 示例类别映射：")
    examples = dict(list(class_to_idx.items())[:5])
    for cls, idx in examples.items():
        print(f"   {cls} -> {idx}")

    # 注入到 checkpoint
    print(f"📦 正在保存修复后的 checkpoint 到: {OUTPUT_PATH}")
    new_checkpoint = checkpoint.copy()
    new_checkpoint['class_to_idx'] = class_to_idx

    torch.save(new_checkpoint, OUTPUT_PATH)
    print("🎉 修复完成！请将 best_fixed.pth 替换为部署模型")

    # 额外保存为 JSON 方便检查
    with open('class_to_idx_debug.json', 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print("📄 已导出 class_to_idx 到 class_to_idx_debug.json")

if __name__ == "__main__":
    main()
