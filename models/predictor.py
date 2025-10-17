# models/predictor.py
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms
import json
import os


class MineralPredictor:
    def __init__(self, model_path, class_mapping_path, full_mapping_path, device=None):
        """
        model_path: 训练保存的模型文件 (.pth)
        class_mapping_path: 训练子集 class_mapping.json（英文名称列表）
        full_mapping_path: 完整中英文映射表 full_class_mapping.json
        """
        self.device = device or "cpu"
        print(f"🎯 加载模型到设备: {self.device}")

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        os.makedirs("data", exist_ok=True)
        subset_mapping_path = os.path.join("data", f"{model_name}_classes.json")

        # 1️⃣ 加载训练子集英文列表
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            train_classes = json.load(f)  # {"0": "actinolite", ...}
        print(f"📚 已加载训练子集英文类别，共 {len(train_classes)} 个")

        # 2️⃣ 加载完整中英文映射表
        with open(full_mapping_path, 'r', encoding='utf-8') as f:
            full_mapping = json.load(f)  # {"0": {"en":"actinolite","zh":"阳起石"}, ...}

        # 3️⃣ 加载 checkpoint
        print(f"🔧 加载模型检查点: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 4️⃣ 获取权重结构
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✅ 检测到嵌套格式（model_state_dict）")
        else:
            state_dict = checkpoint
            print("✅ 检测到纯 state_dict 格式")

        # 自动推断类别数
        if 'fc.weight' in state_dict:
            num_classes_model = state_dict['fc.weight'].shape[0]
            in_features = state_dict['fc.weight'].shape[1]
            print(f"✅ 模型输出类别: {num_classes_model}，fc 输入维度: {in_features}")
        else:
            raise RuntimeError("❌ 模型权重中未找到 fc.weight")

        # 5️⃣ 构建骨干网络
        if in_features == 512:
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(512, num_classes_model)
        elif in_features == 2048:
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Linear(2048, num_classes_model)
        else:
            print(f"⚠️ 未知的 fc 输入维度 {in_features}，使用 ResNet18 兼容加载")
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(512, num_classes_model)

        self.model.to(self.device)

        # 6️⃣ 加载权重
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️ 有缺失的层未加载: {len(missing)} 个")
        if unexpected:
            print(f"⚠️ 有多余的层未匹配: {len(unexpected)} 个")
        print("✅ 模型权重加载完成（已忽略不匹配层）")

        # 在加载权重后添加
        print("🔍 检查模型权重...")
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                print(f"🔍 {name}: {param.data.abs().mean().item():.6f}")
                break

        # 检查模型是否在训练模式
        print(f"🔍 模型模式: {'训练' if self.model.training else '评估'}")
        # 7️⃣ 修正名称映射逻辑
        # 构建完整映射的英文到中文字典（使用原始英文名称）
        en_to_zh = {}
        for item in full_mapping.values():
            en_name = item['en'].strip()
            zh_name = item['zh'].strip()
            en_to_zh[en_name] = zh_name

        print(f"📖 完整映射表包含 {len(en_to_zh)} 个类别")

        # 构建训练索引 -> 英文 + 中文
        self.idx_to_class = {}
        sorted_keys = sorted(train_classes.keys(), key=int)

        found_count = 0
        not_found_classes = []

        for idx, key in enumerate(sorted_keys):
            en_name = train_classes[key].strip()
            zh_name = en_to_zh.get(en_name)

            if zh_name:
                self.idx_to_class[idx] = {"en": en_name, "zh": zh_name}
                found_count += 1
            else:
                # 如果直接匹配失败，尝试模糊匹配（忽略大小写和空格）
                en_name_lower = en_name.lower().replace(' ', '')
                found = False

                for full_en, full_zh in en_to_zh.items():
                    full_en_lower = full_en.lower().replace(' ', '')
                    if en_name_lower == full_en_lower:
                        self.idx_to_class[idx] = {"en": en_name, "zh": full_zh}
                        found_count += 1
                        found = True
                        print(f"🔄 模糊匹配成功: '{en_name}' -> '{full_en}'")
                        break

                if not found:
                    not_found_classes.append(en_name)
                    self.idx_to_class[idx] = {"en": en_name, "zh": "未知"}
                    print(f"⚠️ 未找到映射: {en_name}")

        print(f"✅ 成功映射 {found_count}/{len(sorted_keys)} 个类别")
        if not_found_classes:
            print(f"❌ 未找到映射的类别: {not_found_classes}")

        # 8️⃣ 保存子集映射 JSON
        with open(subset_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.idx_to_class, f, ensure_ascii=False, indent=2)
        print(f"💾 已生成模型类别文件: {subset_mapping_path}")

        # 9️⃣ 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 打印前几个类别作为验证
        print("🔍 映射验证（前5个类别）:")
        for i in range(min(5, len(self.idx_to_class))):
            info = self.idx_to_class[i]
            print(f"  {i}: {info['en']} -> {info['zh']}")

        # 🔥 在这里添加测试调用 - 在初始化完成后测试模型
        self.test_random_prediction()
    def predict(self, image_path, top_k=3):
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"📷 成功加载图像: {image_path}")
            print(f"📐 图像尺寸: {image.size}")
        except Exception as e:
            return {'error': f"无法加载图像: {str(e)}"}

        image = self.transform(image).unsqueeze(0).to(self.device)
        print(f"🔧 图像预处理完成，输入张量形状: {image.shape}")

        with torch.no_grad():
            # 设置为评估模式
            self.model.eval()

            output = self.model(image)
            print(f"📊 模型原始输出形状: {output.shape}")
            print(f"📊 模型原始输出值: {output}")

            probs = torch.softmax(output, dim=1)
            print(f"📈 Softmax后概率形状: {probs.shape}")
            print(f"📈 概率值: {probs}")

            top_probs, top_indices = torch.topk(probs, top_k)
            print(f"🏆 Top {top_k} 索引: {top_indices}")
            print(f"🏆 Top {top_k} 概率: {top_probs}")

        results = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            label_info = self.idx_to_class.get(idx, {"en": "Unknown", "zh": "未知"})
            results.append({
                'index': idx,
                'label_en': label_info['en'],
                'label_zh': label_info['zh'],
                'confidence': round(prob * 100, 2)
            })
            print(f"🔍 结果 {i + 1}: 索引={idx}, 类别={label_info}, 置信度={prob:.4f}")

        return results

    def test_random_prediction(self):
        """生成随机输入测试模型输出"""
        print("🧪 运行随机输入测试...")
        random_input = torch.randn(1, 3, 224, 224).to(self.device)

        with torch.no_grad():
            self.model.eval()
            output = self.model(random_input)
            probs = torch.softmax(output, dim=1)
            max_prob, max_idx = torch.max(probs, 1)

            print(f"🧪 随机输入测试 - 最大概率索引: {max_idx.item()}")
            print(f"🧪 随机输入测试 - 最大概率值: {max_prob.item():.4f}")
            print(f"🧪 随机输入测试 - 所有概率分布: {probs[0][:10]}...")  # 只显示前10个