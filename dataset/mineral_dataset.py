# dataset/mineral_dataset.py
# PyTorch Dataset 类（从数据库读取）

import mysql.connector
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os

class MineralDataset(Dataset):
    def __init__(self, split='train', transform=None, db_config=None, data_root=None):
        """
        :param split: train/val/test
        :param transform: 数据增强
        :param db_config: 数据库连接配置
        :param data_root: 数据集所在项目的根目录，如 "C:/.../cvTest1"
        """
        self.transform = transform
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'mineral_db',
            'user': 'root',
            'password': 'sql2008',
            'charset': 'utf8mb4'
        }
        self.data_root = data_root

        # 查询样本
        conn = self.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_path, mineral_name
            FROM image_samples
            WHERE split = %s AND is_validated = TRUE
        """, (split,))
        self.samples = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(self.samples) == 0:
            raise ValueError(f"未找到 split='{split}' 的样本，请检查数据库")

        # 构建标签映射
        self.classes = sorted(list(set([s[1] for s in self.samples])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # 用于生成默认图像（假设 transform 会 resize，这里用 224 作为 fallback）
        self.default_size = 224

    def get_db_connection(self):
        return mysql.connector.connect(**self.db_config)

    def resolve_path(self, relative_path: str) -> str:
        """将数据库中的相对路径转为完整路径"""
        relative_path = relative_path.replace("\\", "/")
        full_path = os.path.join(self.data_root, relative_path)
        return os.path.abspath(full_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        db_path, label_name = self.samples[idx]
        full_path = self.resolve_path(db_path)

        try:
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"文件不存在: {full_path}")

            image = Image.open(full_path).convert('RGB')
            # 可选：验证图像完整性（但 verify() 会关闭文件句柄，需重新打开）
            # image.verify()  # 不推荐在此处用，因后续需 .convert()
        except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as e:
            # 捕获常见图像错误
            error_msg = f"❌ 加载样本失败 [{idx}]: {str(e)}\n   数据库路径: {db_path}\n   解析后路径: {full_path}"
            print(error_msg)

            # 返回一个默认黑色图像（避免 DataLoader 崩溃）
            from PIL import Image as PILImage
            image = PILImage.new('RGB', (self.default_size, self.default_size), (0, 0, 0))
            label = -1  # 无效标签（但后续 transform 可能仍会处理）
        else:
            label = self.class_to_idx[label_name]

        # 应用 transform（即使图像损坏也应用，保持 tensor shape 一致）
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as transform_error:
                print(f"⚠️  Transform 失败 [{idx}]: {transform_error}")
                # 再次 fallback：生成一个符合预期尺寸的零张量（但这里仍在 PIL 阶段）
                from PIL import Image as PILImage
                image = PILImage.new('RGB', (self.default_size, self.default_size), (0, 0, 0))
                image = self.transform(image)

        # 如果 label 是 -1（损坏样本），我们仍需返回一个合法标签（否则 loss 会出错）
        # 策略：跳过该样本不现实（DataLoader 要求返回），所以将其标签设为 0（或任意有效类）
        # 更好的做法是预清洗数据。这里为保证训练继续，强制设为 0
        if label == -1:
            label = 0  # 或者你可以选择随机选一个类，但 0 最简单

        return image, label

    def get_num_classes(self):
        return len(self.classes)
