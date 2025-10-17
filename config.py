# config.py
import os

class Config:
    DATABASE = {
        'host': 'localhost',
        'database': 'mineral_db',
        'user': 'root',
        'password': 'sql2008',
        'charset': 'utf8mb4'
    }

    # -------------------- 模型相关 --------------------
    MODEL_PATH = 'checkpoints/mineral_model_best/mineral_model_best.pth'
    CLASS_MAPPING_PATH = 'checkpoints/mineral_model_best/class_mapping.json'        # 训练模型用的英文类别映射
    FULL_MAPPING_PATH = 'data/classes2062.json'  # 完整中英文映射表（可选）
    PREDICTOR_DEVICE = 'cpu'                            # 预测使用的设备，可选 'cpu' / 'cuda'
    PREDICT_TOP_K = 3                                   # 返回 top K 个预测结果

    # -------------------- 文件上传 --------------------
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit

    # -------------------- 自动创建目录 --------------------
    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
