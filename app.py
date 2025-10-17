from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid
import logging
import json
from config import Config
from utils.db import get_db_connection
from threading import Lock

# 在导入任何会加载底层 BLAS / OpenMP 的库前设置线程限制
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

app = Flask(__name__, static_folder=None, template_folder="templates")
app.config.from_object(Config)
Config.init_app(app)

# -------------------- Logger --------------------
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# -------------------- 文件上传目录 --------------------
UPLOAD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_ROOT, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_ROOT

# -------------------- CORS --------------------
from flask_cors import CORS

CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"])

# -------------------- 延迟初始化模型 --------------------
predictor = None
predict_lock = Lock()


def get_predictor():
    global predictor
    if predictor is None:
        from models.predictor import MineralPredictor

        # 延迟导入 torch，并在导入后限制线程（避免主进程/WSGI 与 native 库冲突）
        import torch
        # 根据 Config 或 CPU 核数设置合理线程数
        max_threads = getattr(Config, 'TORCH_NUM_THREADS', None)
        if max_threads is None:
            try:
                cpus = os.cpu_count() or 4
                max_threads = min(8, max(2, cpus // 2))
            except Exception:
                max_threads = 4
        try:
            torch.set_num_threads(int(max_threads))
            torch.set_num_interop_threads(1)
            # 可选：在 CPU 上禁用 mkldnn 以避免稀有 MKLDNN 崩溃
            torch.backends.mkldnn.enabled = False
        except Exception:
            pass

        device = getattr(Config, 'PREDICTOR_DEVICE', 'cpu')
        app.logger.info(f"🎯 使用设备: {device}")

        # 检查文件是否存在并创建 predictor（模型加载在内部）
        if not os.path.exists(Config.MODEL_PATH):
            app.logger.error(f"❌ 模型文件不存在: {Config.MODEL_PATH}")
            raise FileNotFoundError(f"模型文件不存在: {Config.MODEL_PATH}")
        if not os.path.exists(Config.CLASS_MAPPING_PATH):
            app.logger.error(f"❌ 类别映射文件不存在: {Config.CLASS_MAPPING_PATH}")
            raise FileNotFoundError(f"类别映射文件不存在: {Config.CLASS_MAPPING_PATH}")

        predictor = MineralPredictor(
            model_path=Config.MODEL_PATH,
            class_mapping_path=Config.CLASS_MAPPING_PATH,
            full_mapping_path=getattr(Config, 'FULL_MAPPING_PATH', None),
            device=device
        )
        app.logger.info("✅ 模型加载完成")
    else:
        app.logger.info("♻️ 使用已加载的模型实例")
    return predictor



# -------------------- 请求日志 --------------------
@app.before_request
def _log_req():
    app.logger.debug("FLASK REQ %s %s from %s", request.method, request.path, request.remote_addr)


# -------------------- 页面路由 --------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin')
def admin_page():
    return render_template('admin.html')


# -------------------- 上传文件访问 --------------------
@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.debug("Serving upload file: %s -> %s", filename, filepath)
    if not os.path.exists(filepath):
        return jsonify({'error': 'not found'}), 404
    # send_from_directory 会安全地返回文件
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



# -------------------- API: /api/predict --------------------
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True}), 200

    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png']:
        return jsonify({'error': '不支持的文件类型'}), 400

    # 保存上传文件
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    try:
        os.chmod(filepath, 0o644)
    except Exception:
        app.logger.debug("chmod failed, continuing")

    normalized_path = filepath.replace('\\', '/')

    # -------------------- 预测 --------------------
    top_k = getattr(Config, 'PREDICT_TOP_K', 3)  # 从 Config 获取 top_k
    with predict_lock:
        try:
            predictor = get_predictor()
            results = predictor.predict(normalized_path, top_k=top_k)
        except Exception as e:
            app.logger.exception("预测过程中发生异常")
            return jsonify({'error': '预测时出错', 'detail': str(e)}), 500

    if isinstance(results, dict) and results.get('error'):
        return jsonify(results), 400

    # -------------------- 保存预测请求到数据库 --------------------
    from utils.db import get_db_connection


    model_used = os.path.basename(Config.MODEL_PATH)
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            top_k_json = json.dumps(results)  # 保存 top_k 结果
            cursor.execute("""
                INSERT INTO prediction_requests
                (image_path, top_k_results, model_used, submitted_at)
                VALUES (%s, %s, %s, NOW())
            """, (filename, top_k_json, model_used))
            conn.commit()
            cursor.close()
            conn.close()
        else:
            app.logger.warning("数据库连接失败，预测请求未入库")
    except Exception as e:
        app.logger.exception("保存预测请求失败")

    return jsonify({
        'success': True,
        'results': results,
        'image_url': f"/uploads/{filename}"
    }), 200



# -------------------- API: /api/feedback --------------------
@app.route('/api/feedback', methods=['POST', 'OPTIONS'])
def api_feedback():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True}), 200

    data = request.get_json(force=True, silent=True) or {}
    image_path = data.get('image_path')
    predicted_label_idx = data.get('predicted_label_idx')
    predicted_label_text = data.get('predicted_label_text')
    predicted_label = data.get('predicted_label')
    true_label = data.get('true_label')
    comment = data.get('comment', '')

    if not image_path or not true_label:
        return jsonify({'error': '缺少必要字段 (image_path 或 true_label)'}), 400

    image_path = image_path.replace('\\', '/')


    pred_idx, pred_text = None, None
    try:
        pred_idx = int(predicted_label_idx or predicted_label)
    except (ValueError, TypeError):
        pred_text = str(predicted_label_text or predicted_label or '')

    true_idx, true_text = None, None
    try:
        true_idx = int(true_label)
    except (ValueError, TypeError):
        true_text = str(true_label)

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': '数据库连接失败'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_feedback (
                image_path,
                predicted_label_idx,
                predicted_label_text,
                true_label_idx,
                true_label_text,
                user_comment,
                status,
                source
            ) VALUES (%s, %s, %s, %s, %s, %s, 'pending', 'web')
        """, (image_path, pred_idx, pred_text, true_idx, true_text, comment))
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.exception("DB insert failed")
        return jsonify({'error': '数据库写入失败', 'detail': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({'message': '反馈已提交，感谢！'}), 200


# -------------------- 管理端接口 --------------------
@app.route('/api/admin/feedbacks', methods=['GET', 'OPTIONS'])
def api_get_admin_feedbacks():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True}), 200

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': '数据库连接失败'}), 500
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM user_feedback ORDER BY submitted_at DESC")
        return jsonify({'feedbacks': cursor.fetchall()}), 200
    finally:
        cursor.close()
        conn.close()


@app.route('/api/admin/feedbacks/<int:fid>/review', methods=['POST', 'OPTIONS'])
def api_admin_review_feedback(fid):
    if request.method == 'OPTIONS':
        return jsonify({'ok': True}), 200

    data = request.get_json(force=True, silent=True) or {}
    action = data.get('action')
    admin_comment = data.get('admin_comment', '')
    admin_user = data.get('admin_user', 'admin')

    if action not in ('approve', 'reject'):
        return jsonify({'error': 'invalid action'}), 400

    used_for_training = 1 if action == 'approve' else 0
    new_status = 'verified' if action == 'approve' else 'rejected'

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': '数据库连接失败'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_feedback
            SET status=%s, admin_comment=%s, reviewed_by=%s, reviewed_at=NOW(), used_for_training=%s
            WHERE id=%s
        """, (new_status, admin_comment, admin_user, used_for_training, fid))
        conn.commit()
        return jsonify({'message': '更新成功'}), 200
    finally:
        cursor.close()
        conn.close()


@app.route('/api/admin/stats', methods=['GET', 'OPTIONS'])
def api_admin_stats():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True}), 200

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': '数据库连接失败'}), 500

    try:
        cursor = conn.cursor(dictionary=True)

        # 获取类别分布（从user_feedback表）
        cursor.execute("""
            SELECT true_label_text AS label, COUNT(*) AS count
            FROM user_feedback
            GROUP BY true_label_text
            ORDER BY count DESC
        """)
        class_distribution = cursor.fetchall()

        # 获取当前模型名称
        current_model = os.path.basename(Config.MODEL_PATH)

        # 获取当前模型的预测请求总数
        cursor.execute("SELECT COUNT(*) AS total_requests FROM prediction_requests WHERE model_used = %s",
                       (current_model,))
        current_model_result = cursor.fetchone()
        current_model_requests = current_model_result['total_requests'] if current_model_result else 0

        # 获取总预测请求数
        cursor.execute("SELECT COUNT(*) AS total_requests FROM prediction_requests")
        total_result = cursor.fetchone()
        total_prediction_requests = total_result['total_requests'] if total_result else 0

        # 修复：使用更安全的查询方式获取当前模型的已验证反馈数量
        # 先获取所有已验证反馈的image_path
        cursor.execute("SELECT image_path FROM user_feedback WHERE status='verified'")
        verified_feedbacks = cursor.fetchall()

        verified_feedback_count = 0
        if verified_feedbacks:
            # 构建image_path列表
            image_paths = [fb['image_path'] for fb in verified_feedbacks if fb.get('image_path')]
            if image_paths:
                # 使用IN查询而不是JOIN，避免复杂连接
                placeholders = ','.join(['%s'] * len(image_paths))
                query = f"""
                    SELECT COUNT(*) as count 
                    FROM prediction_requests 
                    WHERE model_used = %s AND image_path IN ({placeholders})
                """
                params = [current_model] + image_paths
                cursor.execute(query, params)
                verified_result = cursor.fetchone()
                verified_feedback_count = verified_result['count'] if verified_result else 0

        # 获取模型使用分布
        cursor.execute("""
            SELECT model_used AS model, COUNT(*) AS count
            FROM prediction_requests
            WHERE model_used IS NOT NULL AND model_used != ''
            GROUP BY model_used
            ORDER BY count DESC
        """)
        model_distribution = cursor.fetchall()

        # 计算当前模型的准确率：(当前模型预测请求总数 - 当前模型已验证反馈数量) / 当前模型预测请求总数
        if current_model_requests > 0:
            accuracy = round((current_model_requests - verified_feedback_count) / current_model_requests, 3)
        else:
            accuracy = 0

        # 计算当前模型使用率
        if total_prediction_requests > 0:
            current_model_usage = round((current_model_requests / total_prediction_requests) * 100, 1)
        else:
            current_model_usage = 0

        return jsonify({
            'class_distribution': class_distribution,
            'model_distribution': model_distribution,
            'accuracy': accuracy,
            'total_requests': total_prediction_requests,
            'current_model': current_model,
            'current_model_usage': current_model_usage,
            'current_model_requests': current_model_requests
        }), 200
    except Exception as e:
        app.logger.exception("获取统计数据失败")
        return jsonify({'error': '获取统计数据失败', 'detail': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# -------------------- API: 用户请求历史 --------------------
import datetime

import json


@app.route('/api/admin/requests', methods=['GET', 'OPTIONS'])
def api_admin_requests():
    if request.method == 'OPTIONS':
        return jsonify({'ok': True}), 200

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': '数据库连接失败'}), 500

    try:
        cursor = conn.cursor(dictionary=True)

        # 获取所有预测记录，包括top_k_results和model_used
        cursor.execute("""
            SELECT 
                pr.id,
                pr.image_path,
                pr.predicted_label_text as predicted_label,
                pr.top_k_results,
                pr.model_used,
                pr.submitted_at as timestamp
            FROM prediction_requests pr
            WHERE pr.image_path IS NOT NULL
            ORDER BY pr.submitted_at DESC
        """)
        predictions = cursor.fetchall()

        # 获取所有反馈记录，用于后续关联
        cursor.execute("""
            SELECT 
                image_path,
                true_label_text,
                status,
                user_comment
            FROM user_feedback
            WHERE status IS NOT NULL
        """)
        feedbacks = cursor.fetchall()

        # 创建反馈记录的映射字典，以image_path为键
        feedback_dict = {}
        for fb in feedbacks:
            image_path = fb['image_path']
            if image_path:
                # 统一处理图片路径格式
                if os.path.isabs(image_path) or 'uploads' in image_path:
                    image_path = os.path.basename(image_path)
                feedback_dict[image_path] = {
                    'true_label_text': fb['true_label_text'],
                    'status': fb['status'],
                    'user_comment': fb['user_comment']
                }

        # 构建最终结果
        requests = []
        for pred in predictions:
            image_path = pred['image_path']
            if image_path:
                # 统一处理图片路径格式
                if os.path.isabs(image_path) or 'uploads' in image_path:
                    image_path = os.path.basename(image_path)

                # 查找对应的反馈信息
                feedback = feedback_dict.get(image_path)

                # 从top_k_results中提取置信度最高的预测矿物
                predicted_mineral = "无"
                top_k_results = pred.get('top_k_results')
                if top_k_results:
                    try:
                        # 解析JSON字符串
                        results_list = json.loads(top_k_results)
                        if isinstance(results_list, list) and len(results_list) > 0:
                            # 找到置信度最高的结果
                            top_result = max(results_list, key=lambda x: x.get('confidence', 0))
                            predicted_mineral = top_result.get('label_zh', '无')
                    except (json.JSONDecodeError, ValueError) as e:
                        app.logger.warning(f"解析top_k_results失败: {e}")

                request_item = {
                    'id': pred['id'],
                    'image_path': image_path,
                    'predicted_label': predicted_mineral,
                    'actual_mineral': feedback['true_label_text'] if feedback else "无",
                    'comments': feedback['user_comment'] if feedback else "无",
                    'review_status': feedback['status'] if feedback else "predicted",
                    'timestamp': pred['timestamp'].isoformat() if isinstance(pred['timestamp'], datetime.datetime) else
                    pred['timestamp'],
                    'type': 'feedback' if feedback else 'prediction',
                    'model_used': pred.get('model_used', 'unknown')  # 添加模型名称
                }
                requests.append(request_item)

        return jsonify({'requests': requests}), 200

    except Exception as e:
        app.logger.exception("获取用户请求历史失败")
        return jsonify({'error': '获取数据失败', 'detail': str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/api/debug/model', methods=['GET'])
def api_debug_model():
    """调试模型信息"""
    try:
        predictor = get_predictor()

        # 获取模型信息
        model_info = {
            'device': str(predictor.device),
            'num_classes': len(predictor.idx_to_class),
            'model_structure': str(type(predictor.model)),
            'sample_mappings': {}
        }

        # 显示前5个类别映射
        for i in range(min(5, len(predictor.idx_to_class))):
            model_info['sample_mappings'][i] = predictor.idx_to_class[i]

        return jsonify(model_info)
    except Exception as e:
        app.logger.exception("调试模型失败")
        return jsonify({'error': str(e)}), 500

# -------------------- 启动 --------------------
if __name__ == '__main__':
    try:
        from waitress import serve
        app.logger.info("使用 waitress 启动（生产/开发均可）")
        serve(app, host='0.0.0.0', port=5050)
    except Exception:
        app.logger.info("waitress 未安装，使用 Flask 内置服务器（单线程、禁用重载器）")
        app.run(debug=False, host='0.0.0.0', port=5050, use_reloader=False, threaded=False)

