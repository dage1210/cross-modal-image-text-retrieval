import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import json
import requests
import base64
from transformers import CLIPProcessor, CLIPModel
import chromadb
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import tempfile
import shutil

# 配置
IMAGE_DIR = r"D:\42-qwen\img"
UPLOAD_IMAGE_DIR = r"D:\42-qwen\uploaded_img"  # 新增：上传图片保存目录
UPLOAD_IMAGE_DIR_baidu = r"D:\BaiduNetdiskDownload\5000-image"  # 新增：上传图片保存目录
CHROMA_PATH = "./chroma_multimodal_db"
IMAGE_COLLECTION_NAME = "image_features_collection"
TEXT_COLLECTION_NAME = "text_features_collection"
OLLAMA_URL = "http://192.168.12.197:11434"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_VL_MODEL = "qwen3-vl:2b"
DESCRIPTIONS_JSON = "image_descriptions.json"
TEMP_UPLOAD_FOLDER = tempfile.mkdtemp()  # 临时上传目录
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['TEMP_UPLOAD_FOLDER'] = TEMP_UPLOAD_FOLDER

# 确保目录存在
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)  # 新增：创建上传目录

class MultimodalSearch:
    def __init__(self):
        """初始化多模态搜索系统"""
        # 初始化CLIP模型（图像特征提取）
        print("Loading CLIP model for image features...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # 获取特征维度
        self.image_feature_dim = self._get_clip_feature_dimension()
        print(f"Image feature dimension (CLIP): {self.image_feature_dim}")
        
        self.text_feature_dim = self._get_text_embedding_dimension()
        print(f"Text feature dimension (qwen3-embedding): {self.text_feature_dim}")
        
        # 初始化Chroma客户端（复用现有集合，不删除旧数据）
        print("Initializing Chroma database...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # 初始化图像特征集合（512维）
        try:
            self.image_collection = self.chroma_client.get_collection(IMAGE_COLLECTION_NAME)
        except:
            self.image_collection = self.chroma_client.create_collection(
                name=IMAGE_COLLECTION_NAME,
                metadata={"dimension": self.image_feature_dim}
            )
        
        # 初始化文本特征集合（1024维）
        try:
            self.text_collection = self.chroma_client.get_collection(TEXT_COLLECTION_NAME)
        except:
            self.text_collection = self.chroma_client.create_collection(
                name=TEXT_COLLECTION_NAME,
                metadata={"dimension": self.text_feature_dim}
            )
        
        print("Connected to Chroma successfully")
        
        # 加载现有描述数据
        self.load_existing_data()
    
    def load_existing_data(self):
        """加载已有的图片描述数据"""
        self.data = []
        if os.path.exists(DESCRIPTIONS_JSON):
            try:
                with open(DESCRIPTIONS_JSON, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"Loaded {len(self.data)} existing descriptions")
            except:
                print("Failed to load existing descriptions, starting fresh")
                self.data = []
    
    def _get_clip_feature_dimension(self):
        """获取CLIP图像特征维度"""
        test_image = Image.new('RGB', (224, 224))
        inputs = self.clip_processor(images=test_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(** inputs)
        return features.shape[1]
    
    def _get_text_embedding_dimension(self):
        """获取文本嵌入维度"""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": "测试"},
                timeout=30
            )
            response.raise_for_status()
            return len(response.json()["embedding"])
        except Exception as e:
            print(f"获取文本嵌入维度失败: {e}")
            raise
    
    def generate_image_description(self, image_path):
        """生成图片描述"""
        try:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                "model": OLLAMA_VL_MODEL,
                "prompt": "请用中文详细描述这张图片的内容，包括主要对象、场景、颜色和活动。",
                "images": [encoded_image],
                "stream": False
            }
            
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip() or "这是一张图片"
        except Exception as e:
            print(f"生成图片描述失败 {image_path}: {e}")
            return "这是一张图片"
            
    def extract_image_features(self, image_path):
        """提取图像特征"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features.cpu().numpy().flatten().astype(np.float32).tolist()
        except Exception as e:
            print(f"提取图像特征失败 {image_path}: {e}")
            return None
    
    def extract_text_features(self, text):
        """提取文本特征（qwen3-embedding）"""
        try:
            if not text.strip():
                return None
                
            if len(text) > 2000:
                text = text[:2000]
                
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not embedding:
                return None
                
            # L2归一化
            embedding_np = np.array(embedding, dtype=np.float32)
            return (embedding_np / np.linalg.norm(embedding_np, ord=2)).tolist()
        except Exception as e:
            print(f"提取文本特征失败: {e}")
            return None
            
    def process_images(self, image_dir):
        """处理初始图片目录"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            
        print(f"找到 {len(image_paths)} 张图片")
        processed_paths = {item['image_path'] for item in self.data}
        
        print("处理初始图片中...")
        for img_path in tqdm(image_paths, desc="处理进度"):
            if img_path in processed_paths:
                continue
                
            self._process_single_image(img_path)
                
        return self.data
    
    def _process_single_image(self, img_path):
        """处理单张图片（抽取特征并保存）"""
        try:
            description = self.generate_image_description(img_path)
            image_feat = self.extract_image_features(img_path)
            text_feat = self.extract_text_features(description)
            
            if image_feat and text_feat:
                if len(image_feat) != self.image_feature_dim:
                    print(f"图像特征维度不匹配 {img_path}")
                    return False
                if len(text_feat) != self.text_feature_dim:
                    print(f"文本特征维度不匹配 {img_path}")
                    return False
                    
                item = {
                    "image_path": img_path,
                    "description": description,
                    "image_features": image_feat,
                    "text_features": text_feat
                }
                self.data.append(item)
                processed_paths = {item['image_path'] for item in self.data}
                
                # 保存到JSON
                with open(DESCRIPTIONS_JSON, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                return True
            else:
                print(f"处理失败 {img_path}")
                return False
        except Exception as e:
            print(f"处理单张图片出错 {img_path}: {e}")
            return False
    
    
    def process_uploaded_images(self, uploaded_files):
        """处理上传的图片（增加重复路径检查）"""
        results = []
        # 先获取已处理的图片路径，避免重复处理
        processed_paths = {item['image_path'] for item in self.data}
        
        for file in uploaded_files:
            try:
                # 保存上传的图片到指定目录
                filename = os.path.basename(file.filename)
                save_path = os.path.join(UPLOAD_IMAGE_DIR, filename)
                
                # 避免文件名重复
                counter = 1
                while os.path.exists(save_path):
                    name, ext = os.path.splitext(filename)
                    save_path = os.path.join(UPLOAD_IMAGE_DIR, f"{name}_{counter}{ext}")
                    counter += 1
                
                # 检查是否已处理过该图片（基于最终保存路径）
                if save_path in processed_paths:
                    results.append({
                        "status": "warning",
                        "filename": filename,
                        "message": "该图片已存在于向量库中，无需重复处理"
                    })
                    continue
                
                file.save(save_path)
                print(f"已保存上传图片: {save_path}")
                
                # 处理图片
                success = self._process_single_image(save_path)
                if success:
                    processed_paths.add(save_path)
                    results.append({
                        "status": "success",
                        "filename": filename,
                        "save_path": save_path,
                        "message": "图片处理成功并添加到向量库"
                    })
                else:
                    results.append({
                        "status": "error",
                        "filename": filename,
                        "message": "图片处理失败"
                    })
            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": file.filename if hasattr(file, 'filename') else "未知文件",
                    "message": f"处理出错: {str(e)}"
                })
        
        # 将新处理的图片特征添加到向量库
        self.save_new_features_to_chroma()
        return results
    
    def save_to_chroma(self, data):
        """批量保存初始数据到向量库"""
        print("保存初始数据到Chroma数据库...")
        self._save_features_to_chroma(data)
    
    def _get_base_id(self, image_path):
        """废弃基于路径的固定ID，改用随机唯一ID"""
        return str(uuid.uuid4())  # 关键修改：使用UUID4随机生成唯一ID

    def save_new_features_to_chroma(self):
        """仅保存新增的特征到向量库（修复重复ID问题）"""
        try:
            # 正确获取现有ID（处理空集合情况）
            existing_image_ids = set(self.image_collection.get()["ids"]) if self.image_collection.count() > 0 else set()
            existing_text_ids = set(self.text_collection.get()["ids"]) if self.text_collection.count() > 0 else set()
        except Exception as e:
            print(f"获取现有ID失败: {e}，假设集合为空")
            existing_image_ids = set()
            existing_text_ids = set()
        
        # 筛选未插入的新数据（基于图片路径去重）
        processed_paths = set()
        new_data = []
        for item in self.data:
            img_path = item["image_path"]
            if img_path in processed_paths:
                continue  # 跳过同一图片的重复数据
            processed_paths.add(img_path)
            
            # 生成临时ID用于检查
            base_id = self._get_base_id(img_path)  # 这里只是临时检查用，实际插入时会重新生成
            image_id = f"{base_id}_image"
            text_id = f"{base_id}_text"
            
            if image_id not in existing_image_ids and text_id not in existing_text_ids:
                new_data.append(item)
        
        if new_data:
            print(f"发现 {len(new_data)} 条新数据，保存到向量库...")
            self._save_features_to_chroma(new_data)
        else:
            print("没有新数据需要保存到向量库")
    
       
    
    
    def _get_base_id(self, image_path):
        """根据图片路径生成唯一ID（确保同一图片ID一致）"""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, image_path))
    
    
    def _save_features_to_chroma(self, data):
        """实际执行特征保存逻辑（增加ID重复检查）"""
        # 图像特征数据
        image_ids = []
        image_embeddings = []
        image_documents = []
        
        # 文本特征数据
        text_ids = []
        text_embeddings = []
        text_documents = []
        
        # 再次获取现有ID，确保没有遗漏
        try:
            existing_image_ids = set(self.image_collection.get()["ids"]) if self.image_collection.count() > 0 else set()
            existing_text_ids = set(self.text_collection.get()["ids"]) if self.text_collection.count() > 0 else set()
        except:
            existing_image_ids = set()
            existing_text_ids = set()
        
        for item in data:
            # 生成唯一ID
            base_id = self._get_base_id(item["image_path"])
            image_id = f"{base_id}_image"
            text_id = f"{base_id}_text"
            
            # 检查并处理ID冲突（极端情况）
            while image_id in existing_image_ids:
                print(f"图像ID {image_id} 已存在，重新生成...")
                base_id = self._get_base_id(item["image_path"])
                image_id = f"{base_id}_image"
                text_id = f"{base_id}_text"
            
            while text_id in existing_text_ids:
                print(f"文本ID {text_id} 已存在，重新生成...")
                base_id = self._get_base_id(item["image_path"])
                image_id = f"{base_id}_image"
                text_id = f"{base_id}_text"
            
            # 添加到列表
            image_ids.append(image_id)
            image_embeddings.append(item["image_features"])
            image_documents.append(f"{item['image_path']}||{item['description']}")
            
            text_ids.append(text_id)
            text_embeddings.append(item["text_features"])
            text_documents.append(f"{item['image_path']}||{item['description']}")
            
            # 更新现有ID集合（避免同一批数据内冲突）
            existing_image_ids.add(image_id)
            existing_text_ids.add(text_id)
        
        # 保存到图像集合
        if image_embeddings:
            print(f"插入 {len(image_embeddings)} 个图像特征 (维度: {self.image_feature_dim})")
            self.image_collection.upsert(
                embeddings=image_embeddings,
                documents=image_documents,
                ids=image_ids
            )
        
        # 保存到文本集合
        if text_embeddings:
            print(f"插入 {len(text_embeddings)} 个文本特征 (维度: {self.text_feature_dim})")
            self.text_collection.upsert(
                embeddings=text_embeddings,
                documents=text_documents,
                ids=text_ids
            )
            
        
    def search_by_text(self, query_text, top_k=5):
        """文本检索"""
        print(f"文本检索: {query_text}")
        
        query_feat = self.extract_text_features(query_text)
        if not query_feat:
            print("提取查询文本特征失败")
            return []
            
        try:
            results = self.text_collection.query(
                query_embeddings=[query_feat],
                n_results=top_k * 2,
                include=["documents", "distances"]
            )
            
            similar_items = []
            added_paths = set()
            
            for i in range(len(results['ids'][0])):
                doc = results['documents'][0][i]
                distance = 1 - (results['distances'][0][i] / 2)  # 转换为相似度
                
                parts = doc.split("||")
                if len(parts) >= 2:
                    img_path = parts[0]
                    desc = parts[1]
                    
                    if img_path in added_paths:
                        continue
                        
                    added_paths.add(img_path)
                    similar_items.append({
                        "image_path": img_path,
                        "description": desc,
                        "similarity": float(distance)
                    })
                    
                    if len(similar_items) >= top_k:
                        break
                        
            return similar_items
        except Exception as e:
            print(f"文本检索失败: {e}")
            return []
            
    def search_by_image(self, query_image_path, top_k=5):
        """图像检索"""
        print(f"图像检索: {query_image_path}")
        
        query_feat = self.extract_image_features(query_image_path)
        if not query_feat:
            print("提取查询图像特征失败")
            return []
            
        try:
            results = self.image_collection.query(
                query_embeddings=[query_feat],
                n_results=top_k * 2,
                include=["documents", "distances"]
            )
            
            similar_items = []
            added_paths = set()
            query_abs = os.path.abspath(query_image_path)
            
            for i in range(len(results['ids'][0])):
                doc = results['documents'][0][i]
                distance = 1 - (results['distances'][0][i] / 2)
                
                parts = doc.split("||")
                if len(parts) >= 2:
                    img_path = parts[0]
                    desc = parts[1]
                    abs_path = os.path.abspath(img_path)
                    
                    if abs_path == query_abs or abs_path in added_paths:
                        continue
                        
                    added_paths.add(abs_path)
                    similar_items.append({
                        "image_path": img_path,
                        "description": desc,
                        "similarity": float(distance)
                    })
                    
                    if len(similar_items) >= top_k:
                        break
                        
            return similar_items
        except Exception as e:
            print(f"图像检索失败: {e}")
            return []
    
    def get_database_info(self):
        """获取向量库总体信息"""
        try:
            # 获取文件数量
            image_count = self.image_collection.count()
            text_count = self.text_collection.count()
            
            # 获取数据库大小
            db_size = self._get_database_size()
            
            return {
                "image_count": image_count,
                "text_count": text_count,
                "total_count": image_count + text_count,
                "db_size": db_size,
                "db_type": "Chroma"
            }
        except Exception as e:
            print(f"获取数据库信息失败: {e}")
            return {
                "image_count": 0,
                "text_count": 0,
                "total_count": 0,
                "db_size": "未知",
                "db_type": "Chroma"
            }
    
    def _get_database_size(self):
        """获取数据库大小"""
        try:
            if os.path.exists(CHROMA_PATH):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(CHROMA_PATH):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                # 转换为MB
                return f"{total_size / (1024*1024):.2f} MB"
            else:
                return "0 MB"
        except Exception as e:
            print(f"计算数据库大小失败: {e}")
            return "未知"
    
    def get_all_items(self, offset=0, limit=20):
        """获取所有向量库中的项目"""
        try:
            # 获取图像集合中的所有项目
            image_results = self.image_collection.get(
                include=["documents"],
                limit=limit,
                offset=offset
            )
            
            items = []
            for i, doc in enumerate(image_results['documents']):
                if i >= limit:
                    break
                    
                parts = doc.split("||")
                if len(parts) >= 2:
                    img_path = parts[0]
                    desc = parts[1]
                    
                    # 获取对应的文本项
                    text_doc = None
                    try:
                        text_results = self.text_collection.get(
                            include=["documents"],
                            where_document={"$contains": img_path}
                        )
                        if text_results['documents']:
                            text_parts = text_results['documents'][0].split("||")
                            if len(text_parts) >= 2:
                                text_doc = text_parts[1]
                    except:
                        pass
                    
                    items.append({
                        "id": image_results['ids'][i],
                        "image_path": img_path,
                        "description": desc,
                        "text_description": text_doc
                    })
            
            return items
        except Exception as e:
            print(f"获取所有项目失败: {e}")
            return []
    
    def search_items_by_filename(self, filename):
        """根据文件名搜索项目"""
        try:
            # 在图像集合中搜索
            image_results = self.image_collection.get(
                include=["documents"],
                where_document={"$contains": filename}
            )
            
            items = []
            for i, doc in enumerate(image_results['documents']):
                parts = doc.split("||")
                if len(parts) >= 2:
                    img_path = parts[0]
                    desc = parts[1]
                    
                    # 获取对应的文本项
                    text_doc = None
                    try:
                        text_results = self.text_collection.get(
                            include=["documents"],
                            where_document={"$contains": img_path}
                        )
                        if text_results['documents']:
                            text_parts = text_results['documents'][0].split("||")
                            if len(text_parts) >= 2:
                                text_doc = text_parts[1]
                    except:
                        pass
                    
                    items.append({
                        "id": image_results['ids'][i],
                        "image_path": img_path,
                        "description": desc,
                        "text_description": text_doc
                    })
            
            return items
        except Exception as e:
            print(f"根据文件名搜索项目失败: {e}")
            return []

# 初始化系统
searcher = MultimodalSearch()
# 处理初始图片（如果需要）
# data = searcher.process_images(IMAGE_DIR)
# searcher.save_to_chroma(data)

# 辅助函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 路由
@app.route('/')
def index():
    """检索首页"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    """新增：图片上传页面"""
    if request.method == 'POST':
        print("=== Upload Debug Info ===")
        print(f"Request Content-Type: {request.content_type}")
        print(f"Request Files Keys: {list(request.files.keys()) if request.files else 'No files'}")
        print(f"Request Form Data: {request.form}")
        print(f"Request Files Count: {len(request.files)}")
        
        # 处理上传的图片
        if 'images' not in request.files:
            print("Error: 'images' key not found in request.files")
            return jsonify({"error": "请选择图片文件"}), 400
            
        files = request.files.getlist('images')
        print(f"Files received: {len(files)} files")
        for i, file in enumerate(files):
            print(f"File {i}: filename='{file.filename}', content_type='{file.content_type}'")
            
        if not files or all(file.filename == '' for file in files):
            print("Error: No valid files selected")
            return jsonify({"error": "请选择至少一张图片"}), 400
            
        # 筛选合法文件
        valid_files = [f for f in files if allowed_file(f.filename)]
        print(f"Valid files: {len(valid_files)} files")
        if not valid_files:
            print("Error: No valid file extensions")
            return jsonify({"error": "没有合法的图片文件（支持png, jpg, jpeg, gif）"}), 400
            
        # 处理上传的图片
        results = searcher.process_uploaded_images(valid_files)
        print(f"Processing results: {results}")
        return jsonify({"results": results})
    else:
        # GET 请求，显示上传页面
        return render_template('upload.html')

@app.route('/upload/process', methods=['POST'])
def upload_process():
    """处理上传的图片"""
    print("=== Upload/Process Debug Info ===")
    print(f"Request Content-Type: {request.content_type}")
    print(f"Request Files Keys: {list(request.files.keys()) if request.files else 'No files'}")
    print(f"Request Form Data: {request.form}")
    print(f"Request Files Count: {len(request.files)}")
    
    # 处理上传的图片
    if 'images' not in request.files:
        print("Error: 'images' key not found in request.files")
        return jsonify({"error": "请选择图片文件"}), 400
        
    files = request.files.getlist('images')
    print(f"Files received: {len(files)} files")
    for i, file in enumerate(files):
        print(f"File {i}: filename='{file.filename}', content_type='{file.content_type}'")
        
    if not files or all(file.filename == '' for file in files):
        print("Error: No valid files selected")
        return jsonify({"error": "请选择至少一张图片"}), 400
        
    # 筛选合法文件
    valid_files = [f for f in files if allowed_file(f.filename)]
    print(f"Valid files: {len(valid_files)} files")
    if not valid_files:
        print("Error: No valid file extensions")
        return jsonify({"error": "没有合法的图片文件（支持png, jpg, jpeg, gif）"}), 400
        
    # 处理上传的图片
    results = searcher.process_uploaded_images(valid_files)
    print(f"Processing results: {results}")
    return jsonify({"results": results})

    valid_files = [f for f in files if allowed_file(f.filename)]
    if not valid_files:
        return jsonify({"error": "没有合法的图片文件（支持png, jpg, jpeg, gif）"}), 400
        
    # 处理上传的图片
    results = searcher.process_uploaded_images(valid_files)
    return jsonify({"results": results})

@app.route('/database')
def database_page():
    """新增：向量库查看页面"""
    return render_template('database.html')

@app.route('/vectorize')
def vectorize_page():
    """新增：向量化处理页面"""
    return render_template('vectorize.html')

@app.route('/search/text', methods=['POST'])
def search_text():
    """文本搜索接口"""
    query = request.form.get('query', '')
    top_k = int(request.form.get('top_k', 5))
    
    if not query:
        return jsonify({"error": "请输入搜索关键词"}), 400
        
    results = searcher.search_by_text(query, top_k)
    return jsonify({"results": results})

@app.route('/search/image', methods=['POST'])
def search_image():
    """图片搜索接口"""
    top_k = int(request.form.get('top_k', 5))
    
    if 'image' not in request.files:
        return jsonify({"error": "请选择一张图片"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "请选择一张图片"}), 400
        
    if file and allowed_file(file.filename):
        # 保存临时查询图片
        temp_filename = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
        file.save(temp_filename)
        
        # 执行搜索
        results = searcher.search_by_image(temp_filename, top_k)
        
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return jsonify({"results": results})
    
    return jsonify({"error": "不支持的图片格式"}), 400

@app.route('/database/info')
def database_info():
    """获取向量库总体信息"""
    info = searcher.get_database_info()
    return jsonify(info)

@app.route('/database/items')
def database_items():
    """获取向量库中的项目"""
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 20))
    items = searcher.get_all_items(offset, limit)
    return jsonify(items)

@app.route('/database/search')
def database_search():
    """根据文件名搜索项目"""
    filename = request.args.get('filename', '')
    if not filename:
        return jsonify([])
    items = searcher.search_items_by_filename(filename)
    return jsonify(items)

@app.route('/image/<path:filename>')
def serve_image(filename):
    """提供图片访问"""
    # 允许访问初始图片目录和上传图片目录
    allowed_dirs = [os.path.abspath(IMAGE_DIR), os.path.abspath(UPLOAD_IMAGE_DIR),os.path.abspath(UPLOAD_IMAGE_DIR_baidu)]
    file_abs_path = os.path.abspath(filename)
    
    if any(file_abs_path.startswith(dir_path) for dir_path in allowed_dirs):
        directory = os.path.dirname(filename)
        file = os.path.basename(filename)
        return send_from_directory(directory, file)
    
    return "图片不存在", 404

def create_frontend():
    """创建前端页面（包含新增的上传页面）"""
    # 创建检索首页
    index_html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多模态图像检索系统</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .navbar {
            margin-bottom: 20px;
            padding: 10px 0;
            border-bottom: 1px solid #ddd;
        }
        .navbar a {
            margin-right: 20px;
            text-decoration: none;
            color: #3498db;
            font-weight: bold;
        }
        .navbar a:hover {
            color: #2980b9;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .search-tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            font-weight: bold;
            color: #666;
            position: relative;
        }
        .tab.active {
            color: #2c3e50;
        }
        .tab.active::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 3px;
            background-color: #3498db;
        }
        .search-panel {
            display: none;
        }
        .search-panel.active {
            display: block;
        }
        .text-search {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        #query {
            flex: 1;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .image-upload {
            margin-bottom: 20px;
            border: 2px dashed #ddd;
            padding: 40px;
            text-align: center;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .image-upload:hover {
            border-color: #3498db;
        }
        #image-preview {
            max-width: 100%;
            max-height: 300px;
            margin-top: 15px;
            display: none;
        }
        .top-k-select {
            margin-bottom: 20px;
        }
        .top-k-select label {
            margin-right: 10px;
            font-size: 14px;
            color: #666;
        }
        #top_k {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .btn {
            padding: 12px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .results {
            margin-top: 30px;
        }
        .result-item {
            display: flex;
            gap: 20px;
            padding: 20px;
            border: 1px solid #eee;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: #fafafa;
        }
        .result-image {
            width: 200px;
            height: 200px;
            object-fit: cover;
            border-radius: 5px;
        }
        .result-info {
            flex: 1;
        }
        .similarity {
            display: inline-block;
            padding: 5px 10px;
            background-color: #2ecc71;
            color: white;
            border-radius: 20px;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .description {
            color: #333;
            line-height: 1.6;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading img {
            width: 50px;
            height: 50px;
        }
        /* Markdown 渲染样式 */
        .markdown-content {
            line-height: 1.6;
        }
        .markdown-content p {
            margin: 0 0 10px 0;
        }
        .markdown-content ul, .markdown-content ol {
            margin: 0 0 10px 20px;
        }
        .markdown-content li {
            margin-bottom: 5px;
        }
        .markdown-content strong {
            font-weight: bold;
        }
        .markdown-content em {
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <a href="/">检索首页</a>
        <a href="/upload">图片上传</a>
        <a href="/database">向量库查看</a>
    </div>
    <div class="container">
        <h1>多模态图像检索系统</h1>
        
        <div class="search-tabs">
            <button class="tab active" data-tab="text">文本检索</button>
            <button class="tab" data-tab="image">图片检索</button>
        </div>
        
        <div class="search-panels">
            <div class="search-panel active" id="text-panel">
                <div class="text-search">
                    <input type="text" id="query" placeholder="输入描述文本进行检索...">
                    <button class="btn" id="text-search-btn">检索</button>
                </div>
                <div class="top-k-select">
                    <label for="top_k">返回结果数量:</label>
                    <select id="top_k">
                        <option value="3">3</option>
                        <option value="5" selected>5</option>
                        <option value="10">10</option>
                        <option value="20">20</option>
                    </select>
                </div>
            </div>
            
            <div class="search-panel" id="image-panel">
                <div class="image-upload" id="image-upload">
                    <p>点击或拖拽图片到此处上传</p>
                    <input type="file" id="image-input" accept="image/*" style="display: none;">
                    <img id="image-preview" src="" alt="预览图">
                </div>
                <div class="top-k-select">
                    <label for="image-top_k">返回结果数量:</label>
                    <select id="image-top_k">
                        <option value="3">3</option>
                        <option value="5" selected>5</option>
                        <option value="10">10</option>
                        <option value="20">20</option>
                    </select>
                </div>
                <button class="btn" id="image-search-btn">检索</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <img src="data:image/gif;base64,R0lGODlhEAAQAPIAAP///wAAAMLCwkJCQgAAAGJiYoKCgpKSkiH/C05FVFNDQVBFMi4wAwEAAAAh/hpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh+QQJCgAAACwAAAAAEAAQAAADMwi63P4wyklrE2MIOggZnAdOmGYJRbExwroUmcG2LmDEwnHQLVsYOd2mBzkYDAdKa+dIAAAh+QQJCgAAACwAAAAAEAAQAAADNAi63P5OjCEgG4QMu7DmikRxQlFUYDEZIGBMRVsaqHwctXXf7WEYB4Ag1xjihkMZsiUkKhIAIfkECQoAAAAsAAAAABAAEAAAAzYIujIjK8pByJDMlFYvBoVjHA70GU7xSUJhmKtwHPAKzLO9HMaoKwJZ7Rf8AYPDDzKpZBqfvwQAIfkECQoAAAAsAAAAABAAEAAAAzMIumIlK8oyhpHsnFZfhYumCYUhDAQxRIdhHBGqRoKw0R8DYlJd8z0fMDgsGo/IpHI5TAAAIfkECQoAAAAsAAAAABAAEAAAAzIIunInK0rnZBTwGPNMgQwmdsNgXGJUlIWEuR5oWUIpz8pAEAMe6TwfwyYsGo/IpFKSAAAh+QQJCgAAACwAAAAAEAAQAAADMwi6IMKQORfjdOe82p4wGccc4CEuQradylesojEMBgsUc2G7sDX3lQGBMLAJibufbSlKAAAh+QQJCgAAACwAAAAAEAAQAAADMgi63P7wCRHZnFVdmgHu2nFwlWCI3WGc3TSWhUFGxTAUkGCbtgENBMJAEJsxgMLWzpEAACH5BAkKAAAALAAAAAAQABAAAAeJHJJ8tbr5ilCWqHBgRMkkGRwmVvWAqLCjJAoWQGAJBOkSw+AbMMBAUMA0b2wdIAIfkECQoAAAAsAAAAABAAEAAAAzIgi63P5tQZ7NgIXglcDYB匀hJpcRlD7JHSAK7GxwIthJqZ2IhYA7AAAJ3gAAAABAAAJAAACbLzWettdbIAOwAAAAAAAAAAAA==" alt="加载中...">
        </div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        // 切换检索面板
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.search-panel').forEach(p => p.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.getAttribute('data-tab') + '-panel').classList.add('active');
            });
        });
        
        // 图片上传预览
        const imageUpload = document.getElementById('image-upload');
        const imageInput = document.getElementById('image-input');
        const imagePreview = document.getElementById('image-preview');
        
        imageUpload.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', (e) => {
            if (e.target.files?.[0]) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(e.target.files[0]);
            }
        });
        
        // 拖拽上传
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            imageUpload.addEventListener(eventName, e => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            imageUpload.addEventListener(eventName, () => {
                imageUpload.style.borderColor = '#3498db';
                imageUpload.style.backgroundColor = 'rgba(52, 152, 219, 0.1)';
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            imageUpload.addEventListener(eventName, () => {
                imageUpload.style.borderColor = '#ddd';
                imageUpload.style.backgroundColor = 'transparent';
            }, false);
        });
        
        imageUpload.addEventListener('drop', (e) => {
            const file = e.dataTransfer.files[0];
            if (file?.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    imageInput.files = dataTransfer.files;
                };
                reader.readAsDataURL(file);
            }
        }, false);
        
        // 文本检索
        document.getElementById('text-search-btn').addEventListener('click', () => {
            const query = document.getElementById('query').value.trim();
            const topK = document.getElementById('top_k').value;
            if (query) searchByText(query, topK);
            else alert('请输入检索关键词');
        });
        
        document.getElementById('query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const query = document.getElementById('query').value.trim();
                const topK = document.getElementById('top_k').value;
                if (query) searchByText(query, topK);
            }
        });
        
        // 图片检索
        document.getElementById('image-search-btn').addEventListener('click', () => {
            const topK = document.getElementById('image-top_k').value;
            if (imageInput.files?.length) searchByImage(topK);
            else alert('请上传一张图片');
        });
        
        // 检索函数
        function searchByText(query, topK) {
            showLoading();
            clearResults();
            
            const formData = new FormData();
            formData.append('query', query);
            formData.append('top_k', topK);
            
            fetch('/search/text', {method: 'POST', body: formData})
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    data.error ? showError(data.error) : displayResults(data.results);
                })
                .catch(e => {
                    hideLoading();
                    showError('检索失败: ' + e.message);
                });
        }
        
        function searchByImage(topK) {
            showLoading();
            clearResults();
            
            const formData = new FormData();
            formData.append('image', imageInput.files[0]);
            formData.append('top_k', topK);
            
            fetch('/search/image', {method: 'POST', body: formData})
                .then(r => r.json())
                .then(data => {
                    hideLoading();
                    data.error ? showError(data.error) : displayResults(data.results);
                })
                .catch(e => {
                    hideLoading();
                    showError('检索失败: ' + e.message);
                });
        }
        
        // 结果展示
        function displayResults(results) {
            const container = document.getElementById('results');
            if (!results.length) {
                container.innerHTML = '<p>没有找到匹配的图片</p>';
                return;
            }
            
            container.innerHTML = results.map((r, i) => `
                <div class="result-item">
                    <img src="/image/${encodeURIComponent(r.image_path)}" class="result-image" alt="相似图片 ${i+1}">
                    <div class="result-info">
                        <span class="similarity">相似度: ${(r.similarity*100).toFixed(2)}%</span>
                        <div><strong>描述:</strong></div>
                        <div class="markdown-content">${renderMarkdown(r.description)}</div>
                        <p><strong>路径:</strong> ${r.image_path}</p>
                    </div>
                </div>
            `).join('');
        }
        
        // Markdown 渲染函数
        function renderMarkdown(text) {
            // 简单的Markdown渲染实现
            // 处理粗体 **text**
            text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // 处理斜体 *text*
            text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
            // 处理无序列表项
            text = text.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
            text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
            // 处理段落
            text = text.replace(/^(.+)$/gm, '<p>$1</p>');
            // 移除额外的段落标签包裹列表
            text = text.replace(/<p>(<ul>.*<\/ul>)<\/p>/g, '$1');
            return text;
        }
        
        // 工具函数
        function showLoading() { document.getElementById('loading').style.display = 'block'; }
        function hideLoading() { document.getElementById('loading').style.display = 'none'; }
        function clearResults() { document.getElementById('results').innerHTML = ''; }
        function showError(msg) { document.getElementById('results').innerHTML = `<p style="color: red;">${msg}</p>`; }
    </script>
</body>
</html>
'''
    # 创建上传页面（新增）
    upload_html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图片上传 - 多模态图像检索系统</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .navbar {
            margin-bottom: 20px;
            padding: 10px 0;
            border-bottom: 1px solid #ddd;
        }
        .navbar a {
            margin-right: 20px;
            text-decoration: none;
            color: #3498db;
            font-weight: bold;
        }
        .navbar a:hover {
            color: #2980b9;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #ddd;
            padding: 60px 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #3498db;
            background-color: rgba(52, 152, 219, 0.05);
        }
        .upload-area p {
            color: #666;
            margin: 10px 0;
        }
        #file-input {
            display: none;
        }
        .preview-container {
            margin: 30px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            min-height: 100px;
        }
        .preview-item {
            width: 150px;
            height: 150px;
            position: relative;
        }
        .preview-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 5px;
            border: 1px solid #eee;
        }
        .remove-btn {
            position: absolute;
            top: -5px;
            right: -5px;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .btn {
            padding: 12px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .results {
            margin-top: 30px;
        }
        .result-item {
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .result-item.success {
            background-color: #eafaf1;
            border: 1px solid #c8e6c9;
        }
        .result-item.error {
            background-color: #fceae8;
            border: 1px solid #ffcdd2;
        }
        .result-item .filename {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading img {
            width: 50px;
            height: 50px;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <a href="/">检索首页</a>
        <a href="/upload">图片上传</a>
        <a href="/database">向量库查看</a>
    </div>
    <div class="container">
        <h1>图片上传</h1>
        
        <div class="upload-area" id="upload-area">
            <h3>点击或拖拽图片到此处</h3>
            <p>支持单张或多张图片（png, jpg, jpeg, gif）</p>
            <p>上传后将自动生成描述并添加到检索库</p>
            <input type="file" id="file-input" accept="image/*" multiple>
        </div>
        
        <div class="preview-container" id="preview-container">
            <!-- 预览图将在这里显示 -->
        </div>
        
        <button class="btn" id="upload-btn">开始上传并处理</button>
        
        <div class="loading" id="loading">
            <img src="data:image/gif;base64,R0lGODlhEAAQAPIAAP///wAAAMLCwkJCQgAAAGJiYoKCgpKSkiH/C05FVFNDQVBFMi4wAwEAAAAh/hpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh+QQJCgAAACwAAAAAEAAQAAADMwi63P4wyklrE2MIOggZnAdOmGYJRbExwroUmcG2LmDEwnHQLVsYOd2mBzkYDAdKa+dIAAAh+QQJCgAAACwAAAAAEAAQAAADNAi63P5OjCEgG4QMu7DmikRxQlFUYDEZIGBMRVsaqHwctXXf7WEYB4Ag1xjihkMZsiUkKhIAIfkECQoAAAAsAAAAABAAEAAAAzYIujIjK8pByJDMlFYvBoVjHA70GU7xSUJhmKtwHPAKzLO9HMaoKwJZ7Rf8AYPDDzKpZBqfvwQAIfkECQoAAAAsAAAAABAAEAAAAzMIumIlK8oyhpHsnFZfhYumCYUhDAQxRIdhHBGqRoKw0R8DYlJd8z0fMDgsGo/IpHI5TAAAIfkECQoAAAAsAAAAABAAEAAAAzIIunInK0rnZBTwGPNMgQwmdsNgXGJUlIWEuR5oWUIpz8pAEAMe6TwfwyYsGo/IpFKSAAAh+QQJCgAAACwAAAAAEAAQAAADMwi6IMKQORfjdOe82p4wGccc4CEuQradylesojEMBgsUc2G7sDX3lQGBMLAJibufbSlKAAAh+QQJCgAAACwAAAAAEAAQAAADMgi63P7wCRHZnFVdmgHu2nFwlWCI3WGc3TSWhUFGxTAUkGCbtgENBMJAEJsxgMLWzpEAACH5BAkKAAAALAAAAAAQABAAAAeJHJJ8tbr5ilCWqHBgRMkkGRwmVvWAqLCjJAoWQGAJBOkSw+AbMMBAUMA0b2wdIAIfkECQoAAAAsAAAAABAAEAAAAzIgi63P5tQZ7NgIXglcDYB匀hJpcRlD7JHSAK7GxwIthJqZ2IhYA7AAAJ3gAAAABAAAJAAACbLzWettdbIAOwAAAAAAAAAAAA==" alt="处理中...">
        </div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const previewContainer = document.getElementById('preview-container');
        const uploadBtn = document.getElementById('upload-btn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        // 存储选中的文件
        let selectedFiles = [];
        
        // 点击上传区域触发文件选择
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // 处理文件选择
        fileInput.addEventListener('change', (e) => {
            const files = e.target.files;
            if (files.length) {
                addFilesToPreview(files);
            }
        });
        
        // 拖拽功能
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadArea.style.borderColor = '#3498db';
            uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.1)';
        }
        
        function unhighlight() {
            uploadArea.style.borderColor = '#ddd';
            uploadArea.style.backgroundColor = 'transparent';
        }
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length) {
                addFilesToPreview(files);
            }
        }
        
        // 添加文件到预览区
        function addFilesToPreview(files) {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                
                // 检查是否为图片
                if (!file.type.startsWith('image/')) {
                    continue;
                }
                
                // 检查是否已添加
                const isDuplicate = selectedFiles.some(f => 
                    f.name === file.name && f.size === file.size && f.lastModified === file.lastModified
                );
                
                if (!isDuplicate) {
                    selectedFiles.push(file);
                    createPreviewItem(file);
                }
            }
        }
        
        // 创建预览项
        function createPreviewItem(file) {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            previewItem.dataset.filename = file.name;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                previewItem.innerHTML = `
                    <img src="${e.target.result}" class="preview-img" alt="${file.name}">
                    <button class="remove-btn" data-filename="${file.name}">×</button>
                `;
                
                // 添加删除事件
                previewItem.querySelector('.remove-btn').addEventListener('click', (e) => {
                    const filename = e.target.dataset.filename;
                    // 从数组中移除
                    selectedFiles = selectedFiles.filter(f => !(f.name === filename && f.size === file.size));
                    // 从DOM中移除
                    previewItem.remove();
                });
                
                previewContainer.appendChild(previewItem);
            };
            
            reader.readAsDataURL(file);
        }
        
        // 上传处理
        uploadBtn.addEventListener('click', () => {
            if (selectedFiles.length === 0) {
                alert('请选择至少一张图片');
                return;
            }
            
            // 显示加载状态
            loading.style.display = 'block';
            results.innerHTML = '';
            
            // 创建FormData
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('images', file);
            });
            
            // 发送请求
            fetch('/upload/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.error) {
                    results.innerHTML = `<div class="result-item error"><div class="filename">错误</div><div class="message">${data.error}</div></div>`;
                } else if (data.results) {
                    let html = '';
                    data.results.forEach(result => {
                        html += `
                        <div class="result-item ${result.status}">
                            <div class="filename">${result.filename}</div>
                            <div class="message">${result.message}</div>
                            ${result.save_path ? `<div class="path">保存路径: ${result.save_path}</div>` : ''}
                        </div>
                        `;
                    });
                    results.innerHTML = html;
                    
                    // 清空选择
                    selectedFiles = [];
                    previewContainer.innerHTML = '';
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                results.innerHTML = `<div class="result-item error"><div class="filename">请求失败</div><div class="message">${error.message}</div></div>`;
            });
        });
    </script>
</body>
</html>
'''
    
    # 创建向量库查看页面
    database_html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>向量库查看 - 多模态图像检索系统</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .navbar {
            margin-bottom: 20px;
            padding: 10px 0;
            border-bottom: 1px solid #ddd;
        }
        .navbar a {
            margin-right: 20px;
            text-decoration: none;
            color: #3498db;
            font-weight: bold;
        }
        .navbar a:hover {
            color: #2980b9;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .database-info {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
        }
        .info-card {
            flex: 1;
            min-width: 200px;
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .info-card h3 {
            margin-top: 0;
            color: #333;
        }
        .info-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .search-box {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }
        #filename-search {
            flex: 1;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .btn {
            padding: 12px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .items-container {
            margin-top: 20px;
        }
        .item-card {
            display: flex;
            gap: 20px;
            padding: 20px;
            border: 1px solid #eee;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: #fafafa;
        }
        .item-image {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 5px;
        }
        .item-info {
            flex: 1;
        }
        .item-info h3 {
            margin-top: 0;
            color: #333;
        }
        .item-id {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
        }
        .description {
            color: #333;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading img {
            width: 50px;
            height: 50px;
        }
        .pagination {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        .pagination button {
            padding: 8px 12px;
            border: 1px solid #ddd;
            background-color: white;
            cursor: pointer;
            border-radius: 3px;
        }
        .pagination button.active {
            background-color: #3498db;
            color: white;
        }
        .pagination button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        /* Markdown 渲染样式 */
        .markdown-content {
            line-height: 1.6;
        }
        .markdown-content p {
            margin: 0 0 10px 0;
        }
        .markdown-content ul, .markdown-content ol {
            margin: 0 0 10px 20px;
        }
        .markdown-content li {
            margin-bottom: 5px;
        }
        .markdown-content strong {
            font-weight: bold;
        }
        .markdown-content em {
            font-style: italic;
        }
        .vector-info {
            background-color: #e8f4f8;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 12px;
        }
        .vector-info summary {
            cursor: pointer;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <a href="/">检索首页</a>
        <a href="/upload">图片上传</a>
        <a href="/database">向量库查看</a>
    </div>
    <div class="container">
        <h1>向量库查看</h1>
        
        <div class="database-info" id="database-info">
            <div class="info-card">
                <h3>图像特征数量</h3>
                <div class="value" id="image-count">-</div>
            </div>
            <div class="info-card">
                <h3>文本特征数量</h3>
                <div class="value" id="text-count">-</div>
            </div>
            <div class="info-card">
                <h3>总特征数量</h3>
                <div class="value" id="total-count">-</div>
            </div>
            <div class="info-card">
                <h3>数据库大小</h3>
                <div class="value" id="db-size">-</div>
            </div>
            <div class="info-card">
                <h3>向量库类型</h3>
                <div class="value" id="db-type">-</div>
            </div>
        </div>
        
        <div class="search-box">
            <input type="text" id="filename-search" placeholder="输入文件名搜索...">
            <button class="btn" id="search-btn">搜索</button>
            <button class="btn" id="reset-btn">重置</button>
        </div>
        
        <div class="loading" id="loading">
            <img src="data:image/gif;base64,R0lGODlhEAAQAPIAAP///wAAAMLCwkJCQgAAAGJiYoKCgpKSkiH/C05FVFNDQVBFMi4wAwEAAAAh/hpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh+QQJCgAAACwAAAAAEAAQAAADMwi63P4wyklrE2MIOggZnAdOmGYJRbExwroUmcG2LmDEwnHQLVsYOd2mBzkYDAdKa+dIAAAh+QQJCgAAACwAAAAAEAAQAAADNAi63P5OjCEgG4QMu7DmikRxQlFUYDEZIGBMRVsaqHwctXXf7WEYB4Ag1xjihkMZsiUkKhIAIfkECQoAAAAsAAAAABAAEAAAAzYIujIjK8pByJDMlFYvBoVjHA70GU7xSUJhmKtwHPAKzLO9HMaoKwJZ7Rf8AYPDDzKpZBqfvwQAIfkECQoAAAAsAAAAABAAEAAAAzMIumIlK8oyhpHsnFZfhYumCYUhDAQxRIdhHBGqRoKw0R8DYlJd8z0fMDgsGo/IpHI5TAAAIfkECQoAAAAsAAAAABAAEAAAAzIIunInK0rnZBTwGPNMgQwmdsNgXGJUlIWEuR5oWUIpz8pAEAMe6TwfwyYsGo/IpFKSAAAh+QQJCgAAACwAAAAAEAAQAAADMwi6IMKQORfjdOe82p4wGccc4CEuQradylesojEMBgsUc2G7sDX3lQGBMLAJibufbSlKAAAh+QQJCgAAACwAAAAAEAAQAAADMgi63P7wCRHZnFVdmgHu2nFwlWCI3WGc3TSWhUFGxTAUkGCbtgENBMJAEJsxgMLWzpEAACH5BAkKAAAALAAAAAAQABAAAAeJHJJ8tbr5ilCWqHBgRMkkGRwmVvWAqLCjJAoWQGAJBOkSw+AbMMBAUMA0b2wdIAIfkECQoAAAAsAAAAABAAEAAAAzIgi63P5tQZ7NgIXglcDYB匀hJpcRlD7JHSAK7GxwIthJqZ2IhYA7AAAJ3gAAAABAAAJAAACbLzWettdbIAOwAAAAAAAAAAAA==" alt="加载中...">
        </div>
        
        <div class="items-container" id="items-container">
            <!-- 项目列表将在这里显示 -->
        </div>
        
        <div class="pagination" id="pagination">
            <!-- 分页控件将在这里显示 -->
        </div>
    </div>

    <script>
        let currentPage = 0;
        const itemsPerPage = 10;
        
        // 页面加载时获取数据库信息
        document.addEventListener('DOMContentLoaded', () => {
            loadDatabaseInfo();
            loadItems();
        });
        
        // 加载数据库信息
        function loadDatabaseInfo() {
            fetch('/database/info')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('image-count').textContent = data.image_count;
                    document.getElementById('text-count').textContent = data.text_count;
                    document.getElementById('total-count').textContent = data.total_count;
                    document.getElementById('db-size').textContent = data.db_size;
                    document.getElementById('db-type').textContent = data.db_type;
                })
                .catch(error => {
                    console.error('加载数据库信息失败:', error);
                });
        }
        
        // 加载项目列表
        function loadItems(offset = 0) {
            showLoading();
            fetch(`/database/items?offset=${offset}&limit=${itemsPerPage}`)
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    displayItems(data);
                    updatePagination(offset);
                })
                .catch(error => {
                    hideLoading();
                    console.error('加载项目列表失败:', error);
                });
        }
        
        // 搜索项目
        function searchItems(filename) {
            if (!filename) {
                loadItems();
                return;
            }
            
            showLoading();
            fetch(`/database/search?filename=${encodeURIComponent(filename)}`)
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    displayItems(data);
                    document.getElementById('pagination').innerHTML = '';
                })
                .catch(error => {
                    hideLoading();
                    console.error('搜索项目失败:', error);
                });
        }
        
        // 显示项目列表
        function displayItems(items) {
            const container = document.getElementById('items-container');
            
            if (!items.length) {
                container.innerHTML = '<div class="no-results">没有找到匹配的项目</div>';
                return;
            }
            
            container.innerHTML = items.map(item => `
                <div class="item-card">
                    <img src="/image/${encodeURIComponent(item.image_path)}" class="item-image" alt="${getFilenameFromPath(item.image_path)}">
                    <div class="item-info">
                        <h3>${getFilenameFromPath(item.image_path)}</h3>
                        <div class="item-id">ID: ${item.id}</div>
                        <div><strong>图像描述:</strong></div>
                        <div class="markdown-content">${renderMarkdown(item.description || '无')}</div>
                        <div><strong>文本描述:</strong></div>
                        <div class="markdown-content">${renderMarkdown(item.text_description || '无')}</div>
                        <details class="vector-info">
                            <summary>查看向量信息</summary>
                            <p><strong>图像路径:</strong> ${item.image_path}</p>
                        </details>
                    </div>
                </div>
            `).join('');
        }
        
        // 从路径中提取文件名
        function getFilenameFromPath(path) {
            return path.split(/[\\\\/]/).pop();
        }
        
        // Markdown 渲染函数
        function renderMarkdown(text) {
            if (!text || text === '无') return text;
            
            // 简单的Markdown渲染实现
            // 处理粗体 **text**
            text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // 处理斜体 *text*
            text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
            // 处理无序列表项
            text = text.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
            text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
            // 处理段落
            text = text.replace(/^(.+)$/gm, '<p>$1</p>');
            // 移除额外的段落标签包裹列表
            text = text.replace(/<p>(<ul>.*<\/ul>)<\/p>/g, '$1');
            return text;
        }
        
        // 更新分页控件
        function updatePagination(offset) {
            const pagination = document.getElementById('pagination');
            const prevDisabled = offset === 0;
            const nextDisabled = false; // 简化处理，实际应根据是否有更多数据判断
            
            pagination.innerHTML = `
                <button id="prev-page" ${prevDisabled ? 'disabled' : ''}>上一页</button>
                <span>第 ${Math.floor(offset/itemsPerPage) + 1} 页</span>
                <button id="next-page" ${nextDisabled ? 'disabled' : ''}>下一页</button>
            `;
            
            document.getElementById('prev-page').addEventListener('click', () => {
                if (offset >= itemsPerPage) {
                    loadItems(offset - itemsPerPage);
                }
            });
            
            document.getElementById('next-page').addEventListener('click', () => {
                loadItems(offset + itemsPerPage);
            });
        }
        
        // 工具函数
        function showLoading() { document.getElementById('loading').style.display = 'block'; }
        function hideLoading() { document.getElementById('loading').style.display = 'none'; }
        
        // 搜索按钮事件
        document.getElementById('search-btn').addEventListener('click', () => {
            const filename = document.getElementById('filename-search').value.trim();
            searchItems(filename);
        });
        
        // 重置按钮事件
        document.getElementById('reset-btn').addEventListener('click', () => {
            document.getElementById('filename-search').value = '';
            loadItems();
        });
        
        // 回车搜索
        document.getElementById('filename-search').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const filename = document.getElementById('filename-search').value.trim();
                searchItems(filename);
            }
        });
    </script>
'''
    
    # 创建模板目录并写入文件
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    with open('templates/upload.html', 'w', encoding='utf-8') as f:
        f.write(upload_html)
    with open('templates/database.html', 'w', encoding='utf-8') as f:
        f.write(database_html)

@app.route('/')
def home():
    """渲染主页"""
    return render_template('index.html')

if __name__ == "__main__":
    # 不再创建前端模板，而是使用已创建的 templates/index.html
    print("启动多模态图像检索系统，访问 http://127.0.0.1:5000 进行使用")
    app.run(debug=False, host='0.0.0.0', port=5000)
