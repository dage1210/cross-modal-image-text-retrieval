多模态图像检索系统

简介：
这是一个基于CLIP模型和Chroma向量数据库的多模态图像检索系统。系统支持文本到图像和图像到图像的检索功能，
并提供Web界面进行操作。用户可以通过文本描述搜索相关图像，也可以通过上传图像来查找相似的图像。

功能特性：
1. 文本到图像检索：输入文本描述，系统会返回与描述最匹配的图像
2. 图像到图像检索：上传图像，系统会返回与之最相似的图像
3. 图像上传和管理：支持上传新图像并自动处理，添加到检索库中
4. 向量库查看：可以浏览所有已处理的图像及其描述信息
5. 基于CLIP的特征提取：使用OpenAI的CLIP模型提取图像特征
6. 中文图像描述生成：使用qwen3-vl模型生成图像的中文描述
7. 向量存储：使用Chroma向量数据库存储和检索图像及文本特征

目录结构：
- visual_search_doubao_101.py: 主程序文件，包含Flask Web服务和检索逻辑
- img/: 初始图像目录
- uploaded_img/: 上传图像保存目录
- chroma_multimodal_db/: Chroma向量数据库目录
- templates/: Web前端模板目录
- image_descriptions.json: 图像描述数据文件

系统要求：
- Python 3.11
- PyTorch
- Flask
- ChromaDB
- Transformers
- Pillow
- Numpy
- Requests
- Tqdm

安装依赖：
pip install -r requirements.txt

配置说明：
在visual_search_doubao_101.py文件中，可以修改以下配置：
- IMAGE_DIR: 初始图像目录路径
- UPLOAD_IMAGE_DIR: 上传图像保存目录路径
- CHROMA_PATH: Chroma数据库路径
- OLLAMA_URL: Ollama服务地址
- OLLAMA_VL_MODEL: 用于图像描述的VL模型名称
- OLLAMA_EMBEDDING_MODEL: 用于文本嵌入的模型名称

运行方法：
1. 确保Ollama服务正在运行，并且已下载qwen3-vl和qwen3-embedding模型
2. 安装所需依赖
3. 运行主程序：python visual_search_doubao_101.py
4. 在浏览器中访问 http://localhost:5000

Web界面说明：
1. 首页（/）：提供文本和图像检索功能
   - 文本检索：在输入框中输入描述文本，点击"检索"按钮
   - 图像检索：上传图像文件，点击"检索"按钮

2. 图像上传页（/upload）：上传新图像到系统
   - 点击或拖拽图像文件到上传区域
   - 点击"开始上传并处理"按钮

3. 向量库查看页（/database）：浏览所有已处理的图像
   - 显示数据库统计信息
   - 列出所有图像及其描述
   - 支持按文件名搜索

API接口：
- POST /search/text: 文本检索接口
  参数：query (文本查询), top_k (返回结果数量)
  
- POST /search/image: 图像检索接口
  参数：image (上传的图像文件), top_k (返回结果数量)
  
- POST /upload/process: 处理上传的图像
  参数：images (上传的图像文件列表)
  
- GET /database/info: 获取数据库信息
  
- GET /database/items: 获取数据库中的项目列表
  参数：offset (偏移量), limit (数量限制)
  
- GET /database/search: 按文件名搜索项目
  参数：filename (文件名)

注意事项：
1. 首次运行时，系统会处理IMAGE_DIR目录中的所有图像
2. 图像处理过程可能需要一些时间，取决于图像数量和系统性能
3. 确保Ollama服务地址配置正确且服务正在运行
4. 图像文件支持格式：png, jpg, jpeg, gif
