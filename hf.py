import os
import asyncio
from huggingface_hub import snapshot_download
from typing import Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer

class AsyncModelLoader:
    """异步模型加载器，支持断点续传和并行初始化[3,6](@ref)"""
    _initialized = False  # 类属性确保单次初始化[1](@ref)

    def __init__(self, model_name: str, local_dir: str = "./models"):
        self.model_name = model_name
        self.local_dir = local_dir
        self.model = None
        self.tokenizer = None

    async def download_model(self) -> None:
        """异步下载模型（支持镜像加速和断点续传）[4,6](@ref)"""
        os.makedirs(self.local_dir, exist_ok=True)
        
        try:
            # 通过事件循环运行阻塞IO操作[6](@ref)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,  # 使用默认线程池
                lambda: self._sync_download()
            )
        except Exception as e:
            raise RuntimeError(f"Download failed: {str(e)}")

    def _sync_download(self) -> None:
        """同步下载实现（供线程池调用）[4](@ref)"""
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        if not os.path.exists(os.path.join(self.local_dir, "config.json")):
            print(f"Downloading {self.model_name}...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )

    async def initialize(self) -> Tuple[AutoModel, AutoTokenizer]:
        """异步初始化模型和分词器（单例模式）[1,9](@ref)"""
        if not self.__class__._initialized:
            await self.download_model()
            
            # 并行加载模型和分词器[2](@ref)
            model_task = asyncio.create_task(self._load_model())
            tokenizer_task = asyncio.create_task(self._load_tokenizer())
            self.model, self.tokenizer = await asyncio.gather(model_task, tokenizer_task)
            
            self.__class__._initialized = True
            print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        return self.model, self.tokenizer

    async def _load_model(self) -> AutoModel:
        """异步加载模型到指定设备[9](@ref)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: AutoModel.from_pretrained(self.local_dir).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )

    async def _load_tokenizer(self) -> AutoTokenizer:
        """异步加载分词器[9](@ref)"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(self.local_dir)
        )


async def initialize_model(model_name: str, local_dir: str = "./models") -> Tuple[AutoModel, AutoTokenizer]:
    """
    异步初始化模型的完整流程（下载+加载）
    
    示例：
    >>> model, tokenizer = await initialize_model("bert-base-uncased")
    """
    loader = AsyncModelLoader(model_name, local_dir)
    return await loader.initialize()