import asyncio
import uuid
import os
import logging
from typing import Dict, List, Optional, Set
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import json
import dotenv
# 清空环境变量
os.environ.clear()
# 设置环境变量
dotenv.load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log")
    ]
)
logger = logging.getLogger("mini_rag")

# 配置项
MIN_CHARS_FOR_TTS = 20  # 触发TTS的最小字符数
AUDIO_FOLDER = "audio"  # 音频文件存储目录
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 确保音频文件夹和静态文件夹存在
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI()

# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局状态管理
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.llm_queues: Dict[str, asyncio.Queue] = {}
        self.audio_queues: Dict[str, asyncio.Queue] = {}
        self.active_tasks: Set[str] = set()
        
    def create_task(self, question: str) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "question": question,
            "status": "processing",
            "created_at": time.time(),
            "text_generated": "",
            "audio_files": []
        }
        self.llm_queues[task_id] = asyncio.Queue()
        self.audio_queues[task_id] = asyncio.Queue()
        self.active_tasks.add(task_id)
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: str):
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            
    def add_audio_file(self, task_id: str, audio_file: str):
        if task_id in self.tasks:
            self.tasks[task_id]["audio_files"].append(audio_file)
            
    def cancel_task(self, task_id: str) -> bool:
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
            self.update_task_status(task_id, "cancelled")
            return True
        return False

# 初始化任务管理器
task_manager = TaskManager()

# 请求模型
class GenerateRequest(BaseModel):
    question: str

class TaskIdRequest(BaseModel):
    task_id: str

# 模拟TTS服务
async def text_to_speech(text: str, task_id: str, segment_id: int) -> str:
    """
    将文本转换为语音（使用Silicon Flow API）
    """
    # 记录TTS输入文本
    logger.info(f"TTS Request [Task {task_id}] [Segment {segment_id}]: {text}")
    
    # 创建音频文件名并保存
    filename = f"{AUDIO_FOLDER}/{task_id}_{segment_id}.wav"
    
    try:
        # 配置API调用
        url = os.getenv("TTS_API_BASE") + "/audio/speech"
        
        # 准备请求参数
        payload = {
            "model": os.getenv("TTS_API_MODEL"),
            "input": text,
            "voice": os.getenv("TTS_API_VOICE"),
            "response_format": "wav",
            "sample_rate": 16000,  # 标准采样率
            "stream": False,       # 整个文件返回而不是流式
            "speed": 1,
            "gain": 0
        }
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {os.getenv('TTS_API_KEY', '')}",
            "Content-Type": "application/json"
        }
        
        # 发送API请求
        logger.info(f"Calling TTS API for task {task_id}, segment {segment_id}")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            
            # 检查响应状态
            if response.status_code == 200:
                # 将二进制响应保存到文件
                with open(filename, 'wb') as f:
                    f.write(response.content)
                logger.info(f"TTS API success [Task {task_id}] [Segment {segment_id}]: Generated {filename}")
            else:
                # 如果API调用失败，记录错误并回退到模拟模式
                logger.error(f"TTS API error [Task {task_id}]: {response.status_code} - {response.text}")
                with open(filename, 'w') as f:
                    f.write(f"TTS AUDIO CONTENT FOR: {text}")
        
    except Exception as e:
        # 出现异常时，回退到模拟模式
        logger.exception(f"TTS processing error [Task {task_id}]: {str(e)}")
        with open(filename, 'w') as f:
            f.write(f"TTS AUDIO CONTENT FOR: {text}")
    
    return filename

# 处理LLM生成和TTS转换的后台任务
async def process_llm_response(task_id: str):
    """处理来自LLM的响应并管理TTS转换流程"""
    
    if task_id not in task_manager.tasks:
        logger.warning(f"Task {task_id} not found in task manager")
        return
    
    question = task_manager.tasks[task_id]["question"]
    llm_queue = task_manager.llm_queues[task_id]
    audio_queue = task_manager.audio_queues[task_id]
    
    logger.info(f"Starting LLM processing for task {task_id}")
    logger.info(f"Question: {question}")
    
    try:
        async with httpx.AsyncClient(base_url=OPENAI_BASE_URL, headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}) as client:
            payload = {
                "model": os.getenv("OPENAI_MODEL"),
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                "stream": True
            }
            
            # 调用OpenAI API获取流式响应
            logger.info(f"Calling OpenAI API for task {task_id}")
            # 添加调试信息
            logger.debug(f"Payload: {payload}")
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                buffer = ""
                segment_id = 0
                is_first_segment = True  # 标记是否为第一个片段
                json_buffer = ""  # 用于累积不完整的JSON数据
                
                async for chunk in response.aiter_text():
                    # 确保任务没有被取消
                    if task_id not in task_manager.active_tasks:
                        logger.info(f"Task {task_id} was cancelled, stopping LLM processing")
                        break
                    
                    # 解析每个数据块
                    if chunk.strip():
                        # 处理数据块，可能包含多个data:行或不完整的JSON
                        chunk_lines = chunk.strip().split('\n')
                        
                        for line in chunk_lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            # 处理data:前缀
                            if line.startswith("data: "):
                                line = line[6:]
                            
                            # 跳过[DONE]消息
                            if line == "[DONE]":
                                logger.debug(f"Received [DONE] for task {task_id}")
                                continue
                            
                            # 累积JSON数据
                            json_buffer += line
                            
                            # 尝试解析累积的JSON
                            try:
                                data = json.loads(json_buffer)
                                # 成功解析后重置缓冲区
                                json_buffer = ""
                                
                                # 处理解析后的数据
                                if "choices" in data and len(data["choices"]) > 0:
                                    if "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                                        text = data["choices"][0]["delta"]["content"]
                                        buffer += text
                                        task_manager.tasks[task_id]["text_generated"] += text
                                        
                                        # 记录LLM输出
                                        logger.debug(f"LLM Output [Task {task_id}]: '{text}'")
                                        
                                        # 放入LLM队列供其他函数处理
                                        await llm_queue.put(text)
                                        
                                        # 为第一个片段的特殊处理
                                        if is_first_segment:
                                            # 检查是否包含任何标点符号(逗号、句号、问号、感叹号等)
                                            punctuation_marks = "，。？！；：、,.?!;:"
                                            has_punctuation = any(mark in buffer for mark in punctuation_marks)
                                            
                                            if (len(buffer) >= 5 and has_punctuation) or len(buffer) >= 30:
                                                # 如果包含标点符号且至少有5个字符，或者达到15字符上限
                                                logger.info(f"First segment threshold reached for task {task_id}, sending to TTS: {len(buffer)} chars")
                                                
                                                # 找出第一个标点符号的位置
                                                cut_position = len(buffer)
                                                for mark in punctuation_marks:
                                                    pos = buffer.find(mark)
                                                    if pos > 0 and pos < cut_position:
                                                        cut_position = pos + 1
                                                
                                                # 如果没找到标点或字数不够，强制截断
                                                if cut_position == len(buffer) and len(buffer) < 5:
                                                    continue  # 继续等待更多文本
                                                
                                                # 分割文本
                                                text_to_convert = buffer[:cut_position]
                                                buffer = buffer[cut_position:]
                                                
                                                # 生成音频文件
                                                audio_file = await text_to_speech(text_to_convert, task_id, segment_id)
                                                segment_id += 1
                                                
                                                # 添加到任务信息并放入音频队列
                                                task_manager.add_audio_file(task_id, audio_file)
                                                await audio_queue.put(audio_file)
                                                
                                                # 标记第一个片段已处理
                                                is_first_segment = False
                                                logger.info(f"First segment processed: '{text_to_convert}'")
                                        else:
                                            # 普通片段处理逻辑
                                            # 检查缓冲区是否达到阈值且包含完整句子
                                            if len(buffer) >= MIN_CHARS_FOR_TTS and "。" in buffer:
                                                # 找到最后一个句号
                                                last_period = buffer.rindex("。") + 1
                                                text_to_convert = buffer[:last_period]
                                                buffer = buffer[last_period:]
                                                
                                                logger.info(f"Buffer threshold reached for task {task_id}, sending to TTS: {len(text_to_convert)} chars")
                                                
                                                # 生成音频文件
                                                audio_file = await text_to_speech(text_to_convert, task_id, segment_id)
                                                segment_id += 1
                                                
                                                # 添加到任务信息并放入音频队列
                                                task_manager.add_audio_file(task_id, audio_file)
                                                await audio_queue.put(audio_file)
                            except json.JSONDecodeError:
                                # JSON不完整，继续等待更多数据
                                # 如果累积的数据过长但仍无法解析，可能是格式错误，此时重置缓冲区
                                if len(json_buffer) > 10000:  # 设置合理的上限
                                    logger.warning(f"JSON buffer overflow for task {task_id}, resetting buffer")
                                    json_buffer = ""
                                continue
                
                # 处理剩余文本
                if buffer and task_id in task_manager.active_tasks:
                    logger.info(f"Processing remaining buffer for task {task_id}: {len(buffer)} chars")
                    audio_file = await text_to_speech(buffer, task_id, segment_id)
                    task_manager.add_audio_file(task_id, audio_file)
                    await audio_queue.put(audio_file)
                
                # 标记任务完成
                if task_id in task_manager.active_tasks:
                    logger.info(f"Task {task_id} completed successfully")
                    await audio_queue.put("TTSDONE")
                    task_manager.update_task_status(task_id, "completed")
                    
    except Exception as e:
        # 处理错误
        logger.exception(f"Error processing task {task_id}: {str(e)}")
        task_manager.update_task_status(task_id, "error")
        await audio_queue.put("TTSDONE")
        
    finally:
        # 清理工作
        if task_id in task_manager.active_tasks:
            task_manager.active_tasks.remove(task_id)
            logger.info(f"Removed task {task_id} from active tasks")

# API 端点
@app.get("/")
async def root():
    """返回首页HTML"""
    return FileResponse("index.html")

@app.post("/generate_audio")
async def generate_audio(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    生成音频端点 - 启动LLM处理并返回任务ID
    """
    task_id = task_manager.create_task(request.question)
    logger.info(f"Created new task {task_id} for question: {request.question}")
    
    # 启动后台任务处理LLM响应
    background_tasks.add_task(process_llm_response, task_id)
    
    return JSONResponse({
        "task_id": task_id,
        "status": "processing",
        "message": "Audio generation started"
    })

@app.get("/get_next_audio/{task_id}")
async def get_next_audio(task_id: str):
    """
    获取下一个音频段落
    """
    if task_id not in task_manager.tasks:
        logger.warning(f"Attempt to get audio for non-existent task {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    
    logger.info(f"Request for next audio segment for task {task_id}")
    audio_queue = task_manager.audio_queues[task_id]
    
    try:
        # 等待最多5秒来获取下一个音频文件
        next_audio = await asyncio.wait_for(audio_queue.get(), timeout=5.0)
        
        # 检查是否完成
        if next_audio == "TTSDONE":
            logger.info(f"All audio segments for task {task_id} have been generated")
            return JSONResponse({
                "status": "completed",
                "message": "All audio segments generated"
            })
            
        logger.info(f"Returning audio file for task {task_id}: {next_audio}")
        return FileResponse(
            path=next_audio,
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(next_audio)}"
            }
        )
    except asyncio.TimeoutError:
        # 队列为空但任务可能仍在处理中
        if task_id in task_manager.active_tasks:
            logger.info(f"No audio ready yet for task {task_id}, still processing")
            return JSONResponse({
                "status": "pending",
                "message": "No audio ready yet, try again later"
            })
        else:
            # 任务已完成且队列为空
            logger.info(f"Task {task_id} completed with no more audio segments")
            return JSONResponse({
                "status": "completed",
                "message": "All audio segments generated"
            })
    except Exception as e:
        logger.exception(f"Error getting next audio for task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting next audio: {str(e)}")

@app.get("/status/{task_id}")
async def status(task_id: str):
    """
    获取任务状态
    """
    task = task_manager.get_task(task_id)
    if not task:
        logger.warning(f"Status check for non-existent task {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    
    logger.info(f"Status request for task {task_id}: {task['status']}")
    return {
        "task_id": task_id,
        "status": task["status"],
        "created_at": task["created_at"],
        "audio_segments": len(task["audio_files"]),
        "is_active": task_id in task_manager.active_tasks
    }

@app.post("/cancel_task_by_id")
async def cancel_task_by_id(request: TaskIdRequest):
    """
    取消指定任务
    """
    success = task_manager.cancel_task(request.task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    
    return {"status": "success", "message": f"Task {request.task_id} cancelled"}

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

# 主入口点
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)