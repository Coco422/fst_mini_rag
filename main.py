import json
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import uuid
import re
import logging
import time
import traceback
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Set
from dotenv import load_dotenv
import os
import httpx

load_dotenv()
# 设置详细日志格式
LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# 创建文件处理器以将日志同时写入文件
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(file_handler)
logger.info("======= 应用启动 =======")
logger.info(f"日志文件路径: {log_file}")

app = FastAPI()
# 配置共享卷的相关路径
SHARED_VOLUME_PATH = os.getenv("SHARED_VOLUME_PATH", "/app/shared")
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio")
AUDIO_PATH = os.path.join(SHARED_VOLUME_PATH, AUDIO_DIR)
AUDIO_URL_PREFIX = os.getenv("AUDIO_URL_PREFIX", "/audio")
# 任务清理配置（单位：秒）
TASK_CLEANUP_DELAY = int(os.getenv("TASK_CLEANUP_DELAY", "300"))  # 默认5分钟后清理
# 最大并发TTS请求数
MAX_CONCURRENT_TTS = int(os.getenv("MAX_CONCURRENT_TTS", "5"))
# TTS音频缓存大小限制
MAX_TTS_CACHE_SIZE = int(os.getenv("MAX_TTS_CACHE_SIZE", "100"))

# 记录环境配置
logger.info(f"环境配置: SHARED_VOLUME_PATH={SHARED_VOLUME_PATH}")
logger.info(f"环境配置: AUDIO_DIR={AUDIO_DIR}")
logger.info(f"环境配置: AUDIO_PATH={AUDIO_PATH}")
logger.info(f"环境配置: AUDIO_URL_PREFIX={AUDIO_URL_PREFIX}")
logger.info(f"环境配置: TASK_CLEANUP_DELAY={TASK_CLEANUP_DELAY}秒")
logger.info(f"环境配置: MAX_CONCURRENT_TTS={MAX_CONCURRENT_TTS}")
logger.info(f"环境配置: MAX_TTS_CACHE_SIZE={MAX_TTS_CACHE_SIZE}")
logger.info(f"环境配置: OpenAI API Base={os.getenv('OPENAI_API_BASE')}")
logger.info(f"环境配置: OpenAI Model={os.getenv('OPENAI_MODEL')}")
logger.info(f"环境配置: TTS API Base={os.getenv('TTS_API_BASE')}")
logger.info(f"环境配置: TTS Model={os.getenv('TTS_API_MODEL')}")

# 确保音频目录存在
os.makedirs(AUDIO_PATH, exist_ok=True)
logger.info(f"确保音频目录存在: {AUDIO_PATH}")

# 挂载静态文件目录
app.mount("/audio", StaticFiles(directory=AUDIO_PATH), name="audio")
logger.info(f"静态文件目录已挂载: /audio -> {AUDIO_PATH}")

# 用于存储任务状态和音频队列
tasks = {}

# TTS缓存，用于存储已转换的文本音频结果
tts_cache = {}

# 流量控制信号量
tts_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)

class TextRequest(BaseModel):
    text: str

class AudioResponse(BaseModel):
    task_id: str
    audio_url: Optional[str] = None
    status: str = "processing"

def calculate_text_hash(text: str) -> str:
    """计算文本的哈希值，用于TTS缓存"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def manage_tts_cache():
    """管理TTS缓存大小，删除最旧的条目"""
    global tts_cache
    if len(tts_cache) > MAX_TTS_CACHE_SIZE:
        # 按访问时间排序，删除最旧的条目
        sorted_cache = sorted(tts_cache.items(), key=lambda x: x[1]['last_access'])
        # 删除超出限制的旧条目
        for key, _ in sorted_cache[:len(tts_cache) - MAX_TTS_CACHE_SIZE]:
            del tts_cache[key]
        logger.info(f"TTS缓存清理完成，当前缓存大小: {len(tts_cache)}")

def split_text_into_segments(text, min_length=20, max_length=100):
    """智能分段文本，优先在句号等标点处断开"""
    if len(text) < min_length:
        return []
        
    segments = []
    # 匹配句末标点
    sentence_boundaries = [m.end() for m in re.finditer(r'[。.!?！？]', text)]
    
    start = 0
    for end in sentence_boundaries:
        # 如果这个句子足够长，就切分出来
        if end - start >= min_length:
            segments.append(text[start:end])
            start = end
        # 如果积累的内容已经很长，即使没到句号也切分
        elif end - start > max_length:
            segments.append(text[start:end])
            start = end
    
    # 保留剩余部分
    remaining = text[start:]
    if remaining:
        if len(remaining) >= min_length:
            segments.append(remaining)
        else:
            # 返回剩余部分作为下一轮的开始
            return segments + [remaining]
            
    return segments

async def call_llm_with_sse(text):
    """获取LLM的SSE响应"""
    start_time = time.time()
    logger.info(f"开始调用LLM生成内容，输入长度: {len(text)} 字符")
    
    baseurl = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    url = f"{baseurl}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": True
    }
    
    total_tokens = 0
    try:
        logger.debug(f"LLM请求URL: {url}, 模型: {model}")
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_detail = await response.read()
                    logger.error(f"LLM请求失败，状态码: {response.status_code}, 错误: {error_detail}")
                    return
                
                logger.info(f"LLM请求成功，开始接收流式响应")
                # 解析每个chunk获取content内容
                async for chunk in response.iter_lines():
                    if not chunk:
                        continue
                    # 去掉"data: "前缀
                    if chunk.startswith(b"data: "):
                        chunk = chunk[6:]
                    # 解析JSON获取content
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data["choices"][0]["delta"].get("content"):
                            content = chunk_data["choices"][0]["delta"]["content"]
                            total_tokens += 1
                            yield content
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析错误: {e}, chunk: {chunk[:100]}")
                        continue
                    except Exception as e:
                        logger.error(f"处理LLM响应时出错: {e}")
                        logger.debug(f"错误详情: {traceback.format_exc()}")
                        continue
    except Exception as e:
        logger.error(f"LLM调用异常: {str(e)}")
        logger.debug(f"错误详情: {traceback.format_exc()}")
        
    duration = time.time() - start_time
    logger.info(f"LLM生成完成，总计: {total_tokens} 个token，耗时: {duration:.2f}秒")

async def text_to_speech(text: str, task_id: str) -> str:
    """调用TTS服务将文本转为语音，保存到共享卷并返回访问路径"""
    # 检查缓存
    text_hash = calculate_text_hash(text)
    if text_hash in tts_cache:
        cache_entry = tts_cache[text_hash]
        # 更新访问时间
        cache_entry['last_access'] = time.time()
        logger.info(f"TTS缓存命中: {text_hash}, URL: {cache_entry['audio_url']}")
        
        # 记录音频文件路径到任务中
        if task_id in tasks and "audio_files" in tasks[task_id]:
            tasks[task_id]["audio_files"].add(cache_entry['file_path'])
            logger.debug(f"已将缓存音频文件 {cache_entry['file_path']} 添加到任务 {task_id} 的文件列表中")
        
        return cache_entry['audio_url']
    
    # 使用信号量控制并发请求数
    async with tts_semaphore:
        start_time = time.time()
        logger.info(f"开始TTS转换，文本长度: {len(text)} 字符, 文本开头: '{text[:30]}...'")
        
        baseurl = os.getenv("TTS_API_BASE")
        api_key = os.getenv("TTS_API_KEY")
        model = os.getenv("TTS_API_MODEL")
        voice = f"{model}:alex"
        
        url = f"{baseurl}/audio/speech" 
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 根据TTS API文档，调整请求参数
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "sample_rate": 16000,  # 标准采样率
            "stream": False,       # 非流式请求
            "speed": 1,            # 正常语速
            "gain": 0              # 默认增益
        }
        
        logger.debug(f"TTS请求URL: {url}, 模型: {model}, 声音: {voice}")
        
        try:
            async with httpx.AsyncClient() as client:
                # 设置超时时间
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                
                if response.status_code == 200:
                    # 生成唯一的文件名
                    filename = f"{uuid.uuid4()}.wav"
                    file_path = os.path.join(AUDIO_PATH, filename)
                    
                    # 获取音频数据大小
                    audio_data = response.content
                    audio_size = len(audio_data)
                    logger.info(f"TTS服务返回音频数据: {audio_size} 字节")
                    
                    # 保存音频数据到文件
                    try:
                        with open(file_path, "wb") as f:
                            f.write(audio_data)
                        logger.info(f"音频文件已保存: {file_path}, 大小: {audio_size} 字节")
                    except Exception as e:
                        logger.error(f"保存音频文件出错: {str(e)}, 路径: {file_path}")
                        logger.debug(f"错误详情: {traceback.format_exc()}")
                        return None
                    
                    # 返回客户端可访问的URL (基于挂载的静态文件路径)
                    audio_url = f"{AUDIO_URL_PREFIX}/{filename}"
                    logger.info(f"生成音频URL: {audio_url}")
                    
                    # 记录音频文件路径到任务中
                    if task_id in tasks and "audio_files" in tasks[task_id]:
                        tasks[task_id]["audio_files"].add(file_path)
                        logger.debug(f"已将音频文件 {file_path} 添加到任务 {task_id} 的文件列表中")
                    
                    # 添加到缓存
                    tts_cache[text_hash] = {
                        'audio_url': audio_url,
                        'file_path': file_path,
                        'last_access': time.time()
                    }
                    # 管理缓存大小
                    manage_tts_cache()
                    
                    duration = time.time() - start_time
                    logger.info(f"TTS转换完成，文本: {len(text)} 字符 -> 音频: {audio_size} 字节, 耗时: {duration:.2f}秒")
                    return audio_url
                else:
                    error_detail = response.text
                    logger.error(f"TTS服务请求失败，状态码: {response.status_code}, 错误信息: {error_detail}")
                    return None
        except httpx.TimeoutException:
            logger.error(f"TTS服务请求超时, URL: {url}")
            return None
        except Exception as e:
            logger.error(f"TTS服务调用异常: {str(e)}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
            return None

async def wait_for_first_audio(task_id, max_iterations=300):  # 30秒(0.1秒 * 300)
    """等待第一段音频生成完成"""
    iterations = 0
    while iterations < max_iterations:
        if task_id not in tasks:
            logger.error(f"任务 {task_id} 不存在")
            return False
            
        if tasks[task_id]["audio_queue"] or tasks[task_id]["completed"]:
            return True
            
        await asyncio.sleep(0.1)
        iterations += 1
        
        if iterations % 50 == 0:  # 每5秒记录一次
            logger.info(f"任务 {task_id}: 等待第一段音频，已等待 {iterations/10:.1f} 秒")
    
    logger.warning(f"任务 {task_id}: 等待第一段音频超时，最大等待时间已到")
    return False

async def cleanup_task(task_id: str):
    """清理任务及其关联的音频文件"""
    logger.info(f"计划任务清理: {task_id}, 延迟: {TASK_CLEANUP_DELAY}秒后执行")
    
    # 等待指定的延迟时间
    await asyncio.sleep(TASK_CLEANUP_DELAY)
    
    logger.info(f"开始执行任务清理: {task_id}")
    
    if task_id in tasks:
        # 获取不在缓存中的音频文件
        audio_files = tasks[task_id].get("audio_files", set())
        files_to_delete = set()
        
        for file_path in audio_files:
            # 检查文件是否被缓存引用
            is_cached = False
            for cache_entry in tts_cache.values():
                if cache_entry['file_path'] == file_path:
                    is_cached = True
                    break
            
            if not is_cached:
                files_to_delete.add(file_path)
        
        logger.info(f"任务 {task_id} 关联的音频文件: {len(audio_files)}个, 将删除: {len(files_to_delete)}个")
        
        deleted_count = 0
        error_count = 0
        
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    logger.info(f"已删除音频文件: {file_path}, 大小: {file_size} 字节")
                    deleted_count += 1
                else:
                    logger.warning(f"音频文件不存在，无需删除: {file_path}")
            except Exception as e:
                logger.error(f"删除音频文件出错: {file_path}, 错误: {str(e)}")
                logger.debug(f"错误详情: {traceback.format_exc()}")
                error_count += 1
        
        # 记录任务数据
        creation_time = tasks[task_id].get("creation_time", 0)
        completion_time = tasks[task_id].get("completion_time", 0)
        task_duration = completion_time - creation_time if completion_time > 0 else 0
        
        # 从任务字典中删除任务数据
        del tasks[task_id]
        logger.info(f"任务数据已清理: {task_id}, 任务运行时长: {task_duration:.2f}秒, 删除文件: {deleted_count}个成功, {error_count}个失败")
    else:
        logger.warning(f"任务 {task_id} 不存在或已被清理")

async def process_llm_response(task_id: str, text: str):
    """处理LLM响应并转换为音频流"""
    logger.info(f"开始处理任务 {task_id}，输入文本长度: {len(text)} 字符")
    start_time = time.time()
    
    buffer = ""
    chunk_count = 0
    audio_count = 0
    
    try:
        async for chunk in call_llm_with_sse(text):
            buffer += chunk
            chunk_count += 1
            
            # 使用智能分段逻辑
            segments = split_text_into_segments(buffer)
            if segments:
                # 保留最后一段不完整的文本
                buffer = segments.pop() if segments else ""
                
                # 并行处理完整段落的TTS转换
                if segments:
                    tts_tasks = [text_to_speech(segment, task_id) for segment in segments]
                    audio_urls = await asyncio.gather(*tts_tasks)
                    
                    for url in audio_urls:
                        if url:
                            tasks[task_id]["audio_queue"].append(url)
                            audio_count += 1
                            logger.info(f"任务 {task_id}: 已生成音频 #{audio_count}, 添加到队列, URL: {url}")
                        else:
                            logger.warning(f"任务 {task_id}: 音频生成失败，跳过一段文本")
        
        # 处理剩余的文本
        if buffer:
            logger.debug(f"任务 {task_id}: 处理最后一段文本 ({len(buffer)} 字符)")
            audio_url = await text_to_speech(buffer, task_id)
            if audio_url:
                tasks[task_id]["audio_queue"].append(audio_url)
                audio_count += 1
                logger.info(f"任务 {task_id}: 已生成最后一段音频 #{audio_count}, 添加到队列, URL: {audio_url}")
            else:
                logger.warning(f"任务 {task_id}: 最后一段音频生成失败，文本: {buffer[:50]}...")
    except Exception as e:
        logger.error(f"任务 {task_id} 处理异常: {str(e)}")
        logger.debug(f"错误详情: {traceback.format_exc()}")
        tasks[task_id]["error"] = str(e)
    finally:
        # 标记任务完成
        tasks[task_id]["completed"] = True
        tasks[task_id]["completion_time"] = time.time()
        duration = tasks[task_id]["completion_time"] - start_time
        
        logger.info(f"任务 {task_id} 处理完成: 处理了 {chunk_count} 个文本块, 生成了 {audio_count} 个音频文件, 耗时: {duration:.2f}秒")
        
        # 在后台启动清理任务
        asyncio.create_task(cleanup_task(task_id))

@app.post("/generate_audio")
async def generate_audio(request: TextRequest, background_tasks: BackgroundTasks):
    """接收文本并开始生成音频流程"""
    start_time = time.time()
    task_id = str(uuid.uuid4())
    
    logger.info(f"收到新请求，创建任务 {task_id}, 输入文本长度: {len(request.text)} 字符")
    logger.debug(f"任务 {task_id} 输入文本: '{request.text[:100]}...'")
    
    tasks[task_id] = {
        "audio_queue": [],
        "audio_files": set(),  # 存储音频文件路径，用于后续清理
        "completed": False,
        "creation_time": start_time
    }
    
    # 后台启动处理任务
    background_tasks.add_task(process_llm_response, task_id, request.text)
    logger.info(f"已启动任务 {task_id} 的后台处理")
    
    # 等待第一段音频生成完毕，设置超时
    try:
        await asyncio.wait_for(
            wait_for_first_audio(task_id),
            timeout=30.0  # 最大等待30秒
        )
    except asyncio.TimeoutError:
        logger.error(f"任务 {task_id}: 等待首段音频生成超时")
    
    # 返回第一段音频和任务ID
    first_audio = tasks[task_id]["audio_queue"].pop(0) if tasks[task_id]["audio_queue"] else None
    is_completed = tasks[task_id]["completed"] and not tasks[task_id]["audio_queue"]
    
    response_data = {
        "task_id": task_id,
        "audio_url": first_audio,
        "status": "completed" if is_completed else "processing"
    }
    
    total_duration = time.time() - start_time
    logger.info(f"任务 {task_id}: 首次响应完成，状态: {response_data['status']}, 音频URL: {first_audio}, 总耗时: {total_duration:.2f}秒")
    
    return JSONResponse(response_data)

@app.get("/get_next_audio/{task_id}")
async def get_next_audio(task_id: str):
    """获取下一段音频"""
    start_time = time.time()
    logger.info(f"收到获取下一段音频请求，任务ID: {task_id}")
    
    if task_id not in tasks:
        logger.warning(f"无效的任务ID: {task_id}")
        return JSONResponse({"error": "Invalid task ID"}, status_code=404)
    
    # 检查队列中是否有音频
    if tasks[task_id]["audio_queue"]:
        next_audio = tasks[task_id]["audio_queue"].pop(0)
        queue_length = len(tasks[task_id]["audio_queue"])
        
        response_data = {
            "task_id": task_id,
            "audio_url": next_audio,
            "status": "processing",
            "queue_length": queue_length  # 返回当前队列长度信息
        }
        
        logger.info(f"任务 {task_id}: 返回下一段音频，URL: {next_audio}, 剩余队列: {queue_length}")
        return JSONResponse(response_data)
    
    # 如果队列为空且任务已完成
    if tasks[task_id]["completed"]:
        logger.info(f"任务 {task_id}: 已完成，无更多音频")
        return JSONResponse({
            "task_id": task_id,
            "audio_url": None,
            "status": "TTSDONE"
        })
    
    # 队列为空但任务仍在处理中
    logger.info(f"任务 {task_id}: 队列暂时为空，任务处理中")
    return JSONResponse({
        "task_id": task_id,
        "audio_url": None,
        "status": "waiting"
    })

@app.get("/status")
async def get_status():
    """获取服务器状态信息"""
    return {
        "active_tasks": len(tasks),
        "tts_cache_size": len(tts_cache),
        "tts_semaphore": tts_semaphore._value,  # 当前可用的信号量数量
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("======= 启动服务器 =======")
    
    # 启动FastAPI应用
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8009,
        reload=True,
        log_level="info"
    )