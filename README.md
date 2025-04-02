# 语音生成服务 (Mini RAG TTS)

这是一个使用 fastapi 构建的rag系统。
重点在于实时语音生成服务，支持将llm 返回的文本进行分块tts。可以在生成文本的同时转换为语音，无需等待全部文本生成完毕。

## 环境变量配置

在使用前，请确保正确设置以下环境变量：

```
# OpenAI API 配置
OPENAI_API_KEY=你的OpenAI API密钥
OPENAI_BASE_URL=https://api.openai.com/v1 或你的代理URL
OPENAI_MODEL=gpt-4o-mini  # 或其他支持的模型

# TTS API 配置
TTS_API_BASE=https://api.example.com  # TTS API的基础URL
TTS_API_KEY=你的TTS API密钥
TTS_API_MODEL=TTS模型名称
TTS_API_VOICE=语音选项
```

## API 接口说明

### 1. 生成音频

启动文本到语音的生成任务。

**请求**:
```
POST /generate_audio
```

**请求体**:
```json
{
  "question": "你想要转换为语音的文本内容"
}
```

**响应**:
```json
{
  "task_id": "e33df09f-38ff-4084-91b8-7278385421ad",
  "status": "processing",
  "message": "Audio generation started"
}
```

### 2. 获取下一个音频段落

按顺序获取生成的音频片段。可以多次调用此接口以获取所有音频片段。

**请求**:
```
GET /get_next_audio/{task_id}
```

**响应**:
- 如果有可用的音频片段，返回音频文件
- 如果正在处理中但尚未有新的音频片段:
  ```json
  {
    "status": "pending",
    "message": "No audio ready yet, try again later"
  }
  ```
- 如果所有音频片段已经生成完毕:
  ```json
  {
    "status": "completed",
    "message": "All audio segments generated"
  }
  ```

### 3. 查询任务状态

查询特定任务的状态和相关信息。

**请求**:
```
GET /status/{task_id}
```

**响应**:
```json
{
  "task_id": "e33df09f-38ff-4084-91b8-7278385421ad",
  "status": "processing", // 可能的值: "processing", "completed", "error", "cancelled"
  "created_at": 1683457892.123456,
  "audio_segments": 3,
  "is_active": true
}
```

### 4. 取消任务

取消正在进行的任务。

**请求**:
```
POST /cancel_task_by_id
```

**请求体**:
```json
{
  "task_id": "e33df09f-38ff-4084-91b8-7278385421ad"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "Task e33df09f-38ff-4084-91b8-7278385421ad cancelled"
}
```

## 使用流程示例

1. 调用 `/generate_audio` 接口开始生成，获取 `task_id`
2. 立即调用 `/get_next_audio/{task_id}` 获取第一个音频片段
3. 继续定期调用 `/get_next_audio/{task_id}` 获取后续音频片段
4. 当 `/get_next_audio/{task_id}` 返回 `"status": "completed"` 时，表示所有音频已生成完毕
5. 如果需要，可以随时通过 `/status/{task_id}` 查询任务状态
6. 如果想取消任务，可以调用 `/cancel_task_by_id` 接口

## 客户端示例代码

```python
import requests
import time
import os

# 配置API基础URL
BASE_URL = "http://localhost:8009"

# 步骤1: 开始语音生成任务
question = "请介绍一下人工智能的历史和发展。"
response = requests.post(f"{BASE_URL}/generate_audio", json={"question": question})
data = response.json()
task_id = data["task_id"]
print(f"任务ID: {task_id}")

# 步骤2: 循环获取音频片段
audio_index = 0
completed = False

while not completed:
    response = requests.get(f"{BASE_URL}/get_next_audio/{task_id}")
    
    # 如果是音频文件
    if response.headers.get('Content-Type') == 'audio/wav':
        filename = f"audio_{audio_index}.wav"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"已保存音频片段: {filename}")
        audio_index += 1
    else:
        data = response.json()
        status = data.get("status")
        
        if status == "completed":
            print("所有音频生成完毕")
            completed = True
        elif status == "pending":
            print("等待生成更多音频...")
            time.sleep(1)  # 等待1秒后再次尝试
        else:
            print(f"未知状态: {status}")
            break

# 可选: 查询任务状态
response = requests.get(f"{BASE_URL}/status/{task_id}")
print(f"任务状态: {response.json()}")
``` 