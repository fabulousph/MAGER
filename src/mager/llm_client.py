"""
LLM客户端 - 支持多种API（OpenAI、Anthropic、本地vLLM部署等）
注意：本地vLLM部署使用与OpenAI相同的接口，统一通过OpenAI兼容API调用
"""
import time
from typing import List, Dict, Optional
from config import ModelConfig, get_model_config

class LLMClient:
    """统一的LLM客户端，支持多种API类型"""
    
    def __init__(self, agent_name: str):
        """
        初始化LLM客户端
        
        Args:
            agent_name: 智能体名称 (agent_a, agent_g, agent_d, agent_e)
        """
        self.config = get_model_config(agent_name)
        self.agent_name = agent_name
        
    def _call_openai_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """调用OpenAI兼容的API（包括线上OpenAI和本地vLLM部署）"""
        import openai
        
        # 设置API密钥和base_url
        # 如果是local类型，base_url会在config中设置；如果是openai类型，base_url为None则使用默认OpenAI地址
        api_kwargs = {
            "api_key": self.config.api_key,
        }
        
        if self.config.base_url:
            api_kwargs["base_url"] = self.config.base_url
        
        try:
            client = openai.OpenAI(**api_kwargs)
        except Exception as e:
            raise Exception(f"创建OpenAI客户端失败: {str(e)}")
        
        try:
            # print(f"enable_thinking: {enable_thinking}")
            # 获取max_tokens，优先使用kwargs中的值
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            
            # 检查是否因为长度限制而截断
            if hasattr(response, 'choices') and response.choices:
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    print(f"  ⚠ 警告：响应因达到max_tokens限制({max_tokens})而截断，可能需要增加max_tokens")
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")
        
        # 检查响应是否有效
        if not response:
            raise Exception("API返回空响应")
        
        if not hasattr(response, 'choices'):
            raise Exception("API响应格式错误：缺少choices字段")
        
        if not response.choices or len(response.choices) == 0:
            raise Exception("API返回的choices为空列表")
        
        if not response.choices[0]:
            raise Exception("API响应格式错误：choices[0]为None")
        
        if not hasattr(response.choices[0], 'message'):
            raise Exception("API响应格式错误：缺少message字段")
        
        if not response.choices[0].message:
            raise Exception("API响应格式错误：message为None")
        
        if not hasattr(response.choices[0].message, 'content'):
            raise Exception("API响应格式错误：缺少content字段")
        
        content = response.choices[0].message.content
        if content is None:
            raise Exception("API返回的content为None")
        
        return content
    
    def _call_anthropic_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """调用Anthropic API"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("请安装anthropic包: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.config.api_key)
        
        # Anthropic API的消息格式略有不同
        system_message = None
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = client.messages.create(
            model=self.config.model_name,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            system=system_message,
            messages=conversation_messages
        )
        
        # 检查响应是否有效
        if not response or not hasattr(response, 'content') or not response.content:
            raise Exception("API返回空响应或无效响应")
        
        if len(response.content) == 0:
            raise Exception("API响应content为空列表")
        
        if not hasattr(response.content[0], 'text'):
            raise Exception("API响应格式错误：缺少text字段")
        
        text = response.content[0].text
        if text is None:
            raise Exception("API返回的text为None")
        
        return text
    
    def call(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        调用LLM
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数（temperature, max_tokens等）
            
        Returns:
            LLM生成的文本
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # 根据API类型选择调用方法
        try:
            if self.config.api_type == "anthropic":
                return self._call_anthropic_api(messages, **kwargs)
            else:  # openai 或 local 都使用OpenAI兼容API（vLLM也使用相同接口）
                return self._call_openai_api(messages, **kwargs)
        except Exception as e:
            raise Exception(f"LLM调用失败 (Agent: {self.agent_name}): {str(e)}")
    
    def call_with_retry(self, prompt: str, system_prompt: Optional[str] = None, 
                       max_retries: int = 3, **kwargs) -> str:
        """
        带重试机制的LLM调用
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_retries: 最大重试次数
            **kwargs: 其他参数
            
        Returns:
            LLM生成的文本
        """
        for attempt in range(max_retries):
            try:
                return self.call(prompt, system_prompt, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # 指数退避
                print(f"调用失败，{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        
        raise Exception("达到最大重试次数，调用失败")

