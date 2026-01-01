"""
阿里云千问 API 连接测试脚本
"""

import os
import yaml

def test_qwen_connection():
    """测试阿里云千问 API 连接"""
    
    # 1. 千问不需要代理（阿里云国内/国际服务）
    # 清除可能存在的代理设置
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        if key in os.environ:
            del os.environ[key]
    print("🌐 使用阿里云 DashScope API（无需代理）")
    
    # 2. 检查 openai 库是否安装
    try:
        import openai
        print(f"✅ openai 库已安装，版本: {openai.__version__}")
    except ImportError:
        print("❌ openai 库未安装，请运行: pip install openai")
        return False
    
    # 3. 从配置文件读取配置
    config_path = "configs/kuka_six_bricks.yaml"
    api_key = None
    model = "qwen-plus"
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # 国际版默认
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            llm_cfg = cfg.get('llm', {})
            api_key = llm_cfg.get('api_key')
            model = llm_cfg.get('model', 'qwen-plus')
            base_url = llm_cfg.get('base_url', base_url)
            print(f"📄 从配置文件读取:")
            print(f"   model={model}")
            print(f"   base_url={base_url}")
    
    if not api_key:
        print("❌ 未找到 API Key，请在配置文件中设置")
        return False
    
    # 隐藏显示 API Key
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"🔑 使用 API Key: {masked_key}")
    
    # 4. 测试连接
    print("\n🔄 正在测试千问 API 连接...")
    
    try:
        from openai import OpenAI
        
        # 关键：必须设置 base_url！
        client = OpenAI(
            api_key=api_key,
            base_url=base_url  # 指向阿里云端点
        )
        
        # 发送一个简单的测试请求
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, connection test successful!' in one short sentence."}
            ],
            max_tokens=50,
            temperature=0.0
        )
        
        reply = response.choices[0].message.content
        print(f"\n✅ API 连接成功!")
        print(f"📨 模型回复: {reply}")
        print(f"📊 使用的模型: {response.model}")
        if response.usage:
            print(f"📊 Token 使用: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        
        return True
        
    except openai.AuthenticationError as e:
        print(f"\n❌ API Key 认证失败: {e}")
        print("   请检查 DashScope API Key 是否正确")
        return False
        
    except openai.APIConnectionError as e:
        print(f"\n❌ API 连接错误: {e}")
        print("   请检查网络连接")
        return False
        
    except Exception as e:
        print(f"\n❌ 未知错误: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("="*50)
    print("阿里云千问 API 连接测试")
    print("="*50 + "\n")
    
    success = test_qwen_connection()
    
    print("\n" + "="*50)
    if success:
        print("🎉 测试完成，千问 API 连接正常!")
    else:
        print("❌ 测试失败，请根据上述提示解决问题")
    print("="*50)