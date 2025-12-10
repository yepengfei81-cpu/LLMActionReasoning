"""
OpenAI API 连接测试脚本
"""

import os
import yaml

def test_openai_connection():
    """测试 OpenAI API 连接"""
    
    # 1. 设置代理（使用你的代理端口 7897）
    proxy_url = "http://127.0.0.1:7897"
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    print(f"🌐 使用代理: {proxy_url}")
    
    # 2. 检查 openai 库是否安装
    try:
        import openai
        print(f"✅ openai 库已安装，版本: {openai.__version__}")
    except ImportError:
        print("❌ openai 库未安装，请运行: pip install openai")
        return False
    
    # 3. 从配置文件读取 API Key
    config_path = "configs/kuka_six_bricks.yaml"
    api_key = None
    model = "gpt-4o-mini"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            llm_cfg = cfg.get('llm', {})
            api_key = llm_cfg.get('api_key')
            model = llm_cfg.get('model', 'gpt-4o-mini')
            print(f"📄 从配置文件读取: model={model}")
    
    # 也检查环境变量
    env_api_key = os.environ.get('OPENAI_API_KEY')
    if env_api_key:
        print("📄 检测到环境变量 OPENAI_API_KEY")
        api_key = env_api_key
    
    if not api_key:
        print("❌ 未找到 API Key，请在配置文件或环境变量中设置")
        return False
    
    # 隐藏显示 API Key（只显示前8位和后4位）
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"🔑 使用 API Key: {masked_key}")
    
    # 4. 测试连接
    print("\n🔄 正在测试 API 连接...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        # 发送一个简单的测试请求
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'Hello, connection test successful!' in one short sentence."}
            ],
            max_tokens=50,
            temperature=0.0
        )
        
        reply = response.choices[0].message.content
        print(f"\n✅ API 连接成功!")
        print(f"📨 模型回复: {reply}")
        print(f"📊 使用的模型: {response.model}")
        print(f"📊 Token 使用: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        
        return True
        
    except openai.AuthenticationError as e:
        print(f"\n❌ API Key 认证失败: {e}")
        print("   请检查 API Key 是否正确、是否过期")
        return False
        
    except openai.RateLimitError as e:
        print(f"\n❌ API 请求频率限制: {e}")
        print("   可能是配额用尽或请求过于频繁")
        return False
        
    except openai.APIConnectionError as e:
        print(f"\n❌ API 连接错误: {e}")
        print("   可能的原因:")
        print("   1. 代理端口 7897 是否正确？")
        print("   2. 代理软件是否正常运行？")
        print("   3. 尝试其他端口如 7890")
        return False
        
    except Exception as e:
        print(f"\n❌ 未知错误: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("="*50)
    print("OpenAI API 连接测试")
    print("="*50 + "\n")
    
    success = test_openai_connection()
    
    print("\n" + "="*50)
    if success:
        print("🎉 测试完成，API 连接正常!")
        print("\n💡 在运行主程序时，请先设置环境变量:")
        print("   export HTTP_PROXY=http://127.0.0.1:7897")
        print("   export HTTPS_PROXY=http://127.0.0.1:7897")
    else:
        print("❌ 测试失败，请根据上述提示解决问题")
    print("="*50)