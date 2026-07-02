import os
import ssl
import certifi

# 1. 清空代理 (保留你原有的逻辑)
proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
for v in proxy_vars:
    if v in os.environ:
        del os.environ[v]

# 2. 【关键步骤】设置环境变量指向 certifi 的证书文件
# 这告诉 requests/urllib3/aiohttp 等库使用 certifi 的证书，而不是系统证书
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# 3. 重新创建默认的 HTTPS 上下文，显式指定使用 certifi 的证书
# 注意：这里不再使用 _create_unverified_context，而是创建一个基于 certifi 的安全上下文
# 如果 dashscope 内部强制验证证书，这样做更安全；如果它忽略验证，这也能避免 ASN1 解析错误
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    # 如果依然报错，可以尝试放宽验证（仅用于调试，生产环境不建议）
    # ssl_context.check_hostname = False
    # ssl_context.verify_mode = ssl.CERT_NONE

    # 覆盖默认上下文
    ssl._create_default_https_context = lambda: ssl_context
except Exception as e:
    print(f"配置 SSL 上下文时出错: {e}")
    # 回退方案：如果上述方法无效，再尝试完全禁用验证（不推荐，但可绕过 ASN1 错误）
    ssl._create_default_https_context = ssl._create_unverified_context

# 4. 设置 API Key
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

# 5. 导入并使用 dashscope
import dashscope
from dashscope import TextEmbedding

dashscope.base_http_api_config.http2 = False
dashscope.base_http_api_config.timeout = 120

resp = TextEmbedding.call(model="text-embedding-v4", input="测试文本")
print("接口正常返回:", resp)
