import asyncio
from psycopg_pool import AsyncConnectionPool


async def test_minimal_pool():
    # 1. 定义连接信息 (请替换为你的真实密码)
    # 注意：localhost 在某些环境下可能解析为 IPv6 (::1)，而 PG 可能只监听 IPv4 (127.0.0.1)
    # 建议先尝试将 localhost 改为 127.0.0.1 进行排查
    conninfo = "postgresql://dhx:dhx2589630@192.168.1.4:5432/postgres?sslmode=disable"

    print(f"正在尝试建立连接池: {conninfo}")

    # 2. 创建连接池
    # min_size=1 确保启动时立即尝试建立至少一个连接
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=2,
        kwargs={"autocommit": True}
    )

    try:
        # 3. 打开池子 (这会触发初始连接的建立)
        await pool.open()
        print("✅ 连接池初始化成功 (Pool Opened)")

        # 4. 尝试获取一个连接并执行 SQL
        async with pool.connection() as conn:
            res = await conn.execute("SELECT 1 as test_connection")
            record = await res.fetchone()
            print(f"✅ SQL 执行成功，结果: {record}")

    except Exception as e:
        print(f"❌ 连接失败: {type(e).__name__}: {e}")
        print("\n排查建议:")
        print("1. 检查 host 是 'localhost' 还是 '127.0.0.1'。尝试切换两者。")
        print("2. 检查端口 5432 是否被防火墙阻止。")
        print("3. 检查用户名/密码是否正确。")
        print("4. 如果是在 Docker/容器中，确保服务名解析正确。")

    finally:
        # 5. 关闭池子
        await pool.close()
        print("🔒 连接池已关闭")


if __name__ == "__main__":
    asyncio.run(test_minimal_pool())
