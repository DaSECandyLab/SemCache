# 公共依赖

- `duckdb/`：唯一 Git 子模块，固定 v1.5.4（`08e34c447bae34eaee3723cac61f2878b6bdf787`）。两个扩展和采样使用相同 ABI，SHA-256 统计库复用其 Mbed TLS 源码。
- `nlohmann/`：JSON 头文件，MIT 许可。
- `cpp-httplib/`：模型 HTTP 客户端使用的头文件，MIT 许可。

初始化使用根目录 `git submodule update --init --recursive`。各依赖的许可保留在原目录。原扩展模板的 extension-ci-tools 已无本地构建消费者，归档后移除；根 Makefile 直接调用 pinned DuckDB 的 CMake。
