# AGENTS.md

## Environment Rules

当前项目的所有 Python 脚本执行，必须使用 conda 的 `cu128` 环境。

禁止：

- 使用系统 python
- 使用默认 python
- 使用 py/python/python3 直接运行脚本
- 使用未激活环境执行 pip

统一使用以下方式执行：

```bash
conda run -n cu128 python xxx.py
```
