"""旧命令名转到新版原生单批基准；旧实验扫描逻辑保存在 legacy 目录。"""

import runpy


def main():
    runpy.run_module("sglang.benchmark.one_batch", run_name="__main__")


if __name__ == "__main__":
    main()
