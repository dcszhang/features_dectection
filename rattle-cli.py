#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rattle.main import main

if __name__ == '__main__':
    # 手动设置参数而不是通过命令行传递
    args = {
        "input": "bytecode",
        "optimize": False,
        "no_split_functions": False,
        "verbosity": "None",
        "supplemental_cfg_file": None,
        "stdout_to": None
    }
    
    # 模拟解析参数后的行为
    # print("Arguments parsed:")
    # for key, value in args.items():
    #     print(f"{key}: {value}")
    main()
