import json
import sys


def build_items(project_type: str) -> list[str]:
    if project_type == "python":
        return [
            "确认版本号、变更说明和发布目标环境",
            "运行自动化测试并确认关键用例通过",
            "检查依赖版本、锁文件和构建产物是否一致",
            "确认配置项、环境变量和密钥注入方式正确",
            "准备回滚方案和发布后验证项",
        ]

    return [
        "确认发布范围",
        "完成基本测试",
        "准备回滚方案",
    ]


def main() -> None:
    project_type = sys.argv[1] if len(sys.argv) > 1 else "generic"
    payload = {
        "project_type": project_type,
        "items": build_items(project_type),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
