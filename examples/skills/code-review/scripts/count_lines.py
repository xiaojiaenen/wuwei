"""示例 skill 脚本：统计代码行数"""
import sys

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not file_path:
        print("请提供文件路径")
        sys.exit(1)

    with open(file_path) as f:
        lines = f.readlines()

    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    print(f"总行数: {total}, 空行: {blank}, 有效行: {total - blank}")
