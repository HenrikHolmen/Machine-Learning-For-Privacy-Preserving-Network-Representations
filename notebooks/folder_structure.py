import os


def print_dir_tree(start_path=".", max_depth=5, prefix=""):
    for root, dirs, files in os.walk(start_path):
        level = root[len(start_path) :].count(os.sep)
        if level >= max_depth:
            continue
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  {f}")


print_dir_tree(".", max_depth=5)
