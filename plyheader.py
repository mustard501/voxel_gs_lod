import sys
import os

def analyze_ply(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    elements = []
    current_element = None
    file_format = ""
    header_lines = 0

    try:
        with open(file_path, 'rb') as f:
            # 检查魔数
            first_line = f.readline().decode('ascii').strip()
            if first_line != 'ply':
                print("错误: 该文件不是有效的 PLY 文件。")
                return

            for line in f:
                header_lines += 1
                line_str = line.decode('ascii').strip()
                parts = line_str.split()

                if not parts:
                    continue

                # 解析格式
                if parts[0] == 'format':
                    file_format = " ".join(parts[1:])
                
                # 解析元素 (Vertex, Face 等)
                elif parts[0] == 'element':
                    name = parts[1]
                    count = int(parts[2])
                    current_element = {
                        'name': name,
                        'count': count,
                        'properties': []
                    }
                    elements.append(current_element)
                
                # 解析属性 (x, y, z, red, nx 等)
                elif parts[0] == 'property':
                    if current_element is not None:
                        prop_type = parts[1]
                        prop_name = parts[-1] # 处理 list 类型时，名称通常在最后
                        current_element['properties'].append((prop_name, prop_type))
                
                # 结束标志
                elif parts[0] == 'end_header':
                    break

        # 打印分析结果
        print("="*40)
        print(f" PLY 文件结构分析报告")
        print("="*40)
        print(f"文件路径: {file_path}")
        print(f"数据格式: {file_format}")
        print(f"头文件行数: {header_lines + 1}")
        print("-"*40)

        for el in elements:
            print(f"元素: 【{el['name']}】")
            print(f"  - 出现次数 (Count): {el['count']}")
            print(f"  - 包含参数 (Properties):")
            for prop_name, prop_type in el['properties']:
                print(f"    - {prop_name:<12} (类型: {prop_type})")
            print("-"*40)

    except Exception as e:
        print(f"解析出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python analyze_ply.py <your_file.ply>")
    else:
        analyze_ply(sys.argv[1])