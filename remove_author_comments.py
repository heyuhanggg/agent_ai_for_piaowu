import os
import re

def remove_author_comments(file_path):
    """删除Java文件中的作者注释块"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 匹配包含 @program 的整个注释块
        pattern = r'/\*\*\s*\n\s*\*\s*@program:.*?\*\s*@author:.*?\*\*/'
        
        # 删除注释块
        new_content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        # 如果内容有变化，写回文件
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def process_directory(root_dir):
    """递归处理目录中的所有Java文件"""
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                if remove_author_comments(file_path):
                    count += 1
                    print(f"已处理: {file_path}")
    return count

if __name__ == '__main__':
    root_directory = r'e:\java\code\damai-ai-master\damai-ai-master'
    print("开始删除作者注释...")
    total = process_directory(root_directory)
    print(f"\n完成! 共处理了 {total} 个文件")
