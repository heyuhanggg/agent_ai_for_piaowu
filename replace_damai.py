import os
import re

def replace_in_file(file_path):
    """替换文件中的"大麦"为"演出票务" """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换"大麦"为"演出票务"
        new_content = content.replace('大麦', '演出票务')
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def main():
    # 需要替换的文件列表
    files = [
        r'vue\src\views\Home.vue',
        r'damai-core-service\src\main\resources\datum\节目订票-相关问题与回答.md',
        r'damai-core-service\src\main\java\org\javaup\ai\constants\DaMaiConstant.java',
        r'damai-core-service\src\main\resources\datum\节目取消和退票-相关问题与回答.md',
        r'damai-core-service\src\main\java\org\javaup\ai\ai\function\call\UserCall.java',
        r'DEPLOY_TO_GITHUB.md',
        r'damai-core-service\pom.xml',
        r'damai-core-service\src\main\java\org\javaup\ai\ai\function\call\OrderCall.java',
        r'damai-core-service\src\main\java\org\javaup\ai\ai\function\call\ProgramCall.java',
        r'damai-core-service\src\main\java\org\javaup\ai\ai\function\call\TicketCategoryCall.java',
        r'damai-core-service\src\main\java\org\javaup\ai\cotroller\ProgramController.java',
        r'damai-mcp-server\damai-mcp-log-service\pom.xml',
        r'damai-mcp-server\damai-mcp-log-service\src\main\java\org\javaup\mcp\tool\LogQueryMcpTool.java',
        r'damai-mcp-server\damai-mcp-metrics-service\pom.xml',
        r'damai-mcp-server\damai-mcp-metrics-service\src\main\java\org\javaup\mcp\tool\MetricsQueryMcpTool.java',
        r'damai-mcp-server\pom.xml',
        r'vue\index.html',
        r'vue\src\App.vue',
        r'vue\src\views\DaMaiAi.vue',
        r'vue\src\views\DaMaiAnalysis.vue',
    ]
    
    base_dir = r'e:\java\code\damai-ai-master\damai-ai-master'
    count = 0
    
    print("开始替换文件中的'大麦'为'演出票务'...\n")
    
    for file in files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            if replace_in_file(file_path):
                count += 1
                print(f"✓ 已替换: {file}")
        else:
            print(f"✗ 文件不存在: {file}")
    
    print(f"\n完成! 共处理了 {count} 个文件")

if __name__ == '__main__':
    main()
