#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
URL内容读取程序
从用户输入的URL获取网页内容并直接输出
"""

import requests
import sys
import time
import random
from urllib.parse import urlparse


def is_valid_url(url):
    """
    验证URL是否有效
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def fetch_url_content(url):
    """
    从指定URL获取网页内容
    
    Args:
        url (str): 要访问的URL
        
    Returns:
        str: 网页内容，如果出错返回None
    """
    try:
        # 添加User-Agent头部，避免某些网站拒绝请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 设置超时时间为30秒
        response = requests.get(url, headers=headers, timeout=30)
        
        # 检查响应状态码
        response.raise_for_status()
        
        # 返回网页内容
        return response.text
        
    except requests.exceptions.Timeout:
        print(f"错误：请求超时 - {url}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"错误：无法连接到URL - {url}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"错误：HTTP错误 {e.response.status_code} - {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"错误：请求异常 - {e}")
        return None
    except Exception as e:
        print(f"错误：未知异常 - {e}")
        return None


def main():
    """
    主函数
    """
    print("=" * 50)
    print("URL内容读取程序")
    print("=" * 50)
    
    # 获取用户输入的URL
    while True:
        url = input("\n请输入要读取的URL (输入 'quit' 退出): ").strip()
        
        if url.lower() == 'quit':
            print("程序退出。")
            sys.exit(0)
        
        if not url:
            print("URL不能为空，请重新输入。")
            continue
        
        # 如果URL没有协议前缀，自动添加http://
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # 验证URL格式
        if not is_valid_url(url):
            print("无效的URL格式，请重新输入。")
            continue
        
        break
    
    print(f"\n正在访问: {url}")
    print("-" * 50)
    
    # 获取网页内容
    content = fetch_url_content(url)
    
    if content is not None:
        print("网页内容:")
        print("=" * 50)
        print(content)
        print("=" * 50)
        print(f"内容长度: {len(content)} 字符")
    else:
        print("无法获取网页内容。")


if __name__ == "__main__":
    main()
