# clear_cache.py
import os
import shutil
from huggingface_hub import constants

# huggingface_hub 라이브러리의 메인 캐시 폴더 경로를 가져옵니다.
# 보통 C:\Users\사용자명\.cache\huggingface\hub 입니다.
cache_dir = constants.HF_HUB_CACHE

if os.path.exists(cache_dir):
    print(f"Found Hugging Face Hub cache directory at: {cache_dir}")
    print("Deleting the entire cache directory...")
    # rmtree로 해당 폴더와 모든 하위 파일을 삭제합니다.
    shutil.rmtree(cache_dir)
    print("Cache directory completely deleted!")
    # 라이브러리가 오류를 일으키지 않도록 빈 폴더를 다시 생성합니다.
    os.makedirs(cache_dir, exist_ok=True)
    print("Recreated empty cache directory.")
else:
    print("No cache directory found. Nothing to clear.")