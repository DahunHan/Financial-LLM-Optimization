# test_gemini.py

import os
import google.generativeai as genai

# 🚨 중요: 실제 코드에는 API 키를 직접 넣지 마세요!
# 여기서는 테스트를 위해 임시로 사용하지만,
# 실제 프로젝트에서는 환경 변수 등을 사용하는 것이 안전합니다.
# https://aistudio.google.com/app/apikey 에서 API 키를 발급받으세요.
api_key = "AIzaSyBgspiRbsSx5NnPJoLsQwI-37_DjwRIsqs" # ⬅️ 여기에 자신의 API 키를 붙여넣으세요.

try:
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-flash')

    print("Gemini에게 무엇이든 물어보세요! (종료하려면 'exit' 입력)")

    while True:
        prompt = input("> ")
        if prompt.lower() == 'exit':
            break

        response = model.generate_content(prompt)
        print("Gemini:", response.text)
        print("-" * 20)

except Exception as e:
    print(f"오류가 발생했습니다: {e}")
