# app.py
import requests

try:
    # 테스트를 위해 구글의 메인 페이지에 요청을 보냅니다.
    response = requests.get("https://www.google.com")

    # 응답 상태 코드를 출력합니다. (200이 나오면 성공)
    print(f"Status Code: {response.status_code}")
    print("`requests` package is working correctly!")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")