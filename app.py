from flask import Flask, request, jsonify, render_template
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import os
from konlpy.tag import Okt
import requests
import re

app = Flask(__name__)

# Firebase 초기화
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Okt 형태소 분석기
okt = Okt()

# 텍스트 전처리 함수 (Flask 앱에서 모델을 불러오기 전에 정의)
def tokenize_korean(text):
    return ' '.join(okt.morphs(text))  # 형태소 분석 후 리스트를 문자열로 결합

# 모델 및 벡터화기 로드 (tokenize_korean 함수가 정의된 후 불러옴)
model = joblib.load('sms_spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 정확도 로드
accuracy = 0.0
if os.path.exists('model_accuracy.txt'):
    with open('model_accuracy.txt', 'r') as f:
        accuracy = float(f.read())

# 스미싱 링크 데이터 가져오기
def get_smishing_links():
    url = "https://api.odcloud.kr/api/15109780/v1/uddi:707478dd-938f-4155-badb-fae6202ee7ed?page=1&perPage=1986"
    service_key = "9IZyKdl19TjST2Cjq0YYi8XwbV%2BGTCOD3DE2XdT%2BTGHY9akOskrVLU28bT8AlUpk8%2B%2BHg2zE5PP3BntcMsiM6Q%3D%3D"
    headers = {"Content-Type": "application/json"}
    
    response = requests.get(url, headers=headers, params={"serviceKey": service_key})
    if response.status_code == 200:
        data = response.json()
        links = [item['phishing_site'] for item in data['data']]
        return links
    return []

# URL 추출 함수
def extract_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

# URL 정규화 함수
def normalize_url(url):
    return re.sub(r'^https?://(www\.)?', '', url).strip().strip('/')

# 메시지 분석 함수
def analyze_message(message):
    smishing_links = get_smishing_links()
    normalized_smishing_links = [normalize_url(link) for link in smishing_links]
    
    urls = extract_urls(message)
    for url in urls:
        normalized_url = normalize_url(url)
        if any(normalized_url in link for link in normalized_smishing_links):
            return 1  # 스미싱으로 판단

    # 인공지능 모델을 사용한 스미싱 분석
    processed_message = tokenize_korean(message)  # 메시지를 형태소 분석 후 벡터화
    features = vectorizer.transform([processed_message])
    prediction = model.predict(features)
    return prediction[0]

# 기본 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy)

@app.route('/declation')
def declation():
    return render_template('declation.html', accuracy=accuracy)

@app.route('/usuage')
def usuage():
    return render_template('usuage.html', accuracy=accuracy)

@app.route('/customer')
def customer():
    return render_template('customer.html', accuracy=accuracy)

# 메시지 분석 API
@app.route('/analyze', methods=['POST'])
def analyze():
    content = request.json
    message = content.get('message')

    is_spam = analyze_message(message)

    data = {
        'message': message,
        'is_spam': bool(is_spam)
    }
    db.collection('sms_analysis').add(data)

    response = {
        'is_spam': bool(is_spam)
    }
    if is_spam:
        response['advice'] = {
            'description': "이 메시지는 피싱 문자로 예상됩니다. 이 절차를 따라 주십시오: ",
            'steps': [
                "1. 절대 메시지에 포함된 링크를 클릭하거나 응답하지 마세요.",
                "2. 메시지를 받은 경우, 금융감독원에 신고하세요.",
                "3. 스미싱 관련 피해가 발생했을 경우, 가까운 경찰서에 방문하여 신고하시기 바랍니다.",
            ],
            'contacts': [
                {"name": "금융감독원", "phone": "1332"},
                {"name": "경찰청 사이버안전국", "phone": "182"},
                {"name": "한국인터넷진흥원(KISA)", "phone": "118"}
            ]
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
