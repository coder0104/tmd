import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
from konlpy.tag import Okt

# 형태소 분석기 설정
okt = Okt()

# 텍스트 전처리 함수 (형태소 분석 후 리스트를 문자열로 결합)
def tokenize_korean(text):
    return ' '.join(okt.morphs(text))

# 한국어 데이터셋 예시
data = {
    'message': [
        "지금 당장 계좌번호를 입력하세요!",
        "축하드립니다! 경품에 당첨되셨습니다. 링크를 클릭하세요.",
        "무료로 쿠폰을 받으세요!",
        "이 메시지는 스팸입니다",
        "이 메시지는 정상입니다",
        "내일 10시에 회의가 있습니다",
        "사랑하는 고객님, 당사의 특별 이벤트에 초대합니다.",
        "신용카드 정보를 입력하고 상품을 받으세요!",
        "주문하신 상품이 발송되었습니다. 확인하시려면 여기를 클릭하세요.",
        "업무 관련된 중요한 회의가 내일 3시에 있습니다.",
        "모바일 청구서를 확인하려면 여기를 클릭하세요.",
        "친구와 함께 여행을 계획하세요! 특별 할인 중입니다.",
        "전화번호를 등록하고 경품에 참여하세요.",
        "무료 영화 예매권을 받으세요!",
        "스팸 메시지를 주의하세요.",
        "고객님의 배송이 지연되고 있습니다. 확인해 주세요.",
        "지금 바로 특별 할인을 받아보세요.",
        "오늘만! 최대 50% 할인 이벤트.",
        "가족과 함께 하는 즐거운 시간을 위한 팁.",
        "비밀번호를 변경하고 보안을 강화하세요."
    ],
    'label': [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]  # 1은 스팸, 0은 정상
}

# DataFrame으로 변환
df = pd.DataFrame(data)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 벡터화 (형태소 분석 후 리스트를 문자열로 변환하여 전달)
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform([tokenize_korean(text) for text in X_train])
X_test_vect = vectorizer.transform([tokenize_korean(text) for text in X_test])

# 모델 학습
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# 모델 평가
y_pred = model.predict(X_test_vect)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# 모델 및 벡터화기 저장
joblib.dump(model, 'sms_spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# 정확도 저장
with open('model_accuracy.txt', 'w') as f:
    f.write(str(accuracy))
