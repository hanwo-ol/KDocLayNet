아이디어 논문: Stock Prices Prediction using the Title of Newspaper Articles with Korean Natural Language Processing
 
링크: "https://ieeexplore.ieee.org/document/8668996"

실제 데이터로 실행하기 위한 단계별 계획서 입니다.


### 1단계: 뉴스 데이터 수집목표: 

특정 주식 종목(예: 카카오)에 대한 뉴스 기사 제목과 해당 기사의 발행 날짜를 수집합니다.

방법:뉴스 크롤링
* 웹 스크레이핑: requests, BeautifulSoup 등의 파이썬 라이브러리를 사용하여 뉴스 웹사이트에서 직접 데이터를 추출할 수 있습니다. 
* 금융 데이터 서비스: 일부 금융 데이터 제공 업체에서 뉴스 데이터를 함께 제공하기도 합니다.결과: 날짜와 뉴스 제목 컬럼을 가진 데이터 (예: CSV 파일 또는 Pandas DataFrame)
* 이번 서브 프로젝트에서는, 이 과정을 무료 오픈소스 크롤러를 이용하겠음.(크롤링 코드 구축 시간 절감을 위함.) ----완료


```
  publish_date         title
0    2023-01-01   카카오, 새해 첫 거래일 상승 마감
1    2023-01-02   카카오뱅크, 신규 대출 상품 출시
...         ...             ...
```

### 2단계: 주가 데이터 수집목표: yfinance package 이용해서 실행
* 완료

### 3단계: 데이터 전처리 및 타겟 변수 생성목표: 

뉴스 데이터와 주가 데이터를 결합하고, 각 뉴스 발행일 기준 5거래일 후의 주가 등락 여부(target)를 계산합니다.주요 작업:데이터 병합: 뉴스 데이터와 주가 데이터를 날짜 기준으로 병합합니다. (Pandas의 merge 함수 활용)5거래일 후 종가 계산: 각 뉴스 발행일에 대해 5거래일 뒤의 날짜를 찾고 해당 날짜의 종가를 가져옵니다. (Pandas의 shift, bdate_range 또는 tseries.offsets.BDay 등을 활용하여 주말/공휴일 제외)수익률 계산: r_t = (P_t+5 / P_t) - 1 공식을 사용하여 수익률을 계산합니다. (P_t: 뉴스 발행일 종가, P_t+5: 5거래일 후 종가)타겟 변수 생성: 수익률(r_t)이 0보다 크면 1(상승), 아니면 0(하락/보합)으로 target 컬럼을 생성합니다.결과: 날짜, 뉴스 제목, 타겟 변수 컬럼을 가진 최종 데이터프레임   


## 4단계: 데이터 통합 및 코드 적용목표: 준비된 실제 데이터를 기존 파이썬 코드(stock_prediction_nlp_cnn_kr)에 적용합니다.방법:기존 코드의 # --- 1. 데이터 준비 (예시 데이터 생성) --- 섹션을 삭제하거나 주석 처리합니다.해당 위치에 3단계에서 생성한 최종 데이터프레임(df)을 로드하는 코드를 추가합니다. (예: df = pd.read_csv('real_stock_news_data.csv'))데이터프레임의 컬럼명이 기존 코드에서 사용된 이름(title, target)과 일치하는지 확인하고, 다르면 맞춰줍니다.이후의 코드(명사 추출, Word2Vec 학습, CNN 모델 학습 및 평가)를 그대로 실행합니다.주의사항:데이터 품질: 수집한 데이터에 결측치나 오류가 없는지 확인하고 정제하는 과정이 중요합니다.거래일 계산: 5 '거래일' 후를 정확히 계산하려면 한국 시장의 휴장일을 고려해야 합니다. FinanceDataReader나 pandas의 비즈니스 데이 관련 기능을 활용하면 도움이 됩니다.데이터 양: 모델 성능은 데이터의 양과 질에 크게 좌우됩니다. 논문과 유사한 성능을 기대하려면 충분한 양의 데이터를 확보해야 합니다.시간 소요: 데이터 수집 및 전처리 과정은 상당한 시간과 노력이 필요할 수 있습니다.이 가이드를 따라서 실제 데이터를 준비하고 기존 코드에 적용하면, 논문의 방법론을 실제 데이터로 테스트해 볼 수 있습니다.
