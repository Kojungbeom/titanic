# Pandas 함수 정리

- pd.Series(list)
- pd.DataFrame(list)
  - 데이터프레임 만들기

- dropna() or dropna(how='all')
  - 누락된 데이터 골라내기

- np.c_[a,b]
  - 옆으로 이어붙이기

- iloc[ : rows, columns]
  - 해당 값 접근하기
  - loc : 축 이름 선택 시 iloc : 정수 색인으로 선택할 경우

- fillna(value)
  - 결측치를 value로 채우기

- fillna({column : value, column : value})
  - 결측치를 column마다 다른 값으로 채우기

- fillna(method='ffill') or fillna(method='ffill', limit=1)
  - 결측치를 마지막 row 값으로 채우기 (한계지정도 가능)

- duplicated() or drop_duplicates() or drop_duplicates(['column_name'])
  - 중복 제거

- groupby(data[column]).sum()
  - column에서 중복된 row를 합치고, 값도 합치고

- rename(index={변경할row : 변한row}, columns={변경할column, 변한column})
  - Row나 Column을 rename하기

- pd.cut(data[column], bins=bins, labels=labels)
  - 데이터를 bins에 명시한 interval로 나누기
  - labels에 명시한 걸로 categories를 대체하기
- pd.value_counts(data)
  - 같은 row가 몇개 있는지 계산
- categories
  - Interval 확인하기

- pd.qcut(data, 4)
  - 데이터 수에 따라서 4분위로 나누기
- data[condition]
  - Condition에 따라서 값 도출하기
- isnull()
  - null인 것은 True, 아닌것은 False
- pd.notnull(data)
  - null인 것은 False, 아닌것은 False
- head(number)
  - 위에서부터 number개 출력
- tail(number)
  - 아래서부터 number개 출력
- 





# NumPy 함수 정리

- np.random.randn(rows, columns)
  - (row * column) array 생성
  - 범위는 몇?
- npl.array(list or value)
  - Array형 데이터로 만들기
- ndim
  - dimension 알아내기
- shape
  - 몇 바이 몇 인지 알아내기
- dtype
  - 데이터 타입 알아내기
- np.zeros(value) or np.zeros(list or tuple)
  - 0으로 된 array생성하기
- np.empty(list or tuple)
  - zeros랑 비슷함
- data_slice = data[first : last + 1]
  - 브로드캐스팅
- np.maximum(array_x, array_y)
  - shape가 같은 두개의 array에서 큰 값만 골라서 새로운  array를 만드는 방법
- 

