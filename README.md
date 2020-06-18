# Titanic
[Kaggle](https://www.kaggle.com/)에 있는 ["Titanic: Machine Learning from Disaster"](https://www.kaggle.com/c/titanic) 데이터를 가지고 어떤 종류의 사람들이 살아남는지 예측하기

이 프로젝트는 `Ubuntu 18.04LTS` 환경에서 제작했습니다.



## Requirement

- Anaconda
  - Python 3.7
  - Scikit-learn
  - Pandas
  - Jupyter notebook 
  - Matplotlib
- 비상한 두뇌



## Installation 

```terminal 
$ conda create -n titanic python=3.7 scikit-learn pandas jupyter
$ conda activate titanic둔
$ conda list
```

1. Anaconda 환경에서 작업하기 위해  **titanic**이라는 새로운 Environment를 생성
   -  생성과 동시에 `Python 3.7` `scikit-learn` `pandas` `jupyter`를 설치
2. `conda activate titanic`으로 **titanic** Environment를 활성화
3. `conda list`로 Package들이 잘 깔렸는지 확인



```
$ pip install kaggle
```

- `pip install kaggle`로 kaggle 다운로드

  
      


<p align="center"><img src="images/1.png" border="1"></p>

- kaggle 홈페이지 - 회원가입 - 로그인 - 우측상단 프로필 이미지 클릭 - My Account

`Create New API Token`을 클릭하고 kaggle.json을 다운로드 받은 후에, `/home/ines`아래 `.kaggle`이라는 이름의 폴더를 만든 후, 다운로드 받은 kaggle.json파일을 넣는다.



```
$ chmod 600 /home/ines/.kaggle/kaggle.json
$ kaggle competitions list
$ cd
$ mkdir ml_assignment
$ cd ml_assignment
$ kaggle competitions download -c titanic
```

- 터미널에서 `./kaggle`에 들어간 뒤, `chmod 600`으로 `kaggle.json`에 읽기, 쓰기 권한을 부여한다. (자세한 내용은 [이 블로그](https://sehoonoverflow.tistory.com/18)을 참고)
- `kaggle competitions list`를 입력하여 잘 나오는지 확인
- `ml_assignment`를 만들어 그 곳에 **titanic dataset**을 다운받았다.



