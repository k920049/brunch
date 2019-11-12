브런치 사용자를 위한 글 추천 대회
==========
## 알고리즘
 대회에 사용된 데이터는 시계열 데이터로 시간축의 변화에 따른 사용자의 취향 변화가 있었습니다. 그러므로 일반적인 Collaborative Filtering, Content-based, Matrix Factorization 의 방법이 잘 적용되지 않을 것이라 생각했습니다. 따라서 time-varying context 를 반영하는 모델을 사용하기로 결정하였습니다. Time-sensitive recommender systems 에는 다음과 같은 분류가 있습니다.

### Temporal Collaborative Filtering
##### Recency-based models
오래된 사용자 데이터일수록 적은 weight 를 주고, 비교적 최근의 사용자 데이터일수록 큰 weight 를 주어서 사용자 데이터의 중요도를 반영하는 방법입니다. 간단히 구현해 사용할 수 있다는 장점이 있지만, 시계열 모델만큼 사용자의 취향 변화를 민감하게 캐치해 낼 수 없다는 단점이 있습니다.
##### Periodic context-based models
시간을 범주형 변수로 분할해(계절, 낮/밤 등등) 모델링하는 방법입니다. Pre-filtering 과 Post-filtering 방법론이 각기 존재합니다. 두 접근방법 모두 일반적인 Pre/Post filtering 이 지니는 단점을 공유합니다.
##### Time-Series models
시간 자체를 독립적인 변수로 취급해 모델링하는 방법입니다. 파라미터 수가 증가한 만큼 시간 변화를 잘 반영할 수 있지만, 데이터가 충분히 크지 않다면 과적합이 발생할 가능성이 큽니다.

### Discrete Temporal models
##### Markovian Models
마르코프 모델을 이용한 Sequence 모델입니다.  Web-log 데이터나 Market Basket 데이터같은 짧은 시간내에 일어나는 취향 변화를 감지할 수 있습니다. 단점으로는 데이터의 변화가 short memory assumption 을 따르지 않으면(즉 직전 상태에 영향을 받지 않는 경우) 취향변화를 잘 감지해 내지 못합니다.

저는 데이터의 변화가 Markov assumption 을 따른다고 가정하고 딥러닝을 접목한 [1] 논문을 기반으로, Recency-based model 의 장점을 반영할 수 있도록 타임스탬프를 feature 에 접합한 [2] 논문을 활용하였습니다.

### 결과
| MAP | NDCG | Entropy |
|:--------|:--------:|--------:|
| 0.019841 | 0.072063 | 6.256637 |


### Reference
[1] Kumar, V., Khattar, D., Gupta, S., Gupta, M., & Varma, V. (2017). Deep Neural Architecture for News Recommendation. CLEF.<br>
[2] Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for YouTube recommendations. Proceedings of the 10th ACM Conference on Recommender Systems, Boston, Massachusetts, USA. pp. 191-198.
