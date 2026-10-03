## 본 내용은 인프런의 강의명(딥러닝 CNN 완벽 가이드 - Fundamental 편)를 듣고 요약했습니다. 
    수업에서 제공하는 이미지입니다. 이미지를 함부로 사용하지 말아주세요!!!

선생님 명언: 기억이 안난다면 돌아가서 다시본다. 돌아가서 보는 것은 **용기**다!<br/>

#### Image Classification
Image -> Feature Extraction -> Image Classification(Fully connected layer)<br/>
Low-level Feature -> Mid-Level Feature -> High-Level Feature -> Trainable Classifier(상세화 -> 추상화)<br/>

![image](https://user-images.githubusercontent.com/57121112/121198441-1872b480-c8ad-11eb-8c6a-749121f02a6b.png)

## CNN(Convolution)을 한다는 것은~?
단순하게 CNN이라고 하면 **Feature Extraction**을 하는 것이다.<br/>
Classification과는 별개!<br/>

- Classification에 맞는 최적의 Feature를 추출
- 최적의 Feature 추출을 위한 최적 Weight값을 계산
- 최적 Feature 추출을 위한 필터(필터 Weight) 값을 계산

![image](https://user-images.githubusercontent.com/57121112/121038008-e26cfc00-c7ea-11eb-8d0c-58c8f886867d.png)

feature map의 크기는 줄지만, channel 수가 증가하여 두껍게 변한다.<br/>
뒤로 갈수록 추상적인 복잡한 feature들이 존재한다.

--------------------------------------------------
### Filter & Kernel(channels)
Filter는 여러 개의 커널(kernel)로 이루어져있다.<br/>
커널의 개수가 channle의 개수이다.<br/>
Conv2D(filter=32, kernel_size=3)(input_tensor)<br/>
아래 이미지로는 filter 32개이고 각 filter 안의 kernel size 3 x 3이고 개수는 알 수없다.<br/>
![image](https://user-images.githubusercontent.com/57121112/121202589-60470b00-c8b0-11eb-9994-0530e8ca5255.png)

Deep Learning CNN은 Fiter 값(가중치?)을 사용자가 만들거나 선택할 필요가 없다.<br/>

------------------------------------------------------
### kernel size 특징
- kernel size(크기)라고 하면 면적(가로X세로)을 의미하며 가로와 세로는 서로 다를 수 있지만 **보통은 일치한다.(정방행렬, 3X3, 5X5, 7X&)**
- kernel 크기가 크면 클수록 입력 Feature Map에서 더 큰 Feature 정보를 가져 올 수 있음.
- 하지만, 큰 사이즈의 Kernel(Filter)로 Convolution 연산을 할 경우 훨씬 많은 연산량과 파라미터가 필요함.

--------------------------------------------------------

### padding
- Filter를 적용하여 Conv 연산 수행 시 출력 Feature Map이 입력 Feature Map 대비 계속적으로 작아지는 것을 막기 위해 적용
- padding='same'은 좌우 끝과 상하 끝에 행과 열을 추가하여 0 값을 추가한다.(zero padding)
- 모서리 주변(좌상단, 우상단, 좌하단, 우하단)의 Conv 연산 횟수가 증가되어 모서리 주변 feature 들의 특징을 보다 강화하는 장점이 있다.
- 모서리 주변에 0 값이 입력되어 Noise가 약간 증가되는 우려도 있지만 큰 영향은 없음.
- padding='same', padding='valid'
![image](https://user-images.githubusercontent.com/57121112/121202090-fd557400-c8af-11eb-9b84-3cd7e7fc773e.png)

### Stride
Conv Filter를 적용할 때 Sliding Window가 **이동하는 가격을 의미**<br/>
stride를 키우면 **공간적인 feature 특성을 손실할 가능성이 높다.** 하지만, 오히려 **불필요한 특성을 제거하는 효과**를 가져오고 또한 **Convolution 연산 속도를 향상** 시킴.
![image](https://user-images.githubusercontent.com/57121112/121206593-8e7a1a00-c8b3-11eb-9696-3400147d2b9e.png)

### pooling(subsmapling)
- Conv 적용된 Feature map의 일정 영역 별로 하나의 값을 추출하여(MAX 또는 Average 적용) Feature map의 사이지를 줄임
- 일정 영역에서 가장 큰 값(MAX) 도는 평균값(Average)을 추출하므로 위치의 변화에 따른 feature의 변화를 일정 수준 중화시킬 수 있다. 
- Feature Map의 크기를 줄이면 위치의 변화에 따른 feature 값의 영향도를 줄여서(Spatial invariance) Generalization, 오버피팅 감소 등의 장점을 얻을 수 있다.
- Max Pooling의 경우 Sharp한 feature 값을 추출하고 Average Pooling의 경우 Smooth한 feature 값을 추출
- LeNet, AlexNet, VGG의 경우는 CNN(Stride/Padding) -> Activation -> Pooling으로 이어지는 전형적인 구조
- 하지만, ResNet부터 이어지는 최근 CNN에서는 최대한 Pooling을 자제하고 Stride를 이용하여 Netwokr를 구성하고 있다.
-----------------------------------------------------------
### 다채널 입력 데이터의 Conv 적용
- input channel 개수에 따라 filter의 kernel channel 수가 동일해야한다. (파랑색 숫자)
- filter의 개수에 따라 output chanel 수가 결정된다. (붉은색 사각형)
- **확실하게 이해하고 넘어가야 다른 글이나 논문들을 읽을 때 shape를 이해할 수 있다!!**

![image](https://user-images.githubusercontent.com/57121112/130015139-bd620f4f-fd25-43a7-bd96-b73c7e311b1d.png)
-----------------------------------------------------------
## Quiz
![image](https://user-images.githubusercontent.com/57121112/119850632-af07b300-bf48-11eb-860a-2142caf04ca5.png)
1. Filter의 개수는?
2. Kernel의 크기(Size)는?
3. Filter의 Channel수는?
4. 출력 Feature Map의 Channel수는?

정답
1. 128개
2. 3 by 3
3. 3 (input의 channel 개수와 filter의 kernel channel 개수가 같아야한다)
4. 128

## 심화 Quiz(불친절)
![image](https://user-images.githubusercontent.com/57121112/119852613-67822680-bf4a-11eb-84eb-fd05b441d7e2.png)


## 출력 Feature Map 크기 계산 공식

$O = \frac{I - F + 2P}{S} + 1$  
I는 입력 Feature Map의 크기  
F는 Filter의 크기(Kernel size), F=3은 3X3 Filter를 의미,   
P는 Padding(정수),  좌우상하 동일하게 적용  
S는 Strides(정수)  
