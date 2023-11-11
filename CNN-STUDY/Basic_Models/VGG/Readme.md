## VGG 특징
1. receptive field
- 3 x 3 filter 를 사용한다.
- 3 x 3 filter 가 두 번 적용 되면 image 에 5 x 5 filter를 적용한 것과 결과가 같다.
- 3 x 3 filter 가 세 번 적용 되면 image 에 7 x 7 filter를 적용한 것과 결과가 같다.
- 하지만, 5 x 5 또는 7 x 7 을 바로 적용하는 것보다 3 x 3 filter 여러 번이 효과가 좋다고 한다.

2. Block
- VGG부터(?) 모델의 구조가 하나의 Block 처럼 쌓는 형태의 구조가 나타난다.
- conv2d, conv2d, pooling

**receptive field** 이다.  

![image](https://user-images.githubusercontent.com/57121112/148213701-acc6efd7-7f0c-4d3c-8b31-5630d413d350.png)

 **Block** 형태를 확인할 수 있다.  

![image](https://user-images.githubusercontent.com/57121112/148214172-91869867-f157-4ae1-bee4-6153baacd73c.png)
