Tensorflow Obeject detection tutorial은 현재 tf.2* 에 맞춰져있다.  
만약, tf1.1* 버전을 사용하면 Error가 발생한다.  

### 방법1.
 두 버전을 사용되게 해야하는데  

 모듈? 라이브러리를 다 호출 후 바로 아래에 아래 코드를 하나 더 작성해줘야한다.  
 tf.compat.v1.enable_eager_execution()  

 그리고  
 tf.compat.v2 or tf.compat.v1을 사용해서 버전을 맞춰줘야한다.  
 model = tf.saved_model.load_v2(str(model_dir)) -->  model = tf.compat.v2.saved_model.load(str(model_dir))  

 아래 Issue 참고 해결 방법이다.   
 https://github.com/tensorflow/models/pull/8556  
 
 
### 방법2
Tensorflow Obeject detection tutoria의 release 버전을 tf1.13으로 맞추고 실행해야한다.  
아래 tf1.13의 주소이다.  
https://github.com/tensorflow/models/blob/57e075203f8fba8d85e6b74f17f63d0a07da233a/research/object_detection/object_detection_tutorial.ipynb  



코드는 최대한 깔끔하게 정리해서 다시 올리겠따~!!  
현재 코드의 재생산성을 높이기 위해..FLAGS를 통해 작성하고 있다.    
단순히 되게끔만 하면 쉽기는 하지만 추후 계속 사용을 고려해야한다.  
python progressbar을 사용해서 진행 상황 표시바도 만들어야된다!!  


### how to make tfrecord!?
convert_tfrecrod.py  
#### 방법


크게 두 가지로 나눈다.
1. tf.train.Example 만드는 함수
  - example을 만들면서 생기는 궁금증
    1. bounding box의 format은 어떤 형태로 저장?
       - pascal VOC or COCO ?!
    2. image는 어떤 형태로 저장?
       - bytes_feature? 
    3. train은 되는데 val도 자동으로 저장을 어떻게?

2. 이미지를 지정한 개수만큼 나눠서(shard) tfrecord 파일에 저장한다.
   - 

참고사이트
 - https://juhyung.kr/t/practical-tf-2-0-tfrecords/50/1
 - https://medium.com/@rodrigobrechard/tfrecords-how-to-use-sharding-94059e2b2c6b
 - https://www.tensorflow.org/tutorials/load_data/tfrecord#creating_a_tftrainexample_message




2021.12.09
- 놀라운 사실 
- tf1_video 파일에서 이미지 1500장 모델 결과 확인하려면 예상시간이 15분이었다.
- tf1_vcideo_version01 은 1500장이 10초면 끝이난다. 왜? 어떻게?
- 모델을 memory에 넣는 부분과 tf.Session() as sess를 아예 밖으로 빼서 한 번만 불러오면 되도록 바꿨다..^^
- 나랑 비슷한 생각을 했던 외쿡 형이 있다. https://stackoverflow.com/questions/54436458/how-to-keep-session-open-while-detecting-over-multiple-images
- 영상 몇 개 뽑는데 실행시켜놓고 퇴근했는데... 감사하다.

2021.12.14
- Detected out of order event.step likely caused by a TensorFlow restart. 이러한 에러가 발생할 경우 아래 tensorboard를 실행할 때 같이 작성해주면 된다.
- 에러 이유는 train 결과와 eval 결과가 겹쳐서(?) 저장 된 것 같은 느낌으로 인한 에러 발생인 것 같다.
 - --purge_orphaned_data false
