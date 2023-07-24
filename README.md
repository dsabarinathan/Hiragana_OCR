# Hiragana_OCR


This GitHub branch contains the Hiragana OCR module, which is a crucial component of our larger Japanese Character Recognition project. The main objective of this branch is to accurately recognize and convert handwritten or printed Hiragana characters into digital text format. 
# Dataset:
[Kuzushiji-49](https://github.com/rois-codh/kmnist) is an MNIST-like dataset that has 49 classes (28x28 grayscale, 270,912 images) from 48 Hiragana characters and one Hiragana iteration mark.

# Test Results:

| class | precision | recall | f1_score | support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 1.00   | 0.97     | 1000    |
| 1     | 0.99      | 0.96   | 0.98     | 1000    |
| 2     | 0.96      | 0.98   | 0.97     | 1000    |
| 3     | 0.94      | 0.94   | 0.94     | 126     |
| 4     | 0.99      | 0.98   | 0.98     | 1000    |
| 5     | 0.97      | 0.92   | 0.94     | 1000    |
| 6     | 0.92      | 0.97   | 0.95     | 1000    |
| 7     | 0.92      | 0.95   | 0.94     | 1000    |
| 8     | 0.94      | 0.98   | 0.96     | 767     |
| 9     | 0.98      | 0.94   | 0.96     | 1000    |
| 10    | 0.98      | 0.97   | 0.98     | 1000    |
| 11    | 0.97      | 0.96   | 0.97     | 1000    |
| 12    | 0.98      | 0.95   | 0.97     | 1000    |
| 13    | 0.97      | 0.95   | 0.96     | 678     |
| 14    | 0.99      | 0.95   | 0.97     | 629     |
| 15    | 0.98      | 0.98   | 0.98     | 1000    |
| 16    | 0.95      | 0.99   | 0.97     | 418     |
| 17    | 0.94      | 0.98   | 0.96     | 1000    |
| 18    | 0.96      | 0.96   | 0.96     | 1000    |
| 19    | 0.98      | 0.97   | 0.98     | 1000    |
| 20    | 0.97      | 0.96   | 0.97     | 1000    |
| 21    | 0.99      | 0.90   | 0.95     | 1000    |
| 22    | 0.96      | 0.99   | 0.97     | 336     |
| 23    | 0.96      | 0.98   | 0.97     | 399     |
| 24    | 0.98      | 0.97   | 0.97     | 1000    |
| 25    | 0.99      | 0.94   | 0.97     | 1000    |
| 26    | 0.98      | 0.98   | 0.98     | 836     |
| 27    | 0.97      | 0.97   | 0.97     | 1000    |
| 28    | 0.99      | 0.93   | 0.96     | 1000    |
| 29    | 0.93      | 0.97   | 0.95     | 324     |
| 30    | 0.94      | 0.99   | 0.96     | 1000    |
| 31    | 0.98      | 0.96   | 0.97     | 498     |
| 32    | 0.97      | 0.94   | 0.95     | 280     |
| 33    | 0.98      | 0.97   | 0.98     | 552     |
| 34    | 0.95      | 0.99   | 0.97     | 1000    |
| 35    | 0.96      | 0.98   | 0.97     | 1000    |
| 36    | 0.97      | 0.97   | 0.97     | 260     |
| 37    | 0.96      | 0.99   | 0.97     | 1000    |
| 38    | 0.92      | 0.96   | 0.94     | 1000    |
| 39    | 0.93      | 0.97   | 0.95     | 1000    |
| 40    | 0.95      | 0.96   | 0.95     | 1000    |
| 41    | 0.96      | 0.99   | 0.97     | 1000    |
| 42    | 0.97      | 0.97   | 0.97     | 348     |
| 43    | 0.97      | 0.94   | 0.96     | 390     |
| 44    | 0.92      | 0.82   | 0.87     | 68      |
| 45    | 0.95      | 0.91   | 0.93     | 64      |
| 46    | 0.99      | 0.98   | 0.98     | 1000    |
| 47    | 0.99      | 0.99   | 0.99     | 1000    |
| 48    | 0.94      | 0.84   | 0.89     | 574     |

# Running the model in Docker


- Step 1: clone the below git repository. 

```
https://github.com/dsabarinathan/Hiragana_OCR/
```

- Step 2: Move the config file , get_coordinate and hiragana_ocr_engine file ,models folder to Docker folder. 


- Step 3: Use the below command for docker compose.

```
docker-compose up -d
```
- Step 4: Run the following the command to build the docker image.

```
docker build -t ocr_model .
```
- Step 5: Start the detection service. 

```
 docker run -it ocr_model
```

- Step 6: Pass the image for testing. 

```
curl -X POST -F 'file=@/home/users/sample/1.jpg' http://172.17.0.2:5000/

```

