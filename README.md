# Hiragana_OCR


This GitHub branch contains the Hiragana OCR module, which is a crucial component of our larger Japanese Character Recognition project. The main objective of this branch is to accurately recognize and convert handwritten or printed Hiragana characters into digital text format. 


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

