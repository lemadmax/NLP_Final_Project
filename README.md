# NLP_Final_Project
Maximize the score for detecting ARG1 on partitives

## Set up Environment
1. Install Python3 (and add python in your PATH environment variables)
2. open a CMD window and cd to the project directory
3. Run: pip install -r requirements.txt

## Run our Systems
using the following console commands:
```sh
python code.py training-data test-data
```


For Example: within the root directory of our project run
```sh
python task2/task2b.py part-dev part-dev
```


the result will be generated to task2/res_2b.txt

to score thie result, run:
```sh
python score.py part-dev task2/res_2b.txt
```

