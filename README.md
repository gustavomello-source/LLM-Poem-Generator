# LLM Poem Generator
<!-- TABLE OF CONTENTS -->

Pipeline for scraping/fetching poetry from the internet, fine-tuning LLM models with such data and
generating poems using a given prompt.

## Table of Contents

- [LLM Poem Generator](#llm-poem-generator)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
    - [Made With](#made-with)
  - [Getting Started](#getting-started)
    - [Pre-requisites](#pre-requisites)
    - [Structure of Files](#structure-of-files)
    - [Installation and Environment Preparation](#installation-and-environment-preparation)
  - [Executing the application](#executing-the-application)
    - [Fetching data from PoetryDB](#fetching-data-from-poetrydb)
    - [Fine-tuning](#fine-tuning)
    - [Generation](#generation)
  - [Edition](#edition)
  - [Future Work](#future-work)

<!-- ABOUT THE PROJECT -->

## About the Project

This repository organizes the data and scripts necessary for scraping poetry from the internet, fine-tuning LLM models and saving them.

The project is divided into three main parts:

1. **Scraping** or **Fetching**: The scraping part is responsible for fetching poetry from the internet. Currently it fetches poetry from the PoetryDB page. It uses the requests library to fetch the data into .json format and then, for training, into .txt.
2. **Fine-tuning**: The fine-tuning part is responsible for fine-tuning the LLM models with the scraped data. It uses the Hugging Face Transformers library to fine-tune the models.
3. **Generation**: The generation part is responsible for generating poems using a given prompt. It uses the fine-tuned models to generate the poems, also comparing the prompts generated before and after, using the same prompt.

### Made With

The project was made using the following technologies:

- [Python](https://www.python.org/) - Programing Languaged used for the project.
- [Transformers](https://huggingface.co/transformers/) - A state-of-the-art library for natural language processing tasks.
- [Requests](https://docs.python-requests.org/en/latest/) - HTTP library for Python.
- [PyTorch](https://pytorch.org/) - An open-source machine learning library for Python, used for applications such as computer vision and natural language processing.

<!-- GETTING STARTED -->

## Getting Started

This section describes the steps to install and run the project.

### Pre-requisites

1. There must be a Python environment installed on the machine. The project was made using Python 3.12.5 version. The installation of Python can be found in the [Python Official Page](https://www.python.org/downloads/).
2. The documentation of the main libraries and technologies used here can be found in the links provided in [Made With](#made-with).
3. It is necessary to follow the file structure described in [File Structure](#file-structure) for the code to work.
4. The user must follow the steps in [Installation and Environment Preparation](#installation-and-environment-preparation).

### Structure of Files

The file structure of the project is as follows, at the moment of installation:

```bash
LLM-Poem-Generator
├── .gitattributes
├── .gitignore
├── config.ini
├── dataset.py
├── fetcher.py
├── LICENSE
├── main.py
├── Poet.py
├── README.md
├── requirements.txt
└── writer.py
```

The main directories and files will be described at the [Edition](#edition) section.

### Installation and Environment Preparation
1. To install and use this project, simply clone the repository. [GitHub Documentation](https://docs.github.com/en).

- Example of cloning a repository:
```
git clone https://repository.url/main-folder.git
```
Where `repository.url` represents the repository address and `main-folder.git` represents the main .git file of the repository.

2. After cloning the project, open a terminal in the root folder of the project and navigate to the "LLM-Poem-Generator" directory.
   
```
$ cd LLM-Poem-Generator
```


3. Create and activate your virtual environment using the following commands:

- **For Unix systems (Linux/macOS):**
```
$ virtualenv -p python3 .venv . .venv/bin/activate
```
- **For Windows:**
```
$ python -m venv .venv .venv\Scripts\activate
```

For more details on how to use virtualenv, visit this [link](https://virtualenv.pypa.io/en/latest/)

4. Install the dependencies
```
(.venv) ....$ pip3 install -r ../requirements.txt
```

## Executing the application
To run the application, the following steps must be followed, after the virtual environment and the steps in [Installation and Environment Preparation](#installation-and-environment-preparation):

1. Execute the following command in the terminal to get help with all the arguments for `main.py`:
```
(.venv) ....$ python main.py -help
```

2. To add fetching the data into the pipeline, execute the command in addition to this argument:
```
(.venv) ....$ python main.py --fetch
```

### Fetching data from PoetryDB
The fetching part of the pipeline is responsible for fetching poetry from the PoetryDB page. It uses the requests library to fetch the data into .json format and then, for training, into .txt. 

All the poems fetched are saved inside the PoetryDB folder, in the root directory of the project.

### Fine-tuning
The fine-tuning part of the pipeline is responsible for fine-tuning the LLM models with the scraped data. It uses the Hugging Face Transformers library to fine-tune the models. At the moment, the only model available for fine-tuning is the opt-350m model.

Note that the fine-tuning part of the pipeline is very time-consuming, so it is recommended to use only GPU for this task, and a powerful one.

### Generation
The generation part of the pipeline is responsible for generating poems using a given prompt. It uses the fine-tuned models to generate the poems, also comparing the prompts generated before and after, using the same prompt. 

**This is not implemented yet.**

## Edition
This section describes the main directories and files of the project:

- **.gitattributes**: Contains the attributes of the files and directories that are pushed to the repository.

- **.gitignore**: Contains the files and directories that are ignored by git.

- **config.ini**: Configuration file of the project, for variables used in, for example, fine-tuning and generation.

- **dataset.py**: Contains the class that is responsible for creating the dataset for the fine-tuning part of the pipeline and the tokenization of the data.

- **fetcher.py**: Contains the class that is responsible for fetching the data from the internet. Currently, it fetches data from the PoetryDB page.

- **LICENSE**: License of the project.

- **main.py**: Contains the main script of the project, responsible for running the pipeline.

- **Poet.py**: Contains the class that is responsible for fine-tuning the LLM models.

- **README.md**: Contains information about the project and instructions on how to install and run it.

- **requirements.txt**: Contains all the dependencies of the project that must be present in the virtual environment or machine running it.

- **writer.py**: Contains the class that is responsible for storing the poems 

## Future Work

- Implement the generation part of the pipeline.
- Implement the fine-tuning part of the pipeline for more models.
- Implement the fetching part of the pipeline for more sources.
- Implement a web application for the project using Django or Flask.