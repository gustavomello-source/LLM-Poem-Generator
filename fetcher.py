import requests
from typing import List, Dict, NoReturn, Any
from writer import Writer
import json
from tqdm import tqdm
import os

import os
import json
from tqdm import tqdm

def convert_json_to_txt(directory: str) -> None:
    '''
    Method to save the json poems found in directory as a txt file.

    Args:
        directory (str): The directory containing the json poems.
    Returns:
        None
    '''
    json_poems = []
    # Get the json files
    for json_poem in os.listdir(directory):
        if json_poem.endswith('.json'):
            json_poems.append(f"{directory}/{json_poem}")

    for json_poem in json_poems:
        if json_poem.endswith('.json'):
            # Load the json
            with open(json_poem, 'r', encoding='utf-8') as f:
                json_poem = json.load(f)

            processed_poems = []

            # Start processing the poems
            for poem in tqdm(json_poem, desc="Processing poems"):
                title = poem['title']
                author = poem['author']
                lines = poem['lines']

                # Format the poem
                formatted_poem = f"Title: {title}\nAuthor: {author}\n\n" + "\n".join(lines)

                # Add separators
                formatted_poem = f"{formatted_poem}\n\n===\n\n"

                processed_poems.append(formatted_poem)

            # Write the processed poems to a file in append mode
            with open('processed_poems.txt', 'a', encoding='utf-8') as f:
                f.writelines(processed_poems)
        
        else:
            raise ValueError("Poems must be a json file.")

def save_poems_as_json(poems: Any, filename: str, directory: str) -> json:
    '''
    Method to save the poems of the writer as a JSON file. 

    Args:
        filename (str): The name of the file.
        dir (str): The directory to save the file.
    Returns:
        JSON: The JSON file containing poems.
    '''
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{filename}.json", "w", encoding='utf-8') as file:
        json.dump(poems, file, indent=4,ensure_ascii=False)

def save_poems_as_txt(poems: json, filename:str, directory:str) -> None:
    '''
    Method to save the poems of the writer as a txt file.

    Args:
        poems (json): The poems of the writer.
        filename (str): The name of the file.
        dir (str): The directory to save the file.
    Returns:
        None
    '''
    if poems is json:
        # Load the json
        with open(poems, 'r', encoding='utf-8') as f:
            poems = json.load(f)

        processed_poems = []

        # Start processing the poems
        for poem in tqdm(poems, desc="Processing poems"):
            title = poem['title']
            author = poem['author']
            lines = poem['lines']

            # Format the poem
            formatted_poem = f"Title: {title}\nAuthor: {author}\n\n" + "\n".join(lines)

            # Add separators
            formatted_poem = f"{formatted_poem}\n\n===\n\n"

            processed_poems.append(formatted_poem)

        # Write the processed poems to a file
        with open('processed_poems.txt' , 'w', encoding='utf-8') as f:
            f.writelines(processed_poems)
    
    else:
        raise ValueError("Poems must be a json file.")

class PoetrydbFetcher:
    def __init__(self, url: str, verbose: bool = False) -> None:
        self.verbose = verbose
        self.url = url
        self.authors = self.get_authors()
        self.poems = self.get_poems(self.authors)
        print(f"\n<-------STARTING TO FETCH POEMS FROM POETRYDB------->")
        print(f"\nFound {len(self.authors)} authors.")
        for writer in Writer._instances.values():
            poems_json = save_poems_as_json(writer.poems, writer.name, 'Poetrydb')
        convert_json_to_txt('./Poetrydb')

    def get_authors(self) -> List[str]:
        '''
        Method to try getting the authors from the website.
        If the request fails, it will return an empty list and print the error.
        If the request is successful, it will return the list of authors and print them,
        and create a Writer object for each author.
        Args:
            None
        Returns:
            List[str]: The list of authors.
        '''
        try :
            content = requests.get(f'{self.url}/author').json()
        except requests.exceptions.RequestException as e:
            print(e)
            return []
        authors = []
        for author in content['authors']:
            authors.append(author)
            Writer(name=author)

        if self.verbose:
            print(f"\nFound {len(content['authors'])} Authors.\n\n---------------->Authors:")
            for author in authors:
                print(f"{author}")
        return authors

    def get_poems(self, authors: List[str]) -> List[Dict[str, str]]:
        '''
        Method to try getting the poems from the website.
        If the request fails, it will return an empty list and print the error.
        If the request is successful, it will return the list of poems and print them.
        Also, it will update the Writer object with the poems variable.
        Args:
            authors (List[str]): The list of authors.
        Returns:
            List[Dict[str, str]]: The list of poems.
        Raises:
            None
        '''
        poems = []
        for author in authors:
            writer_instance = Writer._instances.get(author)
            author = author.replace(' ', '%20')
            try:
                content = requests.get(f'{self.url}/author/{author}').json()
            except requests.exceptions.RequestException as e:
                print(e)
                return []
            
            if writer_instance:
                poems = content
                writer_instance.poems = poems
            else:
                print(f"Writer {author} not found.")

            if self.verbose:
                print(f"Found {len(content)} poems for {writer_instance.name}")
        return poems

    @property
    def verbose(self) -> bool:
        '''
        Method return the bool value of the verbose attribute.
        
        Returns:
            bool: The url of the website.
        '''
        return self.__verbose
    
    @verbose.setter
    def verbose(self, value: bool) -> NoReturn:
        '''
        Method receives the bool value of the verbose attribute and validates it.

        Args:
            value (bool): The bool value of the verbose attribute.
        Raises:
            None
        '''
        if value is None:
            value = False
        if not isinstance(value, bool):
            value = False
        self.__verbose = value

    @property
    def url(self) -> str:
        '''
        Method return the url of the website to be scraped.
        
        Returns:
            str: The url of the website.
        '''
        return self.__url
    
    @url.setter
    def url(self, value: str) -> NoReturn:
        '''
        Method receives the url of the website to be scraped and validates it.
        
        Args:
            value (str): The url of the website.
        Raises:
            ValueError: If the value is None or an empty string.
            ValueError: If the url does not start with 'http' or 'https'. This can be removed, since some of them do not have it.
            ValueError: If the url is not valid.
        '''
        if value is None or value == '':
            raise ValueError("URL cannot be None or an empty string.")
        if not value.startswith('http'):
            raise ValueError("URL must start with 'http' or 'https'.")
        if requests.get(value).status_code != 200:
            raise ValueError("URL is not valid.")
        self.__url = value

    @property
    def authors(self) -> List[str]:
        '''
        Method returns the list of authors.
        
        Returns:
            List[str]: The list of authors.
        '''
        return self.__authors

    @authors.setter
    def authors(self, value: List[str]) -> NoReturn:
        '''
        Method receives the list of authors and validates them.
        
        Args:
            value (List[str]): The list of authors.
        Raises:
            ValueError: If the value is None or an empty list.
        '''
        if value is None or len(value) == 0:
            raise ValueError("Authors cannot be None or an empty list.")
        self.__authors = value