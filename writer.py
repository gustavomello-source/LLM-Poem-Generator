from typing import List, Dict, Any, NoReturn
class Writer:
    '''
    Singleton Class to represent a writer.
    Contains the name of the writer and a list of poems.
    '''
    _instances = {}

    def __new__(cls, name: str):
        '''
        Factory method to create a new instance of the class.
        If the instance already exists, it will return it.
        If the instance does not exist, it will create it and return it.

        Args:
            name (str): The name of the writer.
        Returns:
            Writer: The instance of the writer.
        '''
        if name in cls._instances:
            return cls._instances[name]
        instance = super(Writer, cls).__new__(cls)
        cls._instances[name] = instance
        return instance
    
    def __init__(self, name: str) -> None:
        '''
        Initializes the class with the name of the writer and an empty list of poems.

        Args:
            name (str): The name of the writer.
        '''
        self.name = name
        self.normalized_name = name.lower().replace(' ', '%20')
        self.poems = []
        self._initialized = True

    def __str__(self) -> str:
        '''
        Method return the name of the writer.
        
        Returns:
            str: The name of the writer.
        '''
        return self.name

    def __repr__(self) -> str:
        return f"Writer(name='{self.name}')"

    @property
    def name(self) -> str:
        '''
        Method return the name of the writer.

        Returns:
            str: The name of the writer.
        '''
        return self.__name
    
    @name.setter
    def name(self, value: str) -> NoReturn:
        '''
        Method receives the name of the writer and validates it.

        Args:
            value (str): The name of the writer.
        Raises:
            ValueError: If the value is None or an empty string.
        '''
        if value is None or value == '':
            raise ValueError("Name cannot be None or an empty string.")
        self.__name = value
    
    @property
    def poems(self) -> list:
        '''
        Method return the list of poems of the writer.
        
        Returns:
            list: The list of poems of the writer.
        '''
        return self.__poems
    
    @poems.setter
    def poems(self, value: List[Dict[str, Any]]) -> NoReturn:
        '''
        Method receives the list of poems of the writer and validates it, adding default values if necessary.

        Args:
            value (List[Dict[str, Any]]): The list containing the poem's information.
        Raises:
            ValueError: If the value is None, an empty list.
        '''
        if value is None:
            raise ValueError("Poems cannot be None.")
        
        required_keys = {
            "title": "Untitled",
            "author": self.name,
            "lines": [],
            "linecount": "0"
        }

        poems = []

        for poem in value:
            if not isinstance(poem, dict):
                raise ValueError("Each poem must be a dictionary.")
            
            for key, default_value in required_keys.items():
                if key not in poem:
                    poem[key] = default_value
                    print(f"Added {key} to poem: {poem['title']}")

            if not isinstance(poem["title"], str) or not poem["title"]:
                poem["title"] = "Untitled"

            if not isinstance(poem["author"], str) or not poem["author"]:
                poem["author"] = self.name
            
            if not isinstance(poem["lines"], list) or len(poem["lines"]) == 0:
                raise ValueError("The 'lines' must be a list of strings.")                
            
            if not isinstance(poem["linecount"], str) or not poem["linecount"].isdigit():
                poem["linecount"] = str(len(poem["lines"]))
            
            poems.append(poem)
        
        self.__poems = poems