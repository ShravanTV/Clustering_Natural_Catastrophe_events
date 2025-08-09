import re
import string
import pandas as pd
from spellchecker import SpellChecker
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS

# Download NLTK resources before using stopwords/tokenizers
# import nltk
# nltk.download('all')

# Main cleaning class
class DataCleaner:
    """
    DataCleaner provides a suite of text preprocessing methods for NLP/data cleaning pipelines.
    Supports HTML removal, emoji/URL stripping, case normalization, contraction/acronym expansion,
    punctuation/special char removal, spell check, and more. Designed for use in large-scale pipelines.
    """
    
    def __init__(self, acronyms_dict=None, contractions_dict=None, acronyms_url=None, contractions_url=None):
        """
        Initialize DataCleaner with optional acronyms/contractions dictionaries or URLs.
        Loads NLTK stopwords, tokenizer, and spellchecker.
        """
        contractions_url = 'https://raw.githubusercontent.com/ShravanTV/Natural_Catastrophe_Events/refs/heads/main/Contractions_lowercase.json'
        acronyms_url = 'https://raw.githubusercontent.com/ShravanTV/Natural_Catastrophe_Events/refs/heads/main/abbrevations.json'

        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.spell = SpellChecker()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))

        # Load acronyms from URL if provided
        if acronyms_url:
            try:
                self.acronyms_dict = pd.read_json(acronyms_url, typ='series').to_dict()
            except Exception:
                self.acronyms_dict = acronyms_dict if acronyms_dict is not None else {}
        else:
            self.acronyms_dict = acronyms_dict if acronyms_dict is not None else {}

        # Load contractions from URL if provided
        if contractions_url:
            try:
                self.contractions_dict = pd.read_json(contractions_url, typ='series').to_dict()
            except Exception:
                self.contractions_dict = contractions_dict if contractions_dict is not None else {}
        else:
            self.contractions_dict = contractions_dict if contractions_dict is not None else {}


    # Removing punctuations
    def remove_punctuation(self, text):
        """
        Remove punctuation from text, keeping apostrophes for handling contractions in later stages.
        Args:
            text (str): Input string.
        Returns:
            str: Text with punctuation removed.
        """
        self.logger.debug("Removing punctuation.")
        
        punct_str = string.punctuation.replace("'", "") # discarding apostrophe from the string to keep the contractions intact
        return text.translate(str.maketrans("", "", punct_str))


    # Removing news source after Pipe symbol (|)
    def remove_news_source(self, text):
        """
        Remove the trailing news source after the last '|' if it looks like a news source.
        
        Recognizes common news keywords and domain endings including '.com' and 'dot com'.
        
        Args:
            text (str): Input string.
            
        Returns:
            str: Cleaned string without trailing news source.
        """
        self.logger.debug("Removing news source.")

        news_keywords = [
            "tribune", "journal", "gazette", "times", "post", "daily", "herald",
            "observer", "review", "report", "sun", "star", "news", "press", "bulletin",
            "the arkansas democrat", "tahoedailytribune", "nytimes", "ktvu",
            "indian television"  
        ]
        domain_pattern = re.compile(r'\.\s*(com|net|org|tv|info|co|us)\b', re.IGNORECASE)
        spelled_domain_pattern = re.compile(r'dot\s*(com|net|org|tv|info|co|us)\b', re.IGNORECASE)

        def normalize(segment):
            # Lowercase, remove special chars except dot and space, fix spaced dots
            segment = re.sub(r'\s*\.\s*', '.', segment.lower())
            segment = re.sub(r'[^a-z0-9. ]+', '', segment)
            return segment

        def is_news_source(segment):
            norm = normalize(segment)
            if domain_pattern.search(norm):
                return True
            if spelled_domain_pattern.search(norm):
                return True
            return any(keyword in norm for keyword in news_keywords)

        parts = [p.strip() for p in text.split('|')]
        if len(parts) > 1 and is_news_source(parts[-1]):
            return ' | '.join(parts[:-1])
        return text


    # Removing structured dates from the title
    def remove_structured_dates(self, text):
        """
        Removes structured date formats and weekday names from text.
        
        Examples removed:
            - '29 January 2024'
            - 'January 30, 2024'
            - '(31 Jan 2024)'
            - 'Tuesday'
            - 'Tuesday, January 30, 2024'
            - '16 : 50 UTC'
        
        Args:
            text (str): Input string.
        Returns:
            str: Cleaned text.
        """
        self.logger.debug("Removing structured dates.")

        months = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?" \
                r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

        weekdays = r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"

        patterns = [
            # Remove full date formats
            rf"\b\d{{1,2}}\s+{months}\s+\d{{4}}\b",                              # e.g. 29 January 2024
            rf"\b{months}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*\d{{4}}\b",         # e.g. January 30, 2024
            rf"\(\s*\d{{1,2}}\s+{months}\s+\d{{4}}\s*\)",                        # e.g. (31 Jan 2024)
            rf"\b{months}\s+\d{{1,2}}(?:st|nd|rd|th)?\s+\d{{4}}\b",              # e.g. January 31st 2024
            r"\b\d{1,2}\s*:\s*\d{2}\s*UTC\b",                                    # e.g. 16 : 50 UTC

            # Remove standalone weekday names (with optional comma)
            rf"\b{weekdays},?\b",                                               # e.g. Tuesday
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up spacing and punctuation
        text = re.sub(r'\s*,\s*,+', ', ', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s*([,;:\-])\s*', r' \1 ', text)
        return text.strip(" ,;- ")


    # Removing HTML tags
    def remove_html(self, text):
        """
        Remove HTML tags from text.
        Args:
            text (str): Input string.
        Returns:
            str: Text with HTML tags removed.
        """
        self.logger.debug("Removing HTML tags.")
        
        html = re.compile(r'<.*?>')
        return html.sub('', text)
    
    # Removing emojis
    def remove_emoji(self, text):
        """
        Remove emojis from text using unicode ranges.
        Args:
            text (str): Input string.
        Returns:
            str: Text with emojis removed.
        """
        self.logger.debug("Removing emojis.")
        
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    

    # Removing URLs    
    def remove_urls(self, text):
        """
        Remove URLs from text, including http(s), www, and bare domains.
        Args:
            text (str): Input string.
        Returns:
            str: Text with URLs removed.
        """
        self.logger.debug("Removing URLs.")
        
        # Combined regex for different URL forms
        pattern = r"""(
            https?://\S+ |                  # URLs starting with http:// or https://
            www(?:\s*\.\s*\S+)+ |           # Space-separated www URLs like www . domain . com
            \b(?:[a-z0-9-]+\.)+[a-z]{2,}\b  # Bare domains like kfyi.iheart.com
        )"""
        return re.sub(pattern, "", text, flags=re.IGNORECASE | re.VERBOSE).strip()
    
    
    # Converting to lowercase
    def to_lower(self, text):
        """
        Convert text to lowercase.
        Args:
            text (str): Input string.
        Returns:
            str: Lowercase text.
        """
        self.logger.debug("Converting to lowercase.")
        
        return text.lower()
    
    # Removing extra whitespace
    def remove_extra_whitespace(self, text):
        """
        Remove extra whitespace from text.
        Args:
            text (str): Input string.
        Returns:
            str: Text with normalized whitespace.
        """
        self.logger.debug("Removing extra whitespace.")
        
        return re.sub(r'\s+', ' ', text).strip()
    

    # Remove special characters (keep alphanumeric and spaces)
    def remove_special_characters(self, text):
        """
        Remove special characters, retaining alphanumeric and spaces.
        Args:
            text (str): Input string.
        Returns:
            str: Cleaned text.
        """
        self.logger.debug("Removing special characters.")
        
        return re.sub(r'[^A-Za-z0-9\s]', '', text)
    

    # Discarding non-alphabetic tokens (In our case we dont need numbers)
    def discard_non_alpha(self, text):
        """
        Discard non-alphabetic tokens from text.
        Args:
            text (str): Input string.
        Returns:
            str: Text with only alphabetic tokens.
        """
        self.logger.debug("Discarding non-alphabetic tokens.")
        
        tokens = self.tokenizer.tokenize(text)
        alpha_tokens = [word for word in tokens if word.isalpha()]
        return ' '.join(alpha_tokens)
    

    # Expand contractions
    def expand_contractions(self, text):
        """
        Expand contractions in text using provided contractions dictionary.
        Args:
            text (str): Input string.
        Returns:
            str: Text with contractions expanded.
        """
        self.logger.debug("Expanding contractions.")
        
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in self.contractions_dict.keys()) + r')\b', flags=re.IGNORECASE)
        def replace(match):
            contraction = match.group(0)
            expanded = self.contractions_dict.get(contraction.lower())
            # If not found, return as is
            return expanded if expanded else contraction
        return pattern.sub(replace, text)


    # Expand acronyms
    def expand_acronyms(self, text):
        """
        Expand acronyms in text using provided acronyms dictionary.
        Args:
            text (str): Input string.
        Returns:
            str: Text with acronyms expanded.
        """
        self.logger.debug("Expanding acronyms.")
        
        if not self.acronyms_dict:
            return text
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in self.acronyms_dict.keys()) + r')\b', flags=re.IGNORECASE)
        def replace(match):
            acronym = match.group(0)
            expanded = self.acronyms_dict.get(acronym.lower())
            return expanded if expanded else acronym
        return pattern.sub(replace, text)


    # Clean basictext pipeline
    def clean_text(self, text, steps=None):
        """
        Clean text using a sequence of steps. If steps is None, applies all steps in standard order.
        Args:
            text (str): Input string.
            steps (list, optional): List of method names as strings. If None, uses default pipeline.
        Returns:
            str: Cleaned text.
        """
        self.logger.debug(f"Cleaning text: {text[:30]}..." if text else "Cleaning empty text.")
        
        if text is None:
            return ''
        if steps is None:
            steps = [
                'to_lower',
                'remove_news_source',
                'remove_structured_dates',
                'remove_urls',
                'remove_html',
                'remove_emoji',
                'remove_special_characters',
                'remove_extra_whitespace',
                'expand_acronyms',
                'expand_contractions'
            ]
        for step in steps:
            text = getattr(self, step)(text)
        return text


# Example usage:
# cleaner = DataCleaner()
# df['cleaned_col'] = df['raw_col'].apply(cleaner.clean_text)
