import spacy
from gensim.parsing.preprocessing import STOPWORDS
from nltk import regexp_tokenize
import threading

# Thread-local storage for spaCy model
_thread_local = threading.local()

def _get_nlp():
    """Get or initialize spaCy model for current thread/process."""
    if not hasattr(_thread_local, 'nlp'):
        _thread_local.nlp = spacy.load("en_core_web_sm")
    return _thread_local.nlp

# Custom stopwords – keep disaster-relevant keywords and prepositions
custom_stopwords = STOPWORDS.difference({
    'after', 'before', 'during', 'against', 'under', 'near', 'over',
    'between', 'while', 'within', 'through', 'until', 'without',
    "earthquake", "quake", "earthquakes",
    "flood","floods","flooding", "landslide", "landslides", "avalanche", "blizzard", "tide", "drought", "inundation", "deluge", "tsunami", "river",
    "tornado", "tornadoes", "storm", "hurricane", "cyclone", "typhoon", "lightning", "heatwave", "twister", "funnelcloud",
    "volcano", "volcanoes", "eruption", "lava", "ash",
    "wildfire", "wildfire", "wildfires", "fire","bushfire", "forestfire"
})
# Custom stopwords which need to be removed
add_stopwords = {
    'year', 'region'
}

# Final combined stopword list
final_stopwords = custom_stopwords.union(add_stopwords)

# Step 1: Lemmatize the text
def lemmatize_text(text):
    """
    Lemmatizes the text by removing stopwords and non-alphabetic words.
    Args:
        text (str): Input text to lemmatize.
    Returns:
        str: Lemmatized text.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    lemmatized = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmatized)

# Step 2: Remove stopwords AFTER lemmatization
def remove_custom_stopwords(text):
    return ' '.join(word for word in text.split() if word not in final_stopwords)

# Step 3: Remove non-alphabetic words (no digits, punctuation, etc.)
def discard_non_alpha(text):
    word_list_non_alpha = [word for word in regexp_tokenize(text, pattern=r'\w+|\$[\d\.]+|\S+') if word.isalpha()]
    return " ".join(word_list_non_alpha)

# Step 4: Remove locations from the titles
def remove_locations(text):
    """Remove location entities from text."""
    nlp = _get_nlp()
    doc = nlp(text)
    # Keep tokens that are not location entities
    tokens = [token.text for token in doc if not token.ent_type_ in ['GPE', 'LOC']]
    return ' '.join(tokens)

# Combine all steps into one function
def lemmatise_text(text):
    lemmatized = lemmatize_text(text)
    no_stopwords = remove_custom_stopwords(lemmatized)
    clean_text = discard_non_alpha(no_stopwords)
    no_locations = remove_locations(clean_text)
    return no_locations
    