"""
natcat_detection.py

Provides NatCatEventDetector: a class for detecting natural catastrophe events in text using a transformer model and custom logic.
Includes multiprocessing support for DataFrame column inference.

Usage:
    from natcat_detection import NatCatEventDetector
    detector = NatCatEventDetector()
    result_df = detector.parallel_detect_df(df, text_col='title', new_col='is_natcat')

"""

import logging
import pandas as pd
from pandarallel import pandarallel
from typing import Optional
import threading
from functools import lru_cache

# Thread-local storage for models
_thread_local = threading.local()

def _get_models():
    """Get or initialize models for current thread/process."""
    if not hasattr(_thread_local, 'models_initialized'):
        import spacy
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        _thread_local.nlp = spacy.load("en_core_web_trf") # Transformer-based model to detect locations
        _thread_local.model_name = "hannybal/disaster-twitter-xlm-roberta-al" # Transformer-based model to detect natural catastrophe events
        _thread_local.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
        _thread_local.model = AutoModelForSequenceClassification.from_pretrained(_thread_local.model_name)
        _thread_local.models_initialized = True
        
    return _thread_local.nlp, _thread_local.tokenizer, _thread_local.model

class NatCatEventDetector:
    """
    Detects natural catastrophe events in text using a transformer model and custom rules.
    Supports multiprocessing for DataFrame application with lazy model loading.
    """
    def __init__(self, threshold: float = 0.7, verbose: bool = False):
        """
        Args:
            threshold (float): Confidence threshold for model.
            verbose (bool): Print debug info.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.threshold = threshold
        self.verbose = verbose
        self.LABELS = {0: "non-disaster", 1: "disaster"}
        self.NATURAL_CATASTROPHE_TYPES = {
            'earthquake', 'flood', 'lava', 'volcano', 'eruption', 'wildfire',
            'tornado', 'cyclone', 'hurricane', 'typhoon', 'tsunami',
            'landslide', 'drought', 'storm', 'blizzard', 'avalanche',
            'heatwave', 'lightning'
        }
        self.EXCLUSION_TERMS = {
            'drill', 'exercise', 'simulation', 'mock', 'test'
        }
        self.past_event_indicators = {
            'struck', 'hit', 'occurred', 'erupted', 'caused', 'killed',
            'damaged', 'destroyed', 'swept', 'triggered', 'sparked',
            'flooded', 'burned', 'ravaged', 'wreaked', 'devastated',
            'reported', 'identified'
        }

    def is_disaster_title(self, text: str) -> bool:
        import torch
        import torch.nn.functional as F
        
        # Get models for current thread/process
        _, tokenizer, model = _get_models()
        
        tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**tokenized).logits
            probs = F.softmax(logits, dim=-1)
            score, pred = torch.max(probs, dim=1)
        is_dis = (pred.item() == 1 and score.item() >= self.threshold)
        if self.verbose:
            print(f"Title: {text}")
            print(f"Prediction: {self.LABELS[pred.item()]}, Confidence: {score.item():.2f}")
        return is_dis

    def is_nat_cat_event(self, title: str) -> bool:
        """
        Detects natural catastrophe events in text using a transformer model and custom rules.
        - If the transformer model predicts a natural catastrophe event, it is further verified using custom rules.
        - Checks for location 
        - Checks for past/present-tense event.
        - If its a natural catastrophe event, having location and past/present-tense event then the title is marked as natural catastrophe event.
        Args:
            title (str): Input text to analyze.
        Returns:
            bool: True if the text is a natural catastrophe event, False otherwise.
        """
        lower_title = title.lower()
        
        # Get models for current thread/process
        nlp, _, _ = _get_models()
        doc = nlp(lower_title)
        
        # Exclusion filter
        if any(term in lower_title for term in self.EXCLUSION_TERMS):
            if self.verbose:
                print("Filtered due to exclusion term")
            return False
        # Model prediction
        disaster_detection_model = self.is_disaster_title(lower_title)
        if not disaster_detection_model:
            if self.verbose:
                print("Natural catastrophe event not detected through transformer model.")
            # Try custom keyword logic as fallback
            disaster_detection_custom = any(word in lower_title for word in self.NATURAL_CATASTROPHE_TYPES)
            if not disaster_detection_custom:
                if self.verbose:
                    print("Natural catastrophe event not detected in custom keyword.")
                return False
        else:
            # If model detects, still require keyword presence for stricter logic
            disaster_detection_custom = any(word in lower_title for word in self.NATURAL_CATASTROPHE_TYPES)
            if not disaster_detection_custom:
                if self.verbose:
                    print("Natural catastrophe event not detected in custom keyword.")
                return False
        # Rule-based: check for location
        has_location = any(ent.label_ in ['GPE', 'LOC'] for ent in doc.ents)
        # Rule-based: check for past/present-tense event
        past_event = any(token.lemma_.lower() in self.past_event_indicators for token in doc)
        if not past_event:
            past_event = any(token.tag_ in ['VBD', 'VBN', 'VBZ'] for token in doc)
        return has_location and past_event

    # @staticmethod
    # def _detect_worker(args):
    #     title, detector_args = args
    #     try:
    #         detector = NatCatEventDetector(**detector_args)
    #         return detector.is_nat_cat_event(title)
    #     except Exception as e:
    #         logging.getLogger("NatCatEventDetector").error(f"Error in detection: {e}")
    #         return False

    # def parallel_detect_df(self, df: pd.DataFrame, text_col: str, new_col: str = 'is_natcat') -> pd.DataFrame:
    #     """
    #     Detect natural catastrophe events in a DataFrame text column in parallel using pandarallel.
    #     Args:
    #         df (pd.DataFrame): Input DataFrame.
    #         text_col (str): Name of column to analyze.
    #         new_col (str): Name of output column.
    #     Returns:
    #         pd.DataFrame: DataFrame with new boolean column for nat-cat event detection.
    #     """
    #     if text_col not in df.columns:
    #         self.logger.error(f"Column '{text_col}' not found in DataFrame.")
    #         raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    #     pandarallel.initialize(progress_bar=True, nb_workers=self.max_workers or 4)
    #     df = df.copy()
    #     df[new_col] = df[text_col].parallel_apply(self.is_nat_cat_event)
    #     self.logger.info(f"Added column '{new_col}' with nat-cat event detection results using pandarallel.")
    #     return df
