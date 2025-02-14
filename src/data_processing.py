from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []

    with open(infile, 'r') as file:
        for line in file:
            if not line:
                continue
            parts = line.strip().rsplit('\t', maxsplit = 1)
            if len(parts) != 2:
                continue
            sentence, label = parts
            words = tokenize(sentence)

            example = SentimentExample(words, int(label))
            examples.append(example)

    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # Count unique words in all the examples from the training set
    
    unique_words = set()

    for example in examples:
        unique_words.update(example.words)

    vocab: Dict[str, int]  = {word: idx for idx, word in enumerate(sorted(unique_words))}


    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.º
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # Initialize a vector of zeros with the size of the vocabulary
    bow = torch.zeros(len(vocab), dtype=torch.float32)

    # Iterate through the words in the text
    for word in text:
        if word in vocab:
            index = vocab[word]  # Get the index of the word in the vocabulary
            if binary:
                bow[index] = 1  # Mark the word's presence as 1 if binary is True
            else:
                bow[index] += 1  # Increment the count if binary is False (frequency-based)


    return bow
