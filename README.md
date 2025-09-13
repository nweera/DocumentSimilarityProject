# Document Similarity Project

A comprehensive implementation of document similarity measurement using custom TF-IDF vectorization and cosine similarity calculation.

## 📋 Overview

This project measures the similarity between two pieces of text using custom implementations of tokenization, vectorization, and similarity calculation. The system processes raw text input and outputs a similarity score between -1 and 1, where values closer to 1 indicate higher similarity.

## 🎯 Core Concepts

The project implements three fundamental concepts:

1. **Text Preprocessing** - Cleaning and standardizing input text
2. **TF-IDF Vectorization** - Converting text to numerical representations
3. **Cosine Similarity** - Calculating similarity between document vectors

## 🔄 Workflow Overview

```
Input Texts → Preprocessing → Vocabulary Building → TF and IDF Calculation → Vectorization → Mapping → Cosine Similarity
```

### Step-by-Step Process:

1. **Input Texts**: Raw text documents to be compared
2. **Preprocessing**: Text cleaning and standardization
3. **Vocabulary Building**: Extract and analyze word frequencies
4. **TF and IDF Calculation**: Compute term frequency and inverse document frequency
5. **Vectorization**: Convert text to numerical vectors
6. **Mapping**: Transform vocabulary to numerical representations
7. **Cosine Similarity**: Calculate final similarity score

## 🛠️ Implementation Details

### 1. Text Preprocessing
- **Purpose**: Prepare texts by removing unwanted characters, converting to lowercase, and expanding contractions
- **Functions**: `decontract()` + `preprocess()`
- **Operations**:
  - Remove special characters and punctuation
  - Convert text to lowercase
  - Expand contractions (e.g., "don't" → "do not")

### 2. Vocabulary Building
- **Purpose**: Analyze text to determine word importance based on frequency
- **Function**: `word_frequency()`
- **Output**: Foundation for numerical representation in later steps

### 3. Word Mapping
- **Purpose**: Transform vocabulary words into numerical representations based on indexes
- **Function**: `create_mapping()`
- **Process**: Assigns unique numerical indices to each word in the vocabulary

### 4. TF-IDF (Term Frequency-Inverse Document Frequency)

#### a) Term Frequency (TF)
- Measures how frequently a term appears in a document
- Higher frequency indicates greater importance within the document
- Formula: `TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)`

#### b) Inverse Document Frequency (IDF)
- Measures term importance across the entire corpus
- Rare words have high IDF scores
- Common words have low IDF scores
- Formula: `IDF(t,D) = log(Total number of documents / Number of documents containing term t)`

#### c) TF-IDF Score
- **High scores**: Words important in a document and rare across the corpus
- **Low scores**: Common words that appear frequently across many documents
- **Formula**: `TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)`

### 5. Cosine Similarity
- **Purpose**: Measure similarity between two vectors based on the cosine of the angle between them
- **Formula**: `cosine_similarity = (A⋅B) / (∥A∥ × ∥B∥)`
  - `A⋅B`: Dot product of vectors A and B
  - `∥A∥` and `∥B∥`: Magnitudes of vectors A and B respectively

#### Output Interpretation:
- **1**: Perfect similarity (texts are identical in meaning)
- **0**: No similarity (texts are orthogonal)
- **-1**: Completely dissimilar (opposite meaning, rare in text applications)

## 🚀 Usage

```python
# Example usage (pseudocode)
from document_similarity import DocumentSimilarity

# Initialize the similarity calculator
doc_sim = DocumentSimilarity()

# Input texts
text1 = "Your first document text here"
text2 = "Your second document text here"

# Calculate similarity
similarity_score = doc_sim.calculate_similarity(text1, text2)

print(f"Similarity Score: {similarity_score}")
```

## 📊 Key Functions

- `decontract()`: Expand contractions in text
- `preprocess()`: Clean and standardize text
- `word_frequency()`: Build vocabulary and calculate word frequencies
- `create_mapping()`: Map words to numerical indices
- `calculate_tf_idf()`: Compute TF-IDF vectors
- `cosine_similarity()`: Calculate final similarity score



## 📝 Notes

- The implementation uses custom algorithms rather than external libraries for educational purposes
- All vectorization and similarity calculations are built from scratch
- The system is designed to handle various text preprocessing challenges
- Results are normalized between -1 and 1 for consistent interpretation

## 🔍 Applications

This document similarity system can be used for:
- Plagiarism detection
- Content recommendation systems
- Document clustering
- Information retrieval
- Text analysis and comparison tasks

---

*For detailed code implementation, please check the source code files.*
