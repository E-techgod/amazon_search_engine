# Amazon Review Opinion Search Engine

## Project Overview
This project implements an opinion-oriented search engine for Amazon product reviews.

## Project Structure

```
amazon_search_engine/
├── Codes/
│   ├── baseline.py            # Boolean search implementation (Tests 1-3)
│   ├── advanced1.py           # First advanced method
│   ├── advanced2.py           # Second advanced method
│   ├── inverted_index.py      # Inverted index builder/handler
│   └── queries.json           # Query definitions for testing
├── Data/
│   ├── data.pkl               # Processed dataset
│   ├── reviews_segment.pkl    # Original Amazon reviews dataset
│   ├── inverted_index.pkl     # Pre-built inverted index
│   ├── lm.pkl                 # Logistic regression model (trained sentiment classifier)
│   ├── positive_words.txt     # Positive opinion lexicon
│   ├── negative_words.txt     # Negative opinion lexicon
│   ├── negation_words.txt     # Negation words list
│   └── intensifier_words.txt  # Intensifier words list
├── Outputs/
│   ├── Baseline/ {aspect_words}_test{number}.txt
│   ├── AdvancedMethod1/ {aspect}_test4.txt
│   └── AdvancedMethod2/ {aspect}_test4.txt
├
├── requirements.txt           # Python dependencies
└── Readme.md                  # This file
```
### 📁 Data Folder
The large `reviews_segment.pkl` and `data.pkl` files are **not included** in this repository due to their size.
To run the full pipeline, place these files inside the `Data/` directory before executing the scripts.
Data/
├── data.pkl
├── reviews_segment.pkl


### Python Dependencies
See `requirements.txt` for complete list. Key dependencies include:

### Step 1: Navigate to Project Directory
cd amazon_search_engine

### Step 2: Create Virtual Environment if not created

# Create virtual environment
python -m venv venv

# Activate virtual environment if not activated
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

## Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Download NLTK Data (if needed)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

## Dataset Setup

Place the dataset files in the `Data/` directory:
- `reviews_segment.pkl`: Main review dataset (210,761 reviews)
- `data.pkl`: Precomputed embeddings (for Advanced Method 2)
- Opinion lexicons: Positive/negative word lists
- Cached indices: Pre-built inverted index for faster retrieval (optional)

## Usage

### Running the Baseline Method (Tests 1-3)
python Codes/baseline_boolean.py

This script:
- Loads queries from `queries.json`
- Implements Boolean search for Tests 1-3
- Generates output files in `Outputs/...`

## Test Types:
- Test 1: Aspect
- Test 2: Aspect AND Opinion 
- Test 3: Aspect OR Opinion
- Test 4: Connotation 

### Running Advanced Method 1
python Codes/advanced_language_model.py

### Running Advanced Method 2
python Codes/advanced_SBERT.py

## Query Format
Queries are defined in `queries.json`:

## Output Format

### Output Files
All output files are saved in `Outputs/` directory.
{aspect_words}_test{number}.txt

Examples:
- `audio_quality_test1.txt` (Baseline Test 1)
- `audio_quality_test2.txt` (Baseline Test 2)
- `audio_quality_test3.txt` (Baseline Test 3)
- `audio_quality_test4.txt` (Advanced method)

### File Content
Each line contains a single review ID:

## Evaluation Queries

The project is evaluated on these 5 queries:

1. audio quality: poor
2. wifi signal: strong
3. mouse button: click problem
4. gps map: useful
5. image quality: sharp

All queries are defined in `queries.json`.


## Acknowledgments

- Dataset: Amazon Review Dataset (Course provided)
- Opinion Lexicon: Hu and Liu (KDD 2004)
- data.pkl: Courtesy of Navid Ayoobi

## Me 
Elias Arellano Campos  
