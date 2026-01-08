import re 
import os 
import json
from pathlib import Path
import nltk
import pandas as pd
from typing import List, Set
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from inverted_index import InvertedIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
QUERIES_PATH = PROJECT_ROOT / "Codes" / "queries.json"


class BooleanSearch:
    def __init__ (self, reviews_df: pd.DataFrame, use_index: bool = True): 

        self.reviews_df = reviews_df # Store the reviews DataFrame
        self.lemmatizer= WordNetLemmatizer()  
        self.cleaned_cache= {} # Cache for cleaned words
        self.lemma_cache= {}  # Cache lemmatized words 
        self.use_index= use_index
        print(f"Starting engine with {len(reviews_df)} reviews")

        # Pre process if not already present 
        if 'clean_review_text' not in self.reviews_df.columns: 
             self.reviews_df['clean_review_text']= self.reviews_df['review_text'].apply(self.clean_text)
        if 'lemmatized_text' not in self.reviews_df.columns:
            self.reviews_df['lemmatized_text']= self.reviews_df['clean_review_text'].apply(self.lemmatize_text)

        # Initialize the inverted index
        if self.use_index:
            self.index= InvertedIndex()
            index_path = DATA_DIR / "inverted_index.pkl"
            if index_path.exists():
                self.index.load_idx(str(index_path))
            else: 
                print("Building inverted index")
                self.index.build_idx(self.reviews_df, 'lemmatized_text')
                self.index.save_idx(str(index_path))
        else:
            self.index= None            

        print("Pre-processed and lemmatized all review text for faster searching\n")
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        text= text.lower() # convert to lowercase
        text= re.sub(r'[^a-z0-9\s]', ' ', text) # remove special characters 
        text=re.sub(r'\s+', ' ', text).strip() # remove extra space 

        return text # return cleaned text ex. "this is a great review"
    
    def get_cleaned_words(self, words: List[str]) -> List[str]:
        # Cache cleaned words to avoid redundant processing
        key = tuple(words)

        if key not in self.cleaned_cache:
            self.cleaned_cache[key]= [self.clean_text(word) for word in words]

        return self.cleaned_cache[key]
    
    def lemmatize_text(self, text: str) -> str:
        # Lemmatize all the words in the text
        if not text:
            return ""
        
        words= text.split()
        lemmatized= [
            self.lemmatizer.lemmatize(
                self.lemmatizer.lemmatize(word, pos='v'),
                pos='n'
            ) 
            for word in words
        ] 
        
        return ' '.join(lemmatized) # Join the lemmatized words back into a single cleaned string separated by spaces
    
    def get_lemmatized_word(self, words: List[str]) -> List[str]:
        # Catche lemmatized words to avoid redundance 
        key= tuple(words)
        
        if key not in self.lemma_cache:
            self.lemma_cache[key]= [self.clean_text(word) for word in words]
            lemmatized= []

            for word in self.lemma_cache[key]:
                # Lemmatize the word as verb and noun 
                verb_lemma= self.lemmatizer.lemmatize(word, pos='v')
                noun_lemma= self.lemmatizer.lemmatize(word, pos='n')

                # Prefer the verb lemma if it changes the word 
                if verb_lemma != word: 
                    lemmatized.append(verb_lemma)
                elif noun_lemma != word:
                    lemmatized.append(noun_lemma)
                else: 
                    lemmatized.append(word)

            self.lemma_cache[key]= lemmatized 

        return self.lemma_cache[key]

    def t1_aspect(self, aspect: List[str]) -> Set[int]:
        # Reviews must contain at least one aspect word: Audio OR Quality
        aspect_cleaned= self.get_cleaned_words(aspect) # Clean aspect words
        aspect_lemmas= self.get_lemmatized_word(aspect_cleaned) # Lemmatize clean aspect words

        if not aspect_lemmas:
            return set()
        
        if self.index is not None: # OR operand logic, it should return any of the aspects
            matching_ids= self.index.search_or(aspect_lemmas)
            # print(f"Aspect Retrieval '{aspect_lemmas}': Found {len(matching_ids)} matching reviews")
            return matching_ids 
        
        return set()    
    
    def t2_aspect_and_opinion(self, aspect: List[str], opinion: List[str]) -> Set[int]:
       # Review must contain both an aspect and an opinion: (Audio OR Quality) AND (Poor)
        aspect_cleaned= self.get_cleaned_words(aspect) # Catch Clean aspect words
        opinion_cleaned= self.get_cleaned_words(opinion) # Catch Clean opinion words
        aspect_lemmas= self.get_lemmatized_word(aspect_cleaned) # Lemmatize clean aspect words
        opinion_lemmas= self.get_lemmatized_word(opinion_cleaned) # Lemmatize clean aspect words

        if not aspect_lemmas or not opinion_lemmas:
            return set()
        
        if self.index is not None: # OR operand logic, it should return any of the aspects
            matching_ids= self.index.search_aspect_and_opinion(aspect_lemmas, opinion_lemmas)
            # print(f"Aspect and Opinion Retrieval '{aspect_lemmas}' & '{opinion_lemmas}': Found {len(matching_ids)} matching reviews")
            return matching_ids 
        
        return set() 
    
    def t3_aspect_or_opinion(self, aspect: List[str], opinion: List[str]) -> Set[int]:
        # Review may have either an aspect or an opinion: (Audio OR Quality) OR (Poor)
        aspect_cleaned= self.get_cleaned_words(aspect) # Catch Clean aspect words
        opinion_cleaned= self.get_cleaned_words(opinion) # Catch Clean opinion words
        aspect_lemmas= self.get_lemmatized_word(aspect_cleaned) # Lemmatize clean aspect words
        opinion_lemmas= self.get_lemmatized_word(opinion_cleaned) # Lemmatize clean aspect words

        if not aspect_lemmas and not opinion_lemmas:
            return set()
        
        if self.index is not None: # OR operand logic, it should return any of the aspects
            both_terms= aspect_lemmas + opinion_lemmas
            matching_ids= self.index.search_or(both_terms)
            # print(f"Aspect or Opinion Retrieval '{aspect_lemmas}' | '{opinion_lemmas}': Found {len(matching_ids)} matching reviews")
            return matching_ids
        
        return set() 

    def save_results(self, review_ids: Set[int], output_file: str):
        sorted_ids = sorted(list(review_ids)) 

        with open(output_file, 'w') as f:
            for rid in sorted_ids:
                clean_id = str(rid).replace("'", "").strip() # Clean quotes/spaces
                f.write(f"{clean_id}\n")  # No single quotes around the ID
        # print(f"Saved {len(sorted_ids)} review IDs to {output_file}")

    def match_with_word_boundary(self, text: str, term: str) -> bool:
        """
        Return True if `term` appears in `text` as a whole word.
        Uses regex word boundaries to avoid partial matches (e.g., 'port' in 'airport').
        """
        if not text or not term:
            return False
        pattern = rf"\b{re.escape(term)}\b"
        return re.search(pattern, text) is not None

def load_queries(json_path: str):
        with open(json_path, "r") as f:
            return json.load(f)
        
def run_b(engine: BooleanSearch, out_dir: str= "Baseline"):
    out_path = OUTPUTS_DIR / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    queries= load_queries(QUERIES_PATH)

    for q_name, terms in queries.items():

        asp= terms['aspect']
        opn= terms['opinion']
        
        # Run all 3 tests
        t1 = engine.t1_aspect(asp)
        t2 = engine.t2_aspect_and_opinion(asp, opn) 
        t3 = engine.t3_aspect_or_opinion(asp, opn)
        
        # Save results
        for i, results in enumerate([t1, t2, t3], 1):
            out_file = out_path / f"{q_name}_test{i}.txt"
            engine.save_results(results, str(out_file))
        
        # Print summary
        print(f"\nSummary for {q_name}:\nTest1: {len(t1)}\nTest2: {len(t2)}\nTest3: {len(t3)}")

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_pickle(DATA_DIR / "reviews_segment.pkl")

    # Initialize booleanSearch with inverted index
    engine= BooleanSearch(df, use_index= True) 

    # Run the baseline
    run_b(engine, out_dir="Baseline")
