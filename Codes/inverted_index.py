import re
import os
import pickle 
import pandas as pd
from typing import Dict, Set, List
from collections import defaultdict

class InvertedIndex:
    # For a fast term lookup
    def __init__(self):
        self.index= defaultdict(set)
        self.review_count=0

    def build_idx(self, reviews_df, text_column= 'lemmatized_text'):
        # Build inverted index
        print("Creating inverted index")
        for idx, row in reviews_df.iterrows():
            review_id= row['review_id']
            text= row.get(text_column, '')

            if not text or pd.isna(text):
                continue

            words= text.split() # Extract words from the review and clean them 

            for word in set(words): # This way we avoid duplicates in the same review
                self.index[word].add(review_id)
            self.review_count+=1

            if (idx+ 1) % 1000 ==0: # check the process 
                print(f"Indexed {idx +1} reviews")
        print(f"Inverted index built with {len(self.index)} unique terms")
        print(f"Indexed {self.review_count} reviews")

    def search_term(self, term: str) -> Set[int]:
        # Find all the reviews Ids that contains the given term
        return self.index.get(term, set())
    
    def search_or(self, terms: List[str]) -> Set[int]:
        # Finds reviews containing the OR operand
        if not terms:
            return set()
        
        result= set()
        for term in terms:
            result.update(self.search_term(term))
        return result
    
    def search_and(self, terms: List[str]) -> Set[int]:
        # Find reviews containing the AND operand
        if not terms:
            return set()
        result= self.search_term(terms[0]) 
        for term in terms[1:]:
            result= result.intersection(self.search_term(term)) # Intersect with the reviews that contain each term
            if not result: 
                break # Terminates early if there are no more reviews left 
        return result
    
    def search_aspect_and_opinion(self, aspects: List[str], opinions: List[str]) -> Set[int]:
        # Find reviews that contain at least one aspect and one opinion: (aspect1 Or aspect2) AND (opinion1 OR opinion2)
        aspect_reviews= self.search_or(aspects)
        opinion_reviews= self.search_or(opinions)

        return aspect_reviews.intersection(opinion_reviews)
    
    def save_idx(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': dict(self.index),
                'review_count': self.review_count
            }, f)
        print (f"Inverted index saved in {filepath}")

    def load_idx(self, filepath: str):
        with open(filepath, 'rb') as f:
            data= pickle.load(f)
            self.index= defaultdict(set, data['index'])
            self.review_count= data['review_count']
        print(f"It contains {len(self.index)} unique terms across {self.review_count} reviews")


    