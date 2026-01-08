import re
import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Set
from baseline_boolean import BooleanSearch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
QUERIES_PATH = PROJECT_ROOT / "Codes" / "queries.json"


class AdvancedMethod1: 
    # Sentiment-rating, enhanced boolean search (baseline)
    def __init__(self, reviews_df: pd.DataFrame, boolean_engine):
        self.reviews_df= reviews_df
        self.boolean_engine= boolean_engine

        # Access the inverted index
        self.inverted_index= getattr(boolean_engine, "index", None)

        # Loads opinion lexicons, Negation words, and Internsifer words
        self.pos_words= self.read_lexicon('positive')
        self.neg_words= self.read_lexicon('negative')
        self.negation_words= self.read_lexicon('negation')
        self.inten_words= self.read_lexicon('intensifier')

        # Initializes the classifier and vectorizer
        self.classi= None
        self.vec= None

    def read_lexicon(self, lexicon_file: str) -> Set[str]:
        
        """
        Lexicons come from here: 
        Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
        Proceedings of the ACM SIGKDD International Conference on Knowledge 
        Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
        Washington, USA, 
        """
        local_files = {
            'positive': DATA_DIR / "positive_words.txt",
            'negative': DATA_DIR / "negative_words.txt",
            'intensifier': DATA_DIR / "intensifier_words.txt",
            'negation': DATA_DIR / "negation_words.txt"
        }
        
        if lexicon_file.lower() not in local_files:
            raise ValueError(f"Sentiment must be type: {list(local_files.keys())}")
        file_path= local_files[lexicon_file]

        try:
            with open(file_path, 'r', encoding='UTF-8') as file:
                words= set()
                for line in file:
                    word = line.strip().lower()
                    # Skips comments and section markers
                    if word and not word.startswith(';') and not word.startswith('['):
                        words.add(word)
            return words
        
        except FileNotFoundError:
            print(f"Lexicon file not found: {file_path}")
            return set() # Empty set if zero files read
        except Exception as e:
            print(f"Error reading lexicon file: {file_path}")
            return set()
        
    def handle_negation(self, text: str) -> str:
        # Handles the negations by flipping the sentiment
        words= text.split() 
        result= []
        i=0

        while i < len(words):
            word= words[i]
            # Checks if current word is a negation word 
            if word in self.negation_words: 
                # Looks for sentiment words within window of 4 words this is Standard for a lexicon-based NLP project
                negation_scoop= min(4,len(words) -i -1)
                for j in range (1, negation_scoop + 1):
                    next_word= words[i+j]
                    # if the next word is a sentiment word, it'll negate it 
                    if next_word in self.pos_words or next_word in self.neg_words:
                        words[i+j]= f"negated_{next_word}"
                        break # Negates only the first sentiment word
            result.append(words[i]) # Add word to result 
            i += 1
        return ' '.join(result)

    def split_into_sentences(self, text:str) -> List[str]:
        # Split text into individual sentences for sentence-level analysis
        if pd.isna(text) or not text:
            return []
        # Splits on punctuation marks
        sentences= re.split(r'[.!?\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
 
    def detetermine_opinion_sentiment(self, op_words: List[str]) -> str:
        clean= [self.boolean_engine.clean_text(w) for w in op_words]
        lemmas= self.boolean_engine.get_lemmatized_word(clean)

        pos_count= sum(1 for word in lemmas if word in self.pos_words)
        neg_count= sum(1 for word in lemmas if word in self.neg_words)

        if neg_count > pos_count:
            return 'negative'
        elif pos_count > neg_count:
            return 'positive'
        else: 
            # Checks original words too
            pos_count= sum(1 for word in op_words if word in self.pos_words)
            neg_count= sum(1 for word in op_words if word in self.neg_words)

            if neg_count > pos_count:
                return 'negative'
            elif pos_count > neg_count:
                return 'positive'
            else: 
                return 'neutral'
            
    def sentiment_score(self, text: str) -> float:
        # Closer to -1 is very negative
        # Closer to 1 is very positive
        if pd.isna(text) or not text:
            return 0.0
        
        # Apply negation handling
        txt_w_negation= self.handle_negation(text) 
        words= txt_w_negation.lower().split()
        
        pos_count= 0
        neg_count= 0

        for i, word in enumerate(words):
            # Check if previous word is an intensifier to boost it 
            inten_multiplier= 1.0
            if i > 0 and words[i-1] in self.inten_words:
                inten_multiplier =1.5 # Boost by 50%

            # Handle negated words    
            if word.startswith('negated_'):
                actual_word= word[len('negated_'):]
                if actual_word in self.pos_words:
                    neg_count += inten_multiplier 
                elif actual_word in self.neg_words:
                    pos_count+=inten_multiplier  
            else:
                # Normal sentiment, no negations or intensifiers
                if word in self.pos_words:
                    pos_count += inten_multiplier
                elif word in self.neg_words:
                    neg_count += inten_multiplier

        total= pos_count + neg_count # Compute final score
        if total ==0:
            return 0.0 # Neutral words
        
        return (pos_count-neg_count) / total 
    
    def sentence_level_relevance(self, text: str, asp_words: List[str], op_words: List[str]) -> float:
        # Check if the aspect and the opinion occur in the same sentence
        if pd.isna(text) or not text:
            return 0.0
        
        sentences= self.split_into_sentences(text)
        if not sentences:
            return 0.0
    
        asp_lemmas= self.boolean_engine.get_lemmatized_word(
            self.boolean_engine.get_cleaned_words(asp_words)
        )
        op_lemmas= self.boolean_engine.get_lemmatized_word(
            self.boolean_engine.get_cleaned_words(op_words)
        )

        match_sentences=0
        for sentence in sentences:
            cleaned_sentence= self.boolean_engine.clean_text(sentence)

            has_asp= any(asp in cleaned_sentence.split() for asp in asp_lemmas)
            has_op= any(op in cleaned_sentence.split() for op in op_lemmas)

            if has_asp and has_op:
                match_sentences+=1

        total_sentences= len(sentences)
        relevant_score= match_sentences / total_sentences if len(sentences) > 0 else 0.0
        return relevant_score # Returns proportion of sentence where both appear together
    
    
    def proximity_score(self, text: str, asp_words: List[str], op_words: List[str]) -> float:
        # Calculates how close an aspect and an opinion word appear together, between 0,1
        if pd.isna(text) or not text:
            return 0.0
        
        words= text.lower().split()
        asp_lemmas= self.boolean_engine.get_lemmatized_word(
            self.boolean_engine.get_cleaned_words(asp_words)
        )
        op_lemmas= self.boolean_engine.get_lemmatized_word(
            self.boolean_engine.get_cleaned_words(op_words)
        )

        # Find positions where aspect and opinion words appear 
        asp_positions= [i for i, word in enumerate(words) 
                           if word in asp_lemmas] 
        op_positions= [i for i, word in enumerate(words) 
                            if word in op_lemmas] 

        if not asp_positions or not op_positions:
            return 0.0
        
        # Calculate min distance
        min_distance= min(
            abs(asp - op) 
            for asp in asp_positions 
            for op in op_positions
        )

        # Checks if there's an intensifier near the opinion words
        inten= False
        for op_pos in op_positions:
            # Checks word before opinion
            if op_pos > 0 and words[op_pos -1] in self.inten_words:
                inten= True
                break
            # Checks word after opinion
            if op_pos < len(words) -1 and words[op_pos +1] in self.inten_words:
                inten= True
                break

        # Convert distance to score, closer = a higher score
        if min_distance <= 3:
            distance_score= 1.0
        elif min_distance <= 7:
            distance_score= 0.8
        elif min_distance <= 15:
            distance_score= 0.5
        elif min_distance <= 25:
            distance_score =0.3
        else:
            distance_score= 0.1

        # Boost the opinon if an intensifier is present
        if inten:
            distance_score= min(1.0, distance_score * 1.2) # Gives 20% boost 

        return distance_score
        
    def title_sentiment_boost(self, review_title: str, op_sentiment: str) -> float:
        # Check if the review title matches the opinion sentiment 
        # Macthes if 1.5
        # Neutral if 1.0
        # No match if 0.5

        if pd.isna(review_title) or not review_title:
            return 1.0
        title_sentiment= self.sentiment_score(review_title.lower())
         
        if op_sentiment == 'positive':
            if title_sentiment > 0.1: # Threshol is lower for shorter titles 
                return 1.5 # Gives extra weight to the score 
            elif title_sentiment < -0.1: # Adds a penalty for contradictions 
                return 0.5
        elif op_sentiment == 'negative':
            if title_sentiment < -0.1:
                return 1.5
            elif title_sentiment > 0.1:
                return 0.5
    
        return 1.0 # If we do not know the query sentiment, becuase its neutral or none, we do nothing
        
    def final_scores(self, row: pd.Series, asp_words: List[str], op_words: List[str], q_sentiment: str) -> float:
        # Calculate relevance score combining title, proximity, sentence level, sentiment (negation) 
        text= row.get('lemmatized_text')
        title= row.get('review_title')

        prox_score= self.proximity_score(text, asp_words, op_words)
        sentence_score= self.sentence_level_relevance(text, asp_words, op_words)
        title_score= self.title_sentiment_boost(title, q_sentiment)
        sentiment_score= self.sentiment_score(text)

        # Align the query polarity with the sentiment 
        if q_sentiment == 'positive':
            aligned_sentiment= max(sentiment_score, 0.0) 
        elif q_sentiment == 'negative':
            aligned_sentiment= max(-sentiment_score, 0.0) 
        else:
            aligned_sentiment= abs(sentiment_score) 

        # Weighted combination
        final_score= (
            prox_score * 0.35 +  
            sentence_score * 0.35 +
            aligned_sentiment* 0.20 +
            title_score * 0.10 
        )

        return final_score
    
    def train(self, sample_size: int= 1000): 
        # Uses star ratings as labels (>3 positive, <=3 negative)
    
        df= self.reviews_df.copy()
        # Convert ratings to numeric once and keep the cleaned frame
        df['customer_review_rating']= pd.to_numeric(df['customer_review_rating'], errors='coerce')
        df= df.sample(
            n= min(sample_size, len(df)),
            random_state=42
        )

        # Drop rows with missing text or rating
        df= df.dropna(subset=['review_text', 'customer_review_rating'])

        # Create binary labels
        df['label']= (df['customer_review_rating']>3).astype(int)
        print(f"Training samples: {len(df)}")

        # Split into train and validation sets
        X_train, X_val, y_train, y_val= train_test_split(
            df['review_text'],
            df['label'],
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )

        # TF-IDF vectorization
        print("creating TF-IDF")
        self.vec= TfidfVectorizer(
            max_features=5000, # Limit vocab size
            min_df=2, # Word must appear in at least 2 docs
            max_df= 0.8,  # Ignore word if appears in >80% of docs
            ngram_range=(1,2), # Uses uni and bigrams
            stop_words='english'
        )

        X_train_vec= self.vec.fit_transform(X_train)
        X_val_vec= self.vec.transform(X_val)

        # Train classifier
        print("Training model")
        self.classi= LogisticRegression(
            max_iter=1000,
            random_state=42, 
            class_weight='balanced'
        )
        self.classi.fit(X_train_vec, y_train)

        # Evaluate on validation set
        y_pred= self.classi.predict(X_val_vec)
        acc= accuracy_score(y_val, y_pred)

        print(f"Validation Accuracy: {acc:.4f}")

    def predict_sentiment(self, review_text: str) -> tuple[int, float]:
        # Predicts the sentiment of a review using the trained classifier
        if self.classi is None or self.vec is None:
            raise ValueError("Classifier is not trained")
        
        # Vectorize the input text 
        text= self.vec.transform([review_text])
        # Get pred and prob
        pred= self.classi.predict(text)[0]
        prob= self.classi.predict_proba(text)[0]
        conf= prob[pred]

        return pred, conf

    def save_m(self, m_path: str= str(DATA_DIR / "lm.pkl")):
        os.makedirs(os.path.dirname(m_path), exist_ok=True)
        m_data= {
            'classifier': self.classi,
            'vectorizer': self.vec
        }
        with open(m_path, 'wb') as f:
            pickle.dump(m_data, f)
    
    def load_m(self, m_path: str= str(DATA_DIR / "lm.pkl")):
        if not os.path.exists(m_path):
            print("Training a new model")
            self.train()
            self.save_m(m_path)
            return 

        with open(m_path, 'rb') as f:
            data= pickle.load(f)
            self.classi= data['classifier']
            self.vec= data['vectorizer']

    def search(self,asp: List[str], op: List[str], prox_threshold: float= 0.3, use_classi: bool= True, classi_threshold: float = 0.6) ->Set[int]:
        # Advanced search with sentiment and rating flitering
        # Args: List of Aspect words, List of Opinion words, Wether to filter by rating consistency, Minimum proximity score to kepp review

        # Step 1: Fast candidate retrieval using the inverted index
        asp_lemmas= self.boolean_engine.get_lemmatized_word(
            self.boolean_engine.get_cleaned_words(asp)
        )
        op_lemmas= self.boolean_engine.get_lemmatized_word(
            self.boolean_engine.get_cleaned_words(op)
        )

        # Uses inverted index
        if self.inverted_index:
            candidates_id= self.inverted_index.search_aspect_and_opinion(asp_lemmas, op_lemmas)
        else: 
            candidates_id= set()
            print("No inverted index available")
            
        if len(candidates_id) ==0:
            return set()
        
        # Step 2: Determines the query sentiment 
        op_sentiment= self.detetermine_opinion_sentiment(op)

        # Step 3: Get candidate reviews dataFrame
        candidates_df= self.reviews_df[self.reviews_df['review_id'].isin(candidates_id)].copy()
        candidates_df['customer_review_rating']= pd.to_numeric(
            candidates_df['customer_review_rating'], 
            errors='coerce'
        )

        # Step 4: Use classfier
        if use_classi and self.classi is not None and op_sentiment != 'neutral':
            # Predict sentiment 
            predictions= []
            confidences= []
            
            for _, row in candidates_df.iterrows():
                pred, conf= self.predict_sentiment(row['review_text'])
                predictions.append(pred)
                confidences.append(conf)
            candidates_df['predicted_sentiment']= predictions
            candidates_df['classifier_confidence']= confidences

            # Filter based on the sentiment aligment and confidence
            if op_sentiment == 'positive':
                # Keep reviews predicted as positive with high confidence
                candidates_df= candidates_df[
                    (candidates_df['predicted_sentiment']==1) & 
                    (candidates_df['classifier_confidence'] >= classi_threshold)
                ]
            elif op_sentiment == 'negative':
                # Keep reviews predicted as negatie with high confidence
                candidates_df= candidates_df[
                    (candidates_df['predicted_sentiment']==0) & 
                    (candidates_df['classifier_confidence'] >= classi_threshold)
                ]
        else:
            print(f"Classifier is not trained")

        if len(candidates_df) ==0:
            return set()
        
        # Step 5: Relevance scores
        candidates_df['final_score']= candidates_df.apply(
            lambda row: self.final_scores(row, asp, op, op_sentiment),
            axis=1
        )

        # Step 6: Filter by score threshold
        candidates_df= candidates_df[candidates_df['final_score'] >= prox_threshold]
        return set(candidates_df['review_id'])
    
    def save_results(self, review_ids: Set[int], out_file: str):
        sorted_ids = sorted(list(review_ids)) 
        with open(out_file, 'w') as f:
            for rid in sorted_ids:
                clean_id = str(rid).replace("'", "").strip() 
                f.write(f"{clean_id}\n")

def load_q(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)
    
def run_m1(boolean_engine: BooleanSearch, reviews_df: pd.DataFrame, out_dir: str= "AdvancedMethod1"):
    out_path = OUTPUTS_DIR / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    qs= load_q(QUERIES_PATH)

    adv1_engine= AdvancedMethod1(reviews_df, boolean_engine)
    model_path= str(DATA_DIR / "lm.pkl")

    if os.path.exists(model_path):
        adv1_engine.load_m(model_path)
    else: 
        adv1_engine.train(sample_size=10000)
        adv1_engine.save_m(model_path)

    results_summary= {}

    for q_name, terms in qs.items():
        
        asp_words = terms['aspect']
        op_words = terms['opinion']

        # Run advanced search with the final score (composite score)
        result_ids= adv1_engine.search(
            asp_words, 
            op_words, 
            prox_threshold=0.3, 
            use_classi=True, 
            classi_threshold=0.6
        )

        # Saves the results for test 4
        out_file = out_path / f"{q_name}_test4.txt"
        adv1_engine.save_results(result_ids, out_file)

        results_summary[q_name]= len(result_ids)

    # Print Summary
    print(f"Advanced Method 1: Summary")
    for q_name, count in results_summary.items():
        print(f"{q_name}: {count} reviews retrieved")

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_pickle(DATA_DIR / "reviews_segment.pkl")

    # Initialize booleanSearch with inverted index
    engine= BooleanSearch(df, use_index=True) 

    run_m1(engine, df, out_dir="AdvancedMethod1")
