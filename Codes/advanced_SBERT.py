import re
import os
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from baseline_boolean import BooleanSearch
from typing import List, Set, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
QUERIES_PATH = PROJECT_ROOT / "Codes" / "queries.json"


class AdvancedMethod2:
    # SBERT-based semantic retieval using sentence embeddings
    def __init__(self, embed_df: pd.DataFrame, simi_thold: float= 0.6, m_name: str='all-MiniLM-L6-v2'):
        self.embed_df= embed_df
        self.simi_thold= simi_thold

        # Device selection, is simillar to how_to_use.py
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[SBERT] using device: {self.device}")

        # Loads the same SBERT model used to create the embedgins
        print(f"Loading sentence transformer model: {m_name}")
        self.model= SentenceTransformer(m_name, device= self.device)

        # Stacks all the embeddings into a big matrix: # sentences X dim
        # embeddings_df['embedding'] is a column of numpy arrays
        print(f"Builds the matrix")
        self.all_embeds= np.vstack(self.embed_df['embedding'].values)
        print(f"The embedding matrix shape: {self.all_embeds.shape}\n")
    
    def build_q_txt(self, asp: List[str], op: List[str]) -> str:
        # Converts aspect and opinion into a simple natural language sentence
        asp_phrase= ' '.join(asp).strip()
        op_phrase=' '.join(op).strip()

        if asp_phrase and op_phrase:
            return f"The {asp_phrase} of this product is {op_phrase}"
        elif asp_phrase:
            return f"This review talks abut the {asp_phrase} of the product"
        else:
            return f"This review expresses a {op_phrase} opinion of the product"
    
    def encode_q(self, q_txt: str) -> np.ndarray:
        # Encodes the query text into a single SBERT embedding vector
        embed= self.model.encode(
            [q_txt],
            convert_to_numpy= True,
            device= self.device
        )
        # Shape (1,dim) to flatten (dim,)
        return embed[0]

    def search(self, asp: List[str], op: List[str], simi_thold: float, top_k: int) -> Set[int]:
        # Semantic search over all setences in data.pkl
        thold= simi_thold if simi_thold is not None else self.simi_thold

        # Step 1: Build query text
        q_txt= self.build_q_txt(asp, op)

        # Step 2: Encode query
        q_embed= self.encode_q(q_txt).reshape(1,-1) # Shape (1, dim)

        # Step 3: Compute cosine similarity to all sentence embeds
        cos_simi= cosine_similarity(self.all_embeds, q_embed).ravel()

        # Step 4: Filter by similarity threshold
        indices= np.where(cos_simi >= thold)[0]

        if len(indices)==0:
            return set()
        
        # Sort in a descending order 
        scored_indices= [(idx, cos_simi[idx]) for idx in indices]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None and len(scored_indices) > top_k:
            scored_indices = scored_indices[:top_k]

        # Step 5: Collects the review IDs
        review_ids: Set[int] = set()
        for idx, score in scored_indices:
            doc_id= self.embed_df.loc[idx, 'document_id']
            review_ids.add(doc_id)
        
        return review_ids
    
    def save_results(self, review_ids: Set[int], output_file: str):
        sorted_ids = sorted(list(review_ids)) 
        with open(output_file, 'w') as f:
            for rid in sorted_ids:
                clean_id = str(rid).replace("'", "").strip() 
                f.write(f"{clean_id}\n")
    
def load_qs(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)
    
def run_m2(simi_thold: float, top_k: int) -> Dict[str, int]:
    out_dir = OUTPUTS_DIR / "AdvancedMethod2"
    out_dir.mkdir(parents=True, exist_ok=True)

    qs= load_qs(QUERIES_PATH)

    # Load embeddings DataFrame
    embed_path= DATA_DIR / "data.pkl"
    embed_df= pd.read_pickle(embed_path)

    # Initialize SBERT engine
    sbert_engine= AdvancedMethod2(
        embed_df=embed_df,
        simi_thold=simi_thold
    )
    results_summary: dict[str, int] = {}
    
    for q_name, terms in qs.items():
        asp_words= terms['aspect']
        op_words= terms['opinion']

        review_ids= sbert_engine.search(
            asp_words,
            op_words,
            simi_thold=simi_thold,
            top_k=top_k
        )

        out_file= out_dir / f"{q_name}_test4.txt"
        sbert_engine.save_results(review_ids, out_file)

        results_summary[q_name]= len(review_ids)
    
    # Summary 
    print("Advanced Method 2: Summary")
    for q_name, count in results_summary.items():
        print(f"{q_name}: {count} reviews retrieved")

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_pickle(DATA_DIR / "reviews_segment.pkl")

    # Initialize booleanSearch with inverted index
    engine= BooleanSearch(df, use_index=True) 

    run_m2(simi_thold=0.60, top_k=300)
