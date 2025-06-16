# HF-RAG

This is the repository for the paper: `Analyzing the Role of Retrieval Models in Labeled and Unlabeled Context Selection for Fact Verification`.

This repository allows the replication of all results reported in the papers. In particular, it is organized as follows:
- [Prerequisites](#Prerequisites)
- [Data Preparation](#Data-Preparation)
  - [Data Preprocessing](#Data-Preprocessing)
  - [Corpus Preprocessing](#Corpus-Preprocessing)
- [How do I search?](#search)
  - [Single-stage Ranker](#Single-stage)
  - [Two-stage Ranker](#Two-stage)
- [Post-processing](#post-processing)
  - [For Single-stage](#forSingle-stage)
  - [For Two-stage](#forTwo-stage)
-  [Replicating Results](#Replicating-Results)
    - [L-RAG](#L-RAG)
    - [U-RAG](#U-RAG)
    - [LU-RAG](#LU-RAG)

## Prerequisites
We recommend running all the things in a Linux environment. 
Please create a conda environment with all required packages, and activate the environment by the following commands:
```
$ conda create -n pyserini_env python==3.10
$ conda activate pyserini_env
```
We have used [pyserini](https://github.com/castorini/pyserini) for each search. 

Install via PyPI:
```
$ pip install pyserini
```
Pyserini is built on Python 3.10 (other versions might work, but YMMV) and Java 21 (due to its dependency on [Anserini](https://github.com/castorini/anserini)). For all the retrievers we need to install Pyserini of Version: 0.44.0, only for bm25>>monot5 we need to install Pyserini of Version: 0.16.1.

## Data Preparation
### Data Preprocessing
Convert your file into a tsv file `/home/user/data/data.tsv` of the format:
```
137334	Fox 2000 Pictures released the film Soul Food.
111897	Telemundo is a English-language television network.
89891	Damon Albarn's debut album was released in 2011.
181634	There is a capital called Mogadishu.
```
Convert your file into a csv file `/home/user/data/data.csv` of the format:
```
Unnamed: 0,id,claim,label
2,137334,Fox 2000 Pictures released the film Soul Food.,1
4,111897,Telemundo is a English-language television network.,0
5,89891,Damon Albarn's debut album was released in 2011.,0
6,181634,There is a capital called Mogadishu.,1
7,219028,Savages was exclusively a German film.,0
```
### Corpus Preprocessing
Convert your corpus of the form:
```
{"id": 83235, "contents": "System of a Down briefly disbanded in limbo."}
{"id": 149579, "contents": "Beautiful reached number two on the Billboard Hot 100 in 2003."}
{"id": 229289, "contents": "Neal Schon was named in 1954."}
{"id": 138117, "contents": "John Wick: Chapter 2 was theatrically released in the Oregon."}
```
save that corpus into ```/home/user/data/corpus/nei/NEI_bucket.jsonl```.
## How do I search?
### Single-stage Ranker
For this paper, we uniformly retrieved the top 50 candidates.
#### BM25
```
## indexing
!python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/user/data/corpus/nei/ \
  --index /home/user/data/index_file/nei/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

## search
!python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index /home/user/data/index_file/nei/ \
  --topics /home/user/data/data.tsv \
  --output /home/user/result/bm25_ret/bm25_nei_ret.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 100
```

#### Contriever-E2E
```
##indexing
!python -m pyserini.encode \
  input \
    --corpus /home/user/data/corpus/nei/NEI_bucket.jsonl \
    --fields text \
    --delimiter "\n" \
    --shard-id 0 \
    --shard-num 1 \
  output \
    --embeddings /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
    --to-faiss \
  encoder \
    --encoder facebook/contriever-msmarco \
    --fields text \
    --batch 32 \
    --fp16

#search
!python -m pyserini.search.faiss \
  --index /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
  --topics /home/user/data/data.tsv \
  --encoder facebook/contriever-msmarco \
  --output /home/user/result/contriever_ret/nei_contriever_results.txt \
  --batch-size 64 \
  --threads 4 \
  --hits 50
```
#### ColBERT-E2E
```
##indexing
!python -m pyserini.encode \
  input \
    --corpus /home/user/data/corpus/nei/NEI_bucket.jsonl \
    --fields text \
    --delimiter "\n" \
    --shard-id 0 \
    --shard-num 1 \
  output \
    --embeddings /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
    --to-faiss \
  encoder \
    --encoder castorini/tct_colbert-v2-hnp-msmarco \
    --fields text \
    --batch 32 \
    --fp16

#search
!python -m pyserini.search.faiss \
  --index /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
  --topics /home/user/data/data.tsv \
  --encoder castorini/tct_colbert-v2-hnp-msmarco \
  --output /home/user/result/contriever_ret/nei_colbert_results.txt \
  --batch-size 64 \
  --threads 4 \
  --hits 50
```
### Two-stage Ranker
In our retrieve-and-rerank approach, we first retrieve the top 100 candidate documents using the same configurations as our single-stage retriever, and then re-ranks (also configured identically as explained in single-stage retriever) to narrow these down to the top 50 documents.
#### BM25»MonoT5




## Post-processing
### For Single-stage
Download the ```train.jsonl``` file from https://fever.ai/dataset/fever.html. (As we have used Fever training data as source data)
```


### For Two-stage
```
The post-processing for BM25>>MonoT5 is as follows:
```



## Replicating Results
### L-RAG
```
!python3 ICL_experiment.py \
   --k give_the_corresponding_labeled_context \  
   --models "TheBloke/Llama-2-70B-Chat-AWQ" \
   --data_path "/home/user/data/data.csv" \
   --wiki_S_path "/home/user/result/bm25_test_ret_support.pickle" \
   --wiki_R_path "/home/user/result/bm25_test_ret_refute.pickle" \
   --wiki_NEI_path "/home/user/result/bm25_test_ret_nei.pickle" \
   --true_pred_dict_file "/home/user/result/3_class/icl_predicted_dict_shot.pickle"
```

### U-RAG

```
!python3 RAG_experiment.py \
     --k give_the_corresponding_unlabeled_context \
     --models "TheBloke/Llama-2-70B-Chat-AWQ" \
     --data_path "/home/user/data/data.csv" \
     --wiki_retrieved_path  "/home/user/result/bm25_test_ret_wiki.pickle" \
     --true_pred_dict_file "/home/user/result/3_class/rag_predicted_dict_shot.pickle"
```
