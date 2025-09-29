import numpy as np
from collections import defaultdict
import json
import pickle
import pandas as pd
import argparse
import os

def parse_trec_file_with_zscores(filepath, top_k=50):
    raw_scores = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid, score = parts[0], parts[2], float(parts[4])
            raw_scores[qid].append((docid, score))

    zscore_data = defaultdict(list)
    for qid, docs in raw_scores.items():
        docs = sorted(docs, key=lambda x: x[1], reverse=True)[:top_k]
        scores = np.array([score for _, score in docs])
        if np.std(scores) == 0:
            z_scores = np.zeros_like(scores)
        else:
            z_scores = (scores - np.mean(scores)) / np.std(scores)
        for (docid, _), z in zip(docs, z_scores):
            zscore_data[qid].append((docid, z))

    return zscore_data

def main(args):
    # Load evidence text
    with open("/media/pbclab/Elements/Fact_verification/bert/collection_file/wiki_collection_sen_proper_abc_dict.pickle", 'rb') as f:
        evidence_text = pickle.load(f)

    # Load in-domain evidence
    evidence_indom = {}
    with open("/media/pbclab/Elements/Fact_verification/bert/fever_dataset/train.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            evidence_indom[json_obj['id']] = (json_obj['claim'], json_obj['label'])

    # Load test data
    test_data = pd.read_csv('/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/data/shared_task_dev_fever_data.csv')
    # test_data = pd.read_csv("/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/dataset/3_class/climate_fever_test_data_wo_defuted.csv")
    id_claim_dict = dict(zip(list(test_data['id']), list(test_data['claim'])))
    id_label_dict = dict(zip(list(test_data['id']), list(test_data['label'])))

    # test_data = pd.read_csv('/media/pbclab/Elements/prompt/prompt_2/sciFact/data/scifact_dataset_unique_claim/3_class/unique_test.csv')
    # id_claim_dict=dict(zip(test_data['Claim_id'],test_data['Claim']))
    # id_label_dict=dict(zip(test_data['Claim_id'],test_data['label']))

# rank_files = [
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/BM25_retrived/retrieved_result/wiki18/bm25_CF_test_wiki_ret.txt',
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/colbert/retrieved_pyserini/wiki_ret_CF_wo_refuted_colbert_results.txt',
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/contriever/retreived_pyserini/wiki_ret_CF_wo_refuted_results_contriever.txt',
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/BM25_retrived/monot5/wiki_CF_bm25_monot5_FeverTr_converted.txt',
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/BM25_retrived/retrieved_result/indomain/bm25_CFtest_combined_indom.txt',
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/colbert/retrieved_pyserini/colbined_ret_climatefever_colbert_combined_indom.txt',
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/contriever/retreived_pyserini/colbined_ret_climatefever_contriever_combined_indom.txt',
#     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/Cilmate_fever/BM25_retrived/monot5/bm25_monot5_CF_feverTr_combined_indom_converted.txt'
# ]

    # rank_files = [
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/bm25_ret/bm25_Scifact_test_wiki_ret.txt',
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/colbert/wiki_ret_Scifact_results_colbert.txt',
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/contriever/wiki_ret_Scifact_results_contriever.txt',
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/bm25_ret/MONOT5/Avrg_Weighted/wiki_scifact_bm25_monot5_weighted_avrg.txt',
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/bm25_ret/bm25_scifact_test_combinedFEVER_ret.txt',
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/colbert/colbined_ret_SCifact_clbert_results.txt',
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/contriever/combined/colbined_ret_SCifact_contriever_results.txt',
    #     '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/scifact/bm25_ret/MONOT5/Avrg_Weighted/bm25_monot5_scifact_feverTr_weightedAvg_ret_normalized_combined_indom.txt'
    # ]

    rank_files = [
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/bm25_ret/bm25_fever_test_wiki_ret.txt',
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/colbert/wiki_ret_fever_results_colbert.txt',
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/contriever/wiki_ret_fever_results_contriever.txt',
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/bm25_ret/monot5/wiki_Fevertest_bm25_monot5_FeverTr_converted.txt',
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/bm25_ret/bm25_Fever_test_combinedFEVERtr_ret.txt',
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/colbert/combined_ret_fever_clbert_results.txt',
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/contriever/combined_ret_fever_contriever_results.txt',
    '/media/pbclab/Expansion1/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/bm25_ret/monot5/bm25_monot5_Fevertest_FevertR_combined_indom_converted.txt'
]


    # Step 1: Load and normalize all files
    all_ranklists_zscore = [parse_trec_file_with_zscores(fp, top_k=args.top_k) for fp in rank_files]

    # Step 2: Merge z-scores
    aggregated_scores = defaultdict(list)
    for ranklist in all_ranklists_zscore:
        for qid, docs in ranklist.items():
            for docid, zscore in docs:
                aggregated_scores[(qid, docid)].append(zscore)

    # Step 3: Aggregate (max or sum)
    final_scores = defaultdict(list)
    for (qid, docid), zscore_list in aggregated_scores.items():
        if args.agg_method == "sum":
            agg_z = sum(zscore_list)
        else:
            agg_z = max(zscore_list)
        final_scores[qid].append((docid, agg_z))

    method_name = f"merged_ranklist_top{args.top_k}_{args.agg_method}Based"
    output_rank_file = os.path.join(args.output_dir, method_name + ".txt")

    # Step 4: Write output ranklist
    with open(output_rank_file, 'w') as out:
        for qid in sorted(final_scores):
            ranked_docs = sorted(final_scores[qid], key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked_docs):
                out.write(f"{qid} Q0 {docid} {rank+1} {score:.6f} merged_zscore\n")
    print(f"✅ Ranklist saved to: {output_rank_file}")

    # Step 5: Build final evidence dict
    sorted_mixed_result = defaultdict(dict)
    with open(output_rank_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid, rank, score, method = int(parts[0]), parts[2], int(parts[3]), float(parts[4]), parts[5]
            source = "indom" if not docid.startswith("doc") else "wiki"
            sorted_mixed_result[qid][docid] = (score, source)

    whole_evidence = {}
    for i in sorted(sorted_mixed_result):
        claim = id_claim_dict[int(i)]
        whole_evidence[i] = {}
        for num, k in enumerate(sorted_mixed_result[i]):
            evi_str = f"{i}_{num}"
            if not k.startswith("doc"):
                doc_id = int(k)
                evidence, evidence_lab = evidence_indom[doc_id]
                whole_evidence[i][evi_str] = [evidence, claim, evidence_lab]
            else:
                doc_id = k[3:]
                evidence = evidence_text[doc_id].strip()
                whole_evidence[i][evi_str] = [evidence, claim, "wiki"]

    # Step 6: Save pickle
    output_pickle = os.path.join(args.output_dir, f"merged_z_scoreOnly_fever_test_{len(rank_files)}RL_{args.agg_method}Based.pickle")
    with open(output_pickle, 'wb') as f:
        pickle.dump(whole_evidence, f)

    print(f"✅ Evidence dictionary saved to: {output_pickle}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ranklists using z-score normalization and aggregate using sum or max.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--agg_method", choices=["sum", "max"], default="sum", help="Aggregation method: sum or max.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K documents to consider from each ranker.")
    args = parser.parse_args()
    main(args)
