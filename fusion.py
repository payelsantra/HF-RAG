import argparse
import pandas as pd
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report, f1_score, accuracy_score
import ast

def parse_arguments():
    parser = argparse.ArgumentParser(description='Climate fever 2 class Data Transfer Learning Experiment for ICL, EVI agnostic')
    parser.add_argument('--k', type=int, required=True, help='# shot' )
    parser.add_argument('--models', type=str, required=True, help='Model path')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--wiki_S_path', type=str, required=True, help='Path to the SUP data')
    parser.add_argument('--true_pred_dict_file', type=str, required=True, help='Path to true_pred_dict_file')
    return parser.parse_args()

def isdotWordPresent(sentence, word):
    s = sentence.split(" ")
    for i in s:
        if (i == word):
            return True
    return False

def isWordPresent(sentence, word):
    s = sentence.split("\n")
    for i in s:
        if (i == word):
            return True
    return False

def main():
    args = parse_arguments()

    llm = LLM(model=args.models,quantization="awq")
    k_val=args.k
    data = pd.read_csv(args.data_path)
    data_avrg = pkl.load(open(args.wiki_S_path, "rb"))

    clm_id=data['id'].tolist()
    claim_lst=data['claim'].tolist()
    label_lst=data['label'].tolist()
    claim_id_claim = dict(zip(clm_id, claim_lst))
    claim_id_labels = dict(zip(clm_id, label_lst))

    label_dict = {0: 'False', 1: 'True', 2: 'Not Enough Information'}  #change
    # label_dict = {0: 'False', 1: 'True'}   #change
    label_id={'unproven':2,'Unproven':2,'UNPROVEN':2,'Not Enough Information': 2,'NOT ENOUGH INFORMATION':2,'SUPPORTS':1,'SUPPORT':1,'REFUTES':0, 'False': 0,'FALSE':0, 'True': 1,"Partially True":1,'true':1,'false':0, "Mostly False":0, "Mostly True":1,'MISLEADING':2,'contradicts':0,'SUPPORTS':1,'False':0,'negative':0,"Negative":0,'NEGATIVE':0,'negatives':0,"Negatives":0,'NEGATIVES':0}
    
#PROMPT4
    instrction_txt='''Your task as a fact verifier is to analyze claims and determine their claim label, which can be either 'True', 'False' or 'Not Enough Information'.You will be provided with a set of examples that may include both labeled and unlabeled claims. Each example consists of an Input (the claim) and may optionally include an Output (its label). If the context includes only labeled examples, you should rely solely on them. If only unlabeled examples are present, make your best judgment using the input context. When both labeled and unlabeled examples are provided, please assign higher weight to these examples compared to the rest of the example set, as they are more likely to guide the model in generating the correct output for the given claim.
'''

    instrction_txt2 = '''\nGiven a claim, you should provide a response in the format {"label": "class"}.'''

    
    #4bucket combined
    from tqdm import tqdm
    # # k_val=10
    # #few-shot with label respective of labels #with additional data
    prompt_list = {}
    for idx,smple in enumerate(tqdm(clm_id)):
        try:
            prompt_txt="{}\n".format(instrction_txt)
            clm=claim_id_claim[smple]
            evi_S=data_avrg[smple]
            for i in range(k_val):   #k-shot
                evi_id=str(smple)+"_"+str(i+1)
                if evi_S[evi_id][-1]=='wiki':
                    wiki_evi=evi_S[evi_id][0]
                    prompt_txt=prompt_txt+'''Input: {}\n'''.format(wiki_evi)
                else:
                    evi_ditect_S=evi_S[evi_id][0]
                    label=evi_S[evi_id][-1]
                    prompt_txt=prompt_txt+'''Input: {}\nOutput: {}\n'''.format(evi_ditect_S, label)
            prompt_txt=prompt_txt+instrction_txt2
            prompt_n=prompt_txt+"\nInput: {}\nOutput: ".format(clm)
            prompt_list[smple]=prompt_n
        except:
            continue
#     print(prompt_list[list(prompt_list.keys())[0]])

    # sampling_params = SamplingParams(temperature=0.0, max_tokens=15, top_p=1, length_penalty=0.7, best_of=10, use_beam_search=True, early_stopping=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_p=1, length_penalty=0.7, best_of=10, use_beam_search=True, early_stopping=True,seed=42)
    
    responses = llm.generate(list(prompt_list.values()), sampling_params)
    response_dict = dict(zip(list(prompt_list.keys()), responses))


    #final cleaning
    #final cleaning
    from tqdm import tqdm
    import ast
    extrcted_lst=[]
    extrcted_dict={}
    not_parsed=[]
    dict_val={}
    for i in tqdm(response_dict):
        try:
            start=response_dict[i].outputs[0].text.strip().find('{')
            end=response_dict[i].outputs[0].text.strip().find('}')
            json_txt=response_dict[i].outputs[0].text.strip()[start:end+1]
    #         print(label_id[str(ast.literal_eval(json_txt)['label'])])
            try:
                extrcted_lst.append(label_id[str(ast.literal_eval(json_txt)['label'])])
                extrcted_dict[i]=label_id[str(ast.literal_eval(json_txt)['label'])]
                dict_val[i]=label_id[str(ast.literal_eval(json_txt)['label'])]
            except:
                extrcted_lst.append(label_id[ast.literal_eval(json_txt)['label']])
                extrcted_dict[i]=label_id[ast.literal_eval(json_txt)['label']]
                dict_val[i]=label_id[ast.literal_eval(json_txt)['label']]
        except:
            json_txt=response_dict[i].outputs[0].text.strip().lower()
            start_label=response_dict[i].outputs[0].text.strip().find(':')
            if (isWordPresent(json_txt, "true")):
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif (isWordPresent(json_txt, '"label": "true"')):
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True']) 
            elif (isWordPresent(json_txt, 'label: true')):
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif (isdotWordPresent(json_txt, '"label": "true",')):
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif (isWordPresent(json_txt, '"label": "true",')):
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif (isWordPresent(json_txt, '"label": "support')):
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif (isWordPresent(json_txt, '{"label": "supports"}')):
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif (isdotWordPresent(json_txt, '“label”: “supports”')):
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif (isdotWordPresent(json_txt, '“label”: “supports”')):
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif (isWordPresent(json_txt, '"label": "supports",')):
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif (isWordPresent(json_txt, "False")):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, "false")):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '“label”: “SUPPORTS”')):
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])    
            elif (isWordPresent(json_txt, '"label": "false"')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False']) 
            elif (isWordPresent(json_txt, 'label: "false"')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '"label": "false",')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '"label": "refut')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '"label": "refutes')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, 'label: false')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '"label": "refutes",')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '"label": "refutes"')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '{"label": "refutes"}')):
    #             print(i)
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '{ "label": "refutes" }')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '"label": "refutes",')):
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif (isWordPresent(json_txt, '"label": "unproven"')):
                dict_val[i]=label_id['unproven']
                extrcted_dict[i]=label_id['unproven']
                extrcted_lst.append(label_id['unproven'])
            elif (isdotWordPresent(json_txt, "false.")):
                dict_val[i]=label_id['False.']
                extrcted_dict[i]=label_id['False.']
                extrcted_lst.append(label_id['False.'])
            elif (isWordPresent(json_txt, "not enough information")):
                dict_val[i]=label_id['Not Enough Information']
                extrcted_dict[i]=label_id['Not Enough Information']
                extrcted_lst.append(label_id['Not Enough Information'])
            elif (isdotWordPresent(json_txt, "not enough information.")):
                dict_val[i]=label_id['Not Enough Information.']
                extrcted_dict[i]=label_id['Not Enough Information.']
                extrcted_lst.append(label_id['Not Enough Information.'])
            else:
                try:
                    try:
                        id_got=label_id[str(response_dict[i].outputs[0].text.strip())]
                        dict_val[i]=id_got
                        extrcted_dict[i]=id_got
                        extrcted_lst.append(id_got)
                    except:
                        id_got=label_id[str(response_dict[i].outputs[0].text.strip().split('\n')[0])]
                        dict_val[i]=id_got
                        extrcted_dict[i]=id_got
                        extrcted_lst.append(id_got)
                except:
                    not_parsed.append(response_dict[i])
                    extrcted_lst.append(2)  #change
                    extrcted_dict[i]=2
                    dict_val[i]=response_dict[i].outputs[0].text.strip()

    true_pred_dict = {}
    for i in extrcted_dict:
        true_val = claim_id_labels[i]
        pred_val = extrcted_dict[i]
        true_pred_dict[i] = {'true': true_val, 'pred': pred_val}

    with open(args.true_pred_dict_file, 'wb') as fp:
        pkl.dump(true_pred_dict, fp)

    true_labels = []
    for i in dict_val:
        lab = claim_id_labels[i]
        true_labels.append(lab)

    pred = classification_report(true_labels, extrcted_lst, digits=4)
    macro_f1_score = f1_score(true_labels, extrcted_lst, average='macro')
    acc_score = accuracy_score(true_labels, extrcted_lst)

    print("k_value is",k_val)
    report = classification_report(true_labels, extrcted_lst, output_dict=True)
    classwise_metrics = {}
    for label, metrics in report.items():
        if label.isdigit():
            classwise_metrics[int(label)] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score']
            }
    for label, metrics in classwise_metrics.items():
        print(f"Class {label}:")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall: {metrics['recall']}")
        print(f"  F1-Score: {metrics['f1-score']}")
        print()

    print("macro f1 score", macro_f1_score)
    print("accuracy_score", acc_score)
    print("classification report", pred)

if __name__ == '__main__':
    main()
