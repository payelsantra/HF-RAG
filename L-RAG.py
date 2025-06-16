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
    parser.add_argument('--wiki_R_path', type=str, required=True, help='Path to the REF data')
    # parser.add_argument('--wiki_NEI_path', type=str, required=True, help='Path to the NEI data')   #change
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

    llm = LLM(model=args.models)
    k_val=args.k
    data = pd.read_csv(args.data_path)
    data_wiki_S = pkl.load(open(args.wiki_S_path, "rb"))
    data_wiki_R = pkl.load(open(args.wiki_R_path, "rb"))
    # data_wiki_NEI = pkl.load(open(args.wiki_NEI_path, "rb"))  #change

    clm_id=data['idx'].tolist()
    claim_lst=data['claim'].tolist()
    label_lst=data['label'].tolist()
    claim_id_claim = dict(zip(clm_id, claim_lst))
    claim_id_labels = dict(zip(clm_id, label_lst))

    # label_dict = {0: 'False', 1: 'True', 2: 'Not Enough Information'}  #change
    label_dict = {0: 'False', 1: 'True'}   #change
    label_id={'unproven':2,'Unproven':2,'UNPROVEN':2,'Not Enough Information': 2,'NOT ENOUGH INFORMATION':2,'SUPPORTS':1,'SUPPORT':1,'REFUTES':0, 'False': 0,'FALSE':0, 'True': 1,"Partially True":1,'true':1,'false':0, "Mostly False":0, "Mostly True":1,'MISLEADING':2,'contradicts':0,'SUPPORTS':1,'False':0}

#   3 class
    # instrction_txt = '''Your task as a fact verifier is to analyze claims and determine their claim label, which can be either 'True', 'False' or 'Not Enough Information'.\n'''
#   2 class  #change
    instrction_txt = '''Your task as a fact verifier is to analyze claims and determine their claim label, which can be either 'True' or 'False'.\n'''
    instrction_txt2 = '''\nGiven a claim, you should provide a response in the format {"label": "class"}.'''

    prompt_list = {}
    for idx,smple in enumerate(tqdm(clm_id)):
        prompt_txt="{}\n".format(instrction_txt)
        clm=claim_id_claim[smple]
        evi_S=data_wiki_S[smple]
        evi_R=data_wiki_R[smple]
        # evi_NEI=data_wiki_NEI[smple]   #change
        for i in range(k_val):   #k-shot
            evi_id=str(smple)+"_"+str(i)
            evi_ditect_S=evi_S[evi_id][0]
            evi_ditect_R=evi_R[evi_id][0]
            # evi_ditect_NEI=evi_NEI[evi_id][0]   #change
            evi_ditect_S_id=str(evi_S[evi_id][1])
            evi_ditect_R_id=str(evi_R[evi_id][1])
            # evi_ditect_NEI_id=str(evi_NEI[evi_id][1])   #change
            prompt_txt=prompt_txt+'''Input: {}\nOutput: {}\n'''.format(evi_ditect_S, 'SUPPORTS')
            prompt_txt=prompt_txt+'''Input: {}\nOutput: {}\n'''.format(evi_ditect_R, 'REFUTES')
            # prompt_txt=prompt_txt+'''Input: {}\nOutput: {}\n'''.format(evi_ditect_NEI, 'NOT ENOUGH INFORMATION')   #change
        prompt_txt=prompt_txt+instrction_txt2
        prompt_n=prompt_txt+"\nInput: {}\nOutput: ".format(clm)
        prompt_list[smple]=prompt_n

    sampling_params = SamplingParams(temperature=0.0, max_tokens=15, top_p=1, length_penalty=0.7, best_of=10, use_beam_search=True, early_stopping=True)
    responses = llm.generate(list(prompt_list.values()), sampling_params)
    response_dict = dict(zip(list(prompt_list.keys()), responses))


    #final cleaning
    extrcted_lst=[]
    extrcted_dict={}
    not_parsed=[]
    dict_val={}
    for i in tqdm(response_dict):
        json_txt=response_dict[i].outputs[0].text.strip()
        # print(json_txt)
        start_label=response_dict[i].outputs[0].text.strip().find(':')
        a=isWordPresent(json_txt, "True")
        b=isWordPresent(json_txt, "False")
        c=isWordPresent(json_txt, '"label": "True"')
        d=isdotWordPresent(json_txt, '"label": "True",')
        e=isWordPresent(json_txt, '"label": "True",')
        f=isWordPresent(json_txt, '"label": "SUPPORT')
        g=isWordPresent(json_txt, '{"label": "supports"}')
        h=isWordPresent(json_txt, '“label”: “SUPPORTS”')
        ii=isWordPresent(json_txt, '"label": "False"')
        j=isWordPresent(json_txt, '"label": "False",')
        k=isWordPresent(json_txt, '"label": "REFUT')
        l=isWordPresent(json_txt, '"label": "REFUTES')
        m=isWordPresent(json_txt, '"label": "REFUTES",')
        n=isWordPresent(json_txt, '"label": "refutes"')
        o=isWordPresent(json_txt, '{"label": "refutes"}')
        p=isWordPresent(json_txt, '{ "label": "refutes" }')
        q=isWordPresent(json_txt, '"label": "refutes",')
        r=isWordPresent(json_txt, '"label": "FALSE"')
        s=isWordPresent(json_txt, '"label": "unproven"')
        t=isWordPresent(json_txt, "Not Enough Information")
        u=isdotWordPresent(json_txt, "False.")
        v=isdotWordPresent(json_txt, "Not Enough Information.")
        w=isdotWordPresent(json_txt, '“label”: “SUPPORTS”')
        x=isWordPresent(json_txt, '"label": "SUPPORTS",')
        y=isWordPresent(json_txt, '"label": "REFUTES"')
        try:
            start=response_dict[i].outputs[0].text.strip().find('{')
            end=response_dict[i].outputs[0].text.strip().find('}')
            json_txt_=response_dict[i].outputs[0].text.strip()[start:end+1]
            try:
                extrcted_lst.append(label_id[str(ast.literal_eval(json_txt_)['label'])])
                extrcted_dict[i]=label_id[str(ast.literal_eval(json_txt_)['label'])]
                dict_val[i]=label_id[str(ast.literal_eval(json_txt_)['label'])]
            except:
                extrcted_lst.append(label_id[ast.literal_eval(json_txt_)['label']])
                extrcted_dict[i]=label_id[ast.literal_eval(json_txt_)['label']]
                dict_val[i]=label_id[ast.literal_eval(json_txt_)['label']]
        except:
            if a==True:
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif b==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif c==True:
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif d==True:
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif e==True:
                dict_val[i]=label_id['True']
                extrcted_dict[i]=label_id['True']
                extrcted_lst.append(label_id['True'])
            elif f==True:
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif g==True:
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif h==True:
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])    
            elif ii==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif j==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif k==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif l==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif m==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif y==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif n==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif o==True:
    #             print(i)
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif p==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            elif q==True:
                dict_val[i]=label_id['False']
                extrcted_dict[i]=label_id['False']
                extrcted_lst.append(label_id['False'])
            # elif s==True:
            #     dict_val[i]=label_id['unproven']
            #     extrcted_dict[i]=label_id['unproven']
            #     extrcted_lst.append(label_id['unproven'])
            elif r==True:
                dict_val[i]=label_id['FALSE']
                extrcted_dict[i]=label_id['FALSE']
                extrcted_lst.append(label_id['FALSE'])  
            # elif t==True:
            #     dict_val[i]=label_id['Not Enough Information']
            #     extrcted_dict[i]=label_id['Not Enough Information']
            #     extrcted_lst.append(label_id['Not Enough Information'])
            elif u==True:
                dict_val[i]=label_id['False.']
                extrcted_dict[i]=label_id['False.']
                extrcted_lst.append(label_id['False.'])
            # elif v==True:
            #     dict_val[i]=label_id['Not Enough Information.']
            #     extrcted_dict[i]=label_id['Not Enough Information.']
            #     extrcted_lst.append(label_id['Not Enough Information.'])
            elif w==True:
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            elif x==True:
                dict_val[i]=label_id['SUPPORT']
                extrcted_dict[i]=label_id['SUPPORT']
                extrcted_lst.append(label_id['SUPPORT'])
            else:
                try:
                    try:
                        id_got=label_id[str(response_dict[i].outputs[0].text.strip())]
                        dict_val[i]=id_got
                        extrcted_dict[i]=id_got
                        extrcted_lst.append(id_got)
                    except:
                        id_got=label_id[str(response_dict[i].outputs[0].text.strip().split('\n')[0])]
            #                     print(id_got)
                        dict_val[i]=id_got
                        extrcted_dict[i]=id_got
                        extrcted_lst.append(id_got)
                except:
                    not_parsed.append(response_dict[i])
                    extrcted_lst.append(0)  #change
                    extrcted_dict[i]=0
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
