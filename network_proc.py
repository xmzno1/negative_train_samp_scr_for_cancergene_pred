import operator
import csv
import os
import matplotlib.pyplot as plt
import networkx as nx

def load_cancer_gene_name_from_file(filename = "cg-2022-06.csv"):
    fields = []
    rows = []
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
      
        # extracting field names through first row
        fields = next(csvreader)
  
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    cancer_gene_names_list = []
    for row in rows:
        cancer_gene_names_list.append(row[0])
    return cancer_gene_names_list

def load_gene_mapping_from_file(filename = "identifier_mappings.txt"):
    fields = []
    rows = []
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile, delimiter = "\t")
      
        # extracting field names through first row
        fields = next(csvreader)
  
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    gene_names_dict = {}
    gene_ids_dict = {}
    gene_id_to_entrez_dict = {}
    gene_entrez_to_id_dict = {}
    for row in rows:
        if row[2] == 'Gene Name':
            gene_names_dict[row[0]] = row[1]
            gene_ids_dict[row[1]] = row[0]
        if row[2] == 'Entrez Gene ID':
            gene_id_to_entrez_dict[row[0]] = row[1]
            gene_entrez_to_id_dict[row[1]] = row[0]
    return gene_names_dict, gene_ids_dict, gene_id_to_entrez_dict, gene_entrez_to_id_dict

def load_weighted_G_from_file(filename):
    G = nx.Graph()
    G = nx.read_weighted_edgelist(filename)
    return G
    
def cal_weight_deg_norm(G):
    wt_deg = {}
    for n, nbrs in G.adj.items():
        wt_sum = 0
        for nbr, eattr in nbrs.items():
            wt = eattr['weight']
            wt_sum += wt
        wt_deg[n] = wt_sum
    wt_deg_max = max(list(wt_deg.values()))
    wt_deg_nor = {}
    for g, g_deg in wt_deg.items():
        wt_deg_nor[g] = g_deg/wt_deg_max
    return wt_deg_nor

def cal_weight_eigen_cen_norm(G):
    cen_eigen = nx.katz_centrality(G,weight='weight')
    cen_eigen_max = max(list(cen_eigen.values()))
    cen_eigen_nor = {}
    for g, g_cen in cen_eigen.items():
        cen_eigen_nor[g] = g_cen/cen_eigen_max
    return cen_eigen_nor

def cal_closeness_cen_norm(G):
    cen_clo = nx.closeness_centrality(G)
    cen_clo_max = max(list(cen_clo.values()))
    cen_clo_nor = {}
    for g, g_cen in cen_clo.items():
        cen_clo_nor[g] = g_cen/cen_clo_max
    return cen_clo_nor

def cal_deg_cen_norm(G):
    deg_cen = nx.degree_centrality(G)
    deg_cen_max = max(list(deg_cen.values()))
    deg_cen_nor = {}
    for g, g_cen in deg_cen.items():
        deg_cen_nor[g] = g_cen/deg_cen_max
    return deg_cen_nor

def cal_comb_weight_cen_sortedlist(G, large_net = False):
    wt_deg_nor = cal_weight_deg_norm(G)
    #cen_eigen_nor = cal_weight_eigen_cen_norm(G)
    if not large_net:
        cen_clo_nor = cal_closeness_cen_norm(G)
    else:
        cen_clo_nor = cal_deg_cen_norm(G)
        #cen_clo_nor = wt_deg_nor
    cen = {}
    for g, g_deg in wt_deg_nor.items():
        #cen[g] = g_deg + cen_eigen_nor[g]
        cen[g] = g_deg + cen_clo_nor[g]
    sort_gene_cen_list = sorted(cen.items(), key = operator.itemgetter(1), reverse = True)
    return sort_gene_cen_list

def is_tp(gene_id, cancer_gene_names_list, gene_names_dict):
    if gene_names_dict[gene_id] in cancer_gene_names_list:
        return True
    else:
        return False

def fpr(pred_p, pred_n, cancer_gene_names_list, gene_names_dict):
    fp_num = 0
    for gene in pred_p:
        if not is_tp(gene[0], cancer_gene_names_list, gene_names_dict):
            fp_num += 1
    tn_num = 0
    for gene in pred_n:
        if not is_tp(gene[0], cancer_gene_names_list, gene_names_dict):
            tn_num += 1
    if (fp_num + tn_num) == 0:
        return 0
    else:
        return fp_num / (fp_num + tn_num)

def tpr(pred_p, pred_n, cancer_gene_names_list, gene_names_dict):
    tp_num = 0
    for gene in pred_p:
        if is_tp(gene[0], cancer_gene_names_list, gene_names_dict):
            tp_num += 1
    fn_num = 0
    for gene in pred_n:
        if is_tp(gene[0], cancer_gene_names_list, gene_names_dict):
            fn_num += 1
    if (tp_num + fn_num) == 0:
        return 0
    else:
        return tp_num / (tp_num + fn_num)

def cal_roc_xy(sort_gene_cen_list, cancer_gene_names_list, gene_names_dict, interval = 100):
    x_fpr = []
    y_tpr = []
    for i in range(1,interval):
        clf_l = int(len(sort_gene_cen_list) * (1/interval) * i)
        pred_p = sort_gene_cen_list[:clf_l]
        pred_n = sort_gene_cen_list[clf_l:]
        x_fpr.append(fpr(pred_p,pred_n, cancer_gene_names_list, gene_names_dict))
        y_tpr.append(tpr(pred_p,pred_n, cancer_gene_names_list, gene_names_dict))
    return x_fpr, y_tpr

def auc(y_tpr):
    area_sum = 0
    for i in range(0, 99):
        area_sum += y_tpr[i] * 0.01
    return area_sum

def merge_edge_weight(G, g1, g2, weight):
    if G.has_edge(g1, g2):
        tmp = G[g1][g2]['weight']
        tmp += weight
        G[g1][g2]['weight'] = tmp
    else:
        G.add_edge(g1, g2, weight = weight)

################  importance calculation ########################

def cal_imp_deg_norm(G):
    wt_deg = {}
    node_nedges = {}
    edges_num = G.number_of_edges()
    for n, nbrs in G.adj.items():
        wt_sum = 0
        node_nedges[n] = len(G.adj[n])
        for nbr, eattr in nbrs.items():
            wt = eattr['weight']
            wt_sum += wt
        wt_deg[n] = wt_sum
    wt_deg_max = max(list(wt_deg.values()))
    wt_deg_nor = {}
    for g, g_deg in wt_deg.items():
        wt_deg_nor[g] = (g_deg/wt_deg_max)*(edges_num/(edges_num-node_nedges[g]))
    return wt_deg_nor

def num_of_bridges(G):
    node_bridges_num = {}
    bridges = nx.bridges(G)
    blist = list(bridges)
    for n in list(G.nodes()):
        num = 0
        for b in blist:
            if n in b:
                num += 1
        node_bridges_num[n] = num
    return node_bridges_num

def get_cc_wt(G):
    cc_wt = {}
    node_bridges_num = num_of_bridges(G)
    ncc = nx.number_connected_components(G)
    #if ncc > 1:
        #S = [G.subgraph(c).copy(as_view = True) for c in nx.connected_components(G)]
    for n in list(G.nodes()):
        if node_bridges_num[n] > 0:
            cc_wt[n] = (node_bridges_num[n] + ncc) / ncc
        else:
            cc_wt[n] = 1
    return cc_wt

def cal_nodes_imp(G):
    nodes_imp = {}
    wt_deg_nor = cal_imp_deg_norm(G)
    cc_wt = get_cc_wt(G)
    for n in list(G.nodes()):
        nodes_imp[n] = wt_deg_nor[n] * cc_wt[n]
    return nodes_imp

#################################################################

##########  file operation function  ##########################

def delete_line(original_file, line_number):
    """ Delete a line from a file at the given line number """
    is_skipped = False
    current_index = 0
    dummy_file = original_file + '.bak'
    # Open original file in read only mode and dummy file in write mode
    with open(original_file, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Line by line copy data from original file to dummy file
        for line in read_obj:
            # If current line number matches the given line number then skip copying
            if current_index != line_number:
                write_obj.write(line)
            else:
                is_skipped = True
            current_index += 1
    # If any line is skipped then rename dummy file as original file
    if is_skipped:
        os.remove(original_file)
        os.rename(dummy_file, original_file)
    else:
        os.remove(dummy_file)
        
def get_net_files_attri(directory = r'.\selected\Co-expression', net_sufix = 'txt'):
    net_weight = {}
    net_edge_num = {}
    net_node_number = {}
    net_max_edge_weight = {}
    net_edge_weights = {}
    total_edge_num = 0
    total_node_num = 0
    for filename in os.listdir(directory):
        if filename.endswith(net_sufix):
            full_path_name = os.path.join(directory, filename)
            G = nx.Graph()
            G = nx.read_weighted_edgelist(full_path_name)
            e_num = G.number_of_edges()
            n_num = G.number_of_nodes()
            net_edge_num[filename] = e_num
            net_node_number[filename] = n_num
            net_weight[filename] = 0
            total_edge_num += e_num
            total_node_num += n_num
            net_edge_weights[filename] = []
            for egs in list(G.edges):
                net_edge_weights[filename].append(G[egs[0]][egs[1]]['weight'])
            net_max_edge_weight[filename] = max(net_edge_weights[filename])
        else:
            continue
    for net, e_num in net_edge_num.items():
        net_weight[net] = e_num / total_edge_num
    return dict({"net_weight": net_weight, "net_edge_num": net_edge_num, "net_node_number": net_node_number, "net_max_edge_weight": net_max_edge_weight, "net_edge_weights": net_edge_weights, "total_edge_num": total_edge_num, "total_node_num": total_node_num})

def write_weighted_net_files(net_max_edge_weight, net_weight, directory = r'.\selected\Co-expression', filesufix = '.weighted'):
    for filename in net_max_edge_weight.keys():
        full_path = os.path.join(directory, filename)
        dummy_file = full_path + filesufix
        with open(full_path, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
            # Line by line write data from original file to dummy file
            for line in read_obj:
                tmp = line.split()
                tmp_weight = (float(tmp[2]) / net_max_edge_weight[filename]) * net_weight[filename]
                write_obj.write(tmp[0] + '\t' + tmp[1] + '\t' + str(tmp_weight) + '\n')
            # os.rename(dummy_file, original_file)

def write_edges_file_from_list(edge_list, directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as write_obj:
        for edge in edge_list:
            write_obj.write(edge[0] + '\t' + edge[1] + '\t' + str(edge[2]) + '\n')

########################   Machine Learning Utilities   ######################################

def get_graph_adjmat_n_nodelist(filename = "./selected/Mania_Combined/COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt"):
    G = load_weighted_G_from_file(filename)
    A = nx.to_numpy_array(G)
    node_list = list(G.nodes())
    G.clear()
    return A, node_list

# sorted gene dict by importance
def get_sorted_gene_dict(filename = "./selected/Mania_Combined/COMBINED-gene-cent.tsv"):
    genes_sorted = {}
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile, delimiter = "\t")
  
        # extracting each data row one by one
        for row in csvreader:
            genes_sorted[row[0]] = float(row[1])
    return genes_sorted

#return cancer gene list ['gene name', 'entrez id']
def get_cancer_genes_list(filename = "cg-2022-06.csv"):
    cancer_gene_list = []
    fields = []
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            cancer_gene_list.append([row[0], row[2]])
    return cancer_gene_list

#return aml gene list ['gene name', 'entrez id']
def get_aml_genes_list(filename = "cg-2022-06.csv"):
    fields = []
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
    aml_genes_list = []
    for row in rows:
        has_aml = False
        for s in row[9].split(','):
            if 'AML' in s:
                has_aml = True
        if has_aml:
            aml_genes_list.append([row[0], row[2]])
    return aml_genes_list

#return mutation genes list ['gene name', 'entrez id']
def get_mut_genes_list(filename = "mut-genes-data-nodup.txt"):
    mut_genes_list = []
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = "\t")
        for row in csvreader:
            rows.append(row)
    for row in rows:
        mut_genes_list.append([row[7], row[8]])
    return mut_genes_list

#get mut genes exist in mania mapping file, return dict: {'mania id': 'gene name or entrez id'}
def get_mut_genes_exist_dict(mut_genes_list, gene_ids_dict, gene_entrez_to_id_dict):
    mut_genes_exist_dict = {}
    for mut in mut_genes_list:
        if mut[0] in list(gene_ids_dict.keys()):
            mut_genes_exist_dict[gene_ids_dict[mut[0]]] = mut[0]
        elif mut[1] in list(gene_entrez_to_id_dict.keys()):
            mut_genes_exist_dict[gene_entrez_to_id_dict[mut[1]]] = mut[1]
    return mut_genes_exist_dict

#return mut genes list in current network ['mania id']
def get_in_net_mut_list(genes_sorted, mut_genes_exist_dict):
    in_net_mut = []
    genes_list = list(genes_sorted.keys())
    for key, value in mut_genes_exist_dict.items():
        if key in genes_list:
            in_net_mut.append(key)
    return in_net_mut

#get aml genes mania id list ['mania id']
def get_aml_genes_id_list(aml_genes_list, gene_ids_dict, gene_entrez_to_id_dict):
    aml_genes_id = []
    for aml in aml_genes_list:
        if aml[0] in list(gene_ids_dict.keys()):
            aml_genes_id.append(gene_ids_dict[aml[0]])
        elif aml[1] in list(gene_entrez_to_id_dict.keys()):
            aml_genes_id.append(gene_entrez_to_id_dict[aml[1]])
    return aml_genes_id

#get mut genes which are not aml genes ['mania id']
def get_unaml_mut_genes_list(in_net_mut, aml_genes_id):
    unaml_mut_genes = []
    for mut in in_net_mut:
        if mut not in aml_genes_id:
            unaml_mut_genes.append(mut)
    return unaml_mut_genes

