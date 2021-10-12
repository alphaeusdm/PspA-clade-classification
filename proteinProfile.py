import sys
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl


def protProfile(seq):
    # get sequence profiles
    profile = {}
    aa_index = 0
    while aa_index + 9 < len(seq):
        window = seq[aa_index:aa_index+10]
        per_p = (window.count("P") / 10) * 100
        profile[aa_index] = per_p
        aa_index += 1
    df = pd.DataFrame(profile.items(), columns=["window start index", "% P"])
    pd.set_option("display.max_rows", None)
    return df

def process(file):
    subseq = {}
    with open(file) as f:
        strain = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                strain = line[line.index('|')+1:]
            else:
                df = protProfile(line)
                strain_name, subsequence = generateSubsequences(line, strain, df)
                subseq[strain_name] = subsequence
    for s in subseq:
        if len(subseq[s]) < 90:
            print(s +' : '+subseq[s])
    dataframe = pd.DataFrame(subseq.items(), columns=['StrainName_start_end_cladeID', 'Subsequence[P-100,P]'])
    with pd.ExcelWriter('holl2.xlsx') as writer:
        # writer.book = openpyxl.load_workbook('subseq_dataset.xlsx')
        dataframe.to_excel(writer, index=False)

# def processFile(file, strains):
#     acc_no = ""
#     prot_seq = ""
#     profiles = {}
#     acc_strain = {}
#     subseq = {}
#     filename = file[file.index('_')+1:file.index('.')]
#     with open(strains, "r") as f:
#         for line in f:
#             acc_strain[line[:line.index("-")-1]] = line[line.index("-")+2:].strip()
#     if filename == 'Brazil':
#         with open(file, "r") as file:
#             for line in file:
#                 if line[0] == ">":
#                     acc_no = line[line.index("|")+1:line.index(".")]
#                 elif line[0].isalpha():
#                     prot_seq += line.strip()
#                 else:
#                     # profiles[acc_strain[acc_no][:acc_strain[acc_no].index('_')]] = df
#                     subseq[acc_strain[acc_no]] = prot_seq
#                     acc_no = ""
#                     prot_seq = ""
#     else:
#         with open(file, "r") as file:
#             for line in file:
#                 if line[0] == ">":
#                     acc_no = line[line.index("|")+1:line.index(".")]
#                 elif line[0].isalpha():
#                     prot_seq += line.strip()
#                 else:
#                     df = protProfile(prot_seq)
#                     profiles[acc_strain[acc_no][:acc_strain[acc_no].index('_')]] = df
#                     strain_name, subsequence = generateSubsequences(prot_seq, acc_strain[acc_no], df)
#                     subseq[strain_name] = subsequence
#                     acc_no = ""
#                     prot_seq = ""
#     dataframe = pd.DataFrame(subseq.items(), columns=['StrainName_start_end_cladeID', 'Subsequence[P-100,P]'])
#     with pd.ExcelWriter('holl2.xlsx') as writer:
#         # writer.book = openpyxl.load_workbook('subseq_dataset.xlsx')
#         dataframe.to_excel(writer, index=False)
#         # dataframe.to_excel(writer, index=False, mode='a', header=None, sheet_name='Sheet1')
#     return profiles


def plotProfiles(dfs, file):
    # plot the profiles
    fig, axs = plt.subplots(ncols=4, nrows=6, sharex=True, sharey=True)
    i, j = 0, 0
    for df in dfs:
        # axs[i, j].scatter(x=dfs[df]['window start index'], y=dfs[df]["% P"], s=0.5)
        axs[i, j].plot(dfs[df]['window start index'], dfs[df]["% P"], '.-', linewidth=0.5, markersize=2.5)
        axs[i, j].set_title(df, fontsize=7)
        axs[i, j].set_xticks([0, 100, 200, 300, 400, 500, 600])
        axs[i, j].set_xticklabels([0, 100, 200, 300, 400, 500, 600], fontsize=7)
        axs[i, j].set_yticks([0, 10, 20, 30, 40, 50])
        axs[i, j].set_yticklabels([0, 10, 20, 30, 40, 50], fontsize=7)
        axs[i, j].axhline(y=30, color='red', ls='--', linewidth=0.5)
        if j < 3:
            j += 1
        else:
            j = 0
            i += 1
    fig.tight_layout()
    # fig.text(0.5, 0.02, 'window start index', ha='center')
    # fig.text(0.02, 0.5, '% P', va='center', rotation='vertical'
    # plt.show()
    plt.savefig(file)


def toExcel(profiles, file):
    mrows = 0
    for profile in profiles:
        if mrows < len(profiles[profile]):
            mrows = len(profiles[profile])
    indices = [x for x in range(0, mrows)]
    df = pd.DataFrame(indices, columns=["Index"])
    for profile in profiles:
        p = profiles[profile]['% P'].tolist()
        df[profile] = pd.Series(p)
    with pd.ExcelWriter(file) as writer:
        df.to_excel(writer, index=False)


def generateSubsequences(sequence, strain, dfs):
    # generate subsequences
    if strain.__contains__('ASP0959'):
        end = 213
    elif 30.0 in dfs['% P'].tolist():
        end = dfs['% P'].tolist().index(30.0)
    else:
        if len(sequence) <= 110:
            return strain, sequence
        else:
            return strain, sequence
    start = end - 100
    if start < 0:
        start = 0
    subseq = sequence[start:end+1]
    strain_name = strain[:strain.index('_')]+"_"+str(start)+"_"+str(end)+strain[strain.index('_'):]
    return strain_name, subseq


def main():
    file = sys.argv[1]
    excel_file = sys.argv[2]
    # fig_file = sys.argv[3]
    # strains = sys.argv[4]
    # profiles = processFile(file, strains)
    process(file)
    # for profile in profiles:
    #     df = profiles[profile]
    #     print(profile)
    #     print(df['% P'].to_list().index(30.0))
    #     print(profile)
    #     print(profiles[profile])
    # plotProfiles(profiles, fig_file)
    # toExcel(profiles, excel_file)


if __name__ == '__main__':
    main()
    # seq = 'SNNHDEFQALYESTQEQIEELKDYNEQISEGEETLILAIQNKISDLDDKIAEAEKKLADSQNGEGVEDYWTSGDEDKLEKLQAEQDELQAELDQLLDEVDGQEPAPEAPAEQPKPEKSAEQQAEEDYARRSEEEYNRLTQQQPPKAEKPAEEPTRPAPAPEAPAEQPKPEKSAEQQAEEDYARRSEEEYNRLTQHQPPKAEKPAEEPTQPAPAPEQPTEPTQPE'
    # seq = 'SNNHDEFQALYESTQEQIEELKDYNEQISEGEETLILAIQNKISDLDDKIAEAEKKLADSQNGEGVEDYWTSGDEDKLEKLQAEQDELQAELDQLLDEVDGQE'
    # print(protProfile(seq))
    s1 = 'MTFLTAFDPEGKTQDELDKEAEEAELDKKADELQNKVADLEKEISNLEILLGGADSEDDTAALQNKLATKKAELEKTQKELDAALNELGPDGDE'
    # LEKAEAELENLLSTLDPEGKTQDELDKEAAEAELNKKVEALQNQVAELEEELSKLEDNLKDAETNNVEDYIKEGLEEAIATKKAELEKTQKELDAALNELGPDGDEs2 = 'LEKAEAELENLLSTLDPEGKTQDELDKEAAEAELNKKVEALQNQVAELEEELSKLEDNLKDAETNNVEDYIKEGLEEAIATKKAELEKTQKELDAALNELGPDGDE'
    print(len(s1))