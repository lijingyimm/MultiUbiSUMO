import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

def getsitepssm(seqpssm,i,win):
    if win <= i  and  len(seqpssm) - win -1 >= i :
        return seqpssm[i-win :i+win+1]  
    elif win > i and len(seqpssm) - win - 1 >= i :
        #left short
        pssm = []
        for j in range(win-i):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        pssm.extend(seqpssm[:i+win+1])
        return pssm
    elif win <= i and len(seqpssm) - 1 - win < i :
        #right short
        pssm = []
        pssm = seqpssm[i-win :len(seqpssm)]
        for j in range(win+i - len(seqpssm) + 1):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return pssm
    else: 
        #left and right short
        pssm = []
        for j in range(win-i):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])        
        pssm.extend(seqpssm)
        for j in range(win+i - len(seqpssm) + 1):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])   
        return pssm 

def subSeqq(seq,id,num):
    
    win = num
    subSeq = ''
    if (id-win)<0 and (id + win)>len(seq):
        for i in range(win-id):
            subSeq+='B'
        for i in range(0,len(seq)):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='B'
    elif (id-win)<0 and (id+win+1)<=len(seq):
        for i in range(win-id):
            subSeq+='B'
        for i in range(0,id+win+1):
            subSeq+=seq[i]
    elif (id-win)>=0 and (id+win+1) > len(seq):
        for i in range(id-win,len(seq)):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='B'
    elif (id-win)>=0 and (id+win+1) <= len(seq):
        for i in range(id-win,id+win+1):
            subSeq+=seq[i]
    return subSeq    


def find(i,id_num):
    for j in range(len(id_num)):
        if (id_num[j]==i):
            return True
    return False


def getpssmfeature(unpac,pssmfilepath):
    
    pssmpicklepath = pssmfilepath + unpac + '.pickle'
    picklefile = open(pssmpicklepath,'rb')
    oneseqpssm = pickle.load(picklefile)
    picklefile.close()
    
    return oneseqpssm  ##### return one sequence pssm binary #####


def read_fasta(fasta_file,windown_length,pssmfilepath,label):
    
    ##### fasta_file: string #####
    ##### windown_length = 24 #####
    ##### pssmfilepath: '/pssmpickle/' #####
    ##### lable: train=true  test=flase #####???
    
    oneofkey_0 = []
    oneofkey_1 = []
    oneofkey_2 = []
    oneofkey_3 = []
    pssm_pos = []
    pssm_neg = []
    num = 0
    id_num = []
    id_list=[]
    name_list=[]
    sum1=0
    sum0=0
    sum2=0
    sum3=0
    with open(fasta_file, 'r') as fp:
        win = windown_length
        neg = 0    #负样本个数
        pos_1 = 0    #正样本个数
        pos_2 = 0
        #aseqpssm = ''
        wrr = 0
        for line in fp:
            flag = 0
            num += 1
            if (num == 2):
                line = line.replace('  ','\t')
                line = line.replace(' ','\t')
                line1 = line.split('\t')
                name1 = line1[0]    #序列名
                # print(name1)
                id = int(line1[1])  #位置
                type1 = line1[2]     #PTM类型  ubi或sumo
                seq1 = line1[3]     #fasta序列
                id_num.append(id-1)   #位点

                del line1           #删除变量


            elif (num > 2):
                line = line.replace('  ','\t')
                line = line.replace(' ','\t')            
                line1 = line.split('\t')
                name = line1[0]   #第二个序列名
                # print(name)
                type = line1[2]   #PTM类型
                # print(type)
                seq = line1[3]    #序列

                if (name == name1 and type1 =='UBI'):     #若为同一序列，加上该位置
                    id_num.append(int(line1[1])-1)
                    # print("id_num_ubi",id_num)

                elif(name != name1 and type1 =='UBI'):
                    #del aseqpssm
                    #aseqpssm = getpssmfeature(name1,pssmfilepath)
                    for i in range(len(seq1)):     #循环序列中位置
                        if (seq1[i]=='K' and find(i, id_num)):
                            wrr += 10
                            pos_1 += 1
                            subSeq = subSeqq(seq1, i, win)  #截窗
                            # print(subSeq)
                            # with open('result/sq1.txt', mode='a') as resFile:
                            #     resFile.write(str(subSeq)+'\r\n')

                            if(type == type1):    
                                final_seq = [1,0] + [AA for AA in subSeq]
                                sum1+=1
                                # print('sum1: ',sum1)
                                oneofkey_1.append(final_seq)
                                del subSeq, final_seq
                            # else:
                            #     final_seq = [1,1] + [AA for AA in subSeq]
                            #     sum3+=1
                            #     print('sum333333333333333333333:',sum3)
                            #     flag=1
                            #     oneofkey_3.append(final_seq)
                            #     del subSeq, final_seq
                                                              
                        elif (seq1[i]=='K' and not find(i, id_num)):   #负样本
                            neg += 1
                            subSeq = subSeqq(seq1, i, win)
                            final_seq = [0,0] + [AA for AA in subSeq]       #0     none
                            sum0 += 1
                            # print('sum0',sum0)
                            oneofkey_0.append(final_seq)
                            del subSeq, final_seq
  

                    wrr = 0
                    id_num = []
                    name1 = name
                    seq1 = seq
                    type1 = type
                    id_num.append(int(line1[1])-1)
                    if(flag):
                        line = line.replace(line,'')     #去掉已经被标记为3的位置和序列
                    del line1
                    del line

                elif (name == name1 and type1 =='SUMO'):     #若为同一序列，加上该位置
                    id_num.append(int(line1[1])-1)
                    # print("id_num_sumo",id_num)
                    del line1
                elif(name != name1 and type1 =='SUMO'):
                    for i in range(len(seq1)):     #循环序列中位置
                        if (seq1[i]=='K' and find(i, id_num)):
                            wrr += 10
                            pos_2 += 1
                            subSeq = subSeqq(seq1, i, win)  #截窗
                            # print(subSeq)
                            # with open('result/sq2.txt', mode='a') as resFile:
                            #     resFile.write(str(subSeq)+'\r\n')
                            final_seq = [0,1] + [AA for AA in subSeq]       #sumo
                            sum2+=1
                            # print('sum2: ',sum2)
                            oneofkey_2.append(final_seq)
                            del subSeq, final_seq

                        elif (seq1[i]=='K' and not find(i, id_num)):   #负样本
                            neg += 1
                            subSeq = subSeqq(seq1, i, win)
                            final_seq = [0,0] + [AA for AA in subSeq]
                            oneofkey_0.append(final_seq)
                            sum0+=1
                            # print('sum0: ',sum0)
                            del subSeq, final_seq


                    wrr = 0
                    id_num = []
                    name1 = name
                    seq1 = seq
                    id_num.append(int(line1[1])-1)

                    del line1
                    del line


        ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
        for i in range(len(seq1)):   #最后一个序列处理
            if (seq1[i]=='K' and find(i, id_num)):           
                pos_2 += 1
                subSeq = subSeqq(seq1, i, win)
                final_seq = [0,1] + [AA for AA in subSeq]
                sum2+=1
                # print('sum2: ',sum2)
                oneofkey_2.append(final_seq)
                del subSeq,final_seq


            elif (seq1[i]=='K' and not find(i,id_num)):              
                neg += 1
                subSeq = subSeqq(seq1,i,win)
                final_seq = [0,0] + [AA for AA in subSeq]  
                sum0+=1
                # print('sum0: ',sum0)
                oneofkey_0.append(final_seq)
                del subSeq,final_seq
    with open(fasta_file, "r") as f:
        k = 0
        i = 0
        for line in f.readlines():
            i += 1
            if i == 1:
                continue
            line = line.replace('  ','\t')
            line = line.replace(' ','\t')
            name = line.split('\t', 4)[0] #去掉列表中每一个元素的换行符
            num = line.split('\t', 4)[1] #去掉列表中每一个元素的换行符
            type = line.split('\t', 4)[2] #去掉列表中每一个元素的换行符
            seq = line.split('\t', 4)[3] #去掉列表中每一个元素的换行符
            if type =='CROSS':
                k += 1
                # print(name, num)
                subSeq = subSeqq(seq, int(num)-1, win)  #截窗
                f_seq = [1,1] + [AA for AA in subSeq]
                # flag=1
                # print(f_seq)
                oneofkey_3.append(f_seq)
                del subSeq, f_seq

            # if type =='UBI':
            #     k += 1
            #     print(name, num)
            #     subSeq = subSeqq(seq, int(num)-1, win)  #截窗
            #     f_seq = [1,1] + [AA for AA in subSeq]
            #     # flag=1
            #     print(f_seq)
            #     oneofkey_1.append(f_seq)
            #     del subSeq, f_seq

            # if type =='SUMO':
            #     k += 1
            #     print(name, num)
            #     subSeq = subSeqq(seq, int(num)-1, win)  #截窗
            #     f_seq = [1,1] + [AA for AA in subSeq]
            #     # flag=1
            #     print(f_seq)
            #     oneofkey_2.append(f_seq)
            #     del subSeq, f_seq


        # print(k)


    # with open(fasta_file, "r") as f:
    #     k = 0
    #     i = 0
    #     for line in f.readlines():
    #         i += 1
    #         if i == 1:
    #             continue

    #         name = line.split('\t', 4)[0] #去掉列表中每一个元素的换行符
    #         num = line.split('\t', 4)[1] #去掉列表中每一个元素的换行符
    #         type = line.split('\t', 4)[2] #去掉列表中每一个元素的换行符
    #         seq = line.split('\t', 4)[3] #去掉列表中每一个元素的换行符
    #         if type =='UBI':
    #             j=0
    #             with open(fasta_file, "r") as f1:
    #                 for line1 in f1.readlines():
    #                     j += 1
    #                     if j == 1:
    #                         continue
    #                     name1 = line1.split('\t', 4)[0] #去掉列表中每一个元素的换行符
    #                     num1 = line1.split('\t', 4)[1] #去掉列表中每一个元素的换行符
    #                     type1 = line1.split('\t', 4)[2] #去掉列表中每一个元素的换行符
    #                     if type1 == 'SUMO':
    #                         if name == name1 and num == num1:
    #                             k += 1
    #                             print(name, num)
    #                             subSeq = subSeqq(seq, int(num)-1, win)  #截窗
    #                             f_seq = [1,1] + [AA for AA in subSeq]
    #                             # flag=1
    #                             print(f_seq)
    #                             oneofkey_3.append(f_seq)
    #                             del subSeq, f_seq
    #     print(k)
        return oneofkey_0,oneofkey_1,oneofkey_2,oneofkey_3
        #return oneofkey_pos,oneofkey_neg,pssm_pos,pssm_neg,oneofkey_pos,oneofkey_neg
    # print('1,2,3',sum0,sum1,sum2)

def read_fasta_test(fasta_file,windown_length,pssmfilepath,label):
    
    ##### fasta_file: string #####
    ##### windown_length = 24 #####
    ##### pssmfilepath: '/pssmpickle/' #####
    ##### lable: train=true  test=flase #####???

    oneofkey_0 = []
    oneofkey_1 = []
    oneofkey_2 = []
    oneofkey_3 = []
    pssm_pos = []
    pssm_neg = []
    num = 0
    id_num = []
    id_list=[]
    name_list=[]
    sum1=0
    sum0=0
    sum2=0
    sum3=0
    with open(fasta_file, 'r') as fp:
        win = windown_length
        neg = 0    #负样本个数
        pos_1 = 0    #正样本个数
        pos_2 = 0
        #aseqpssm = ''
        wrr = 0
        for line in fp:
            flag = 0
            num += 1
            if (num == 2):
                line = line.replace('  ','\t')
                line = line.replace(' ','\t')
                line1 = line.split('\t')
                name1 = line1[0]    #序列名
                name_list.append(name1)

                # print(name1)
                id = int(line1[1])  #位置
                id_list.append(id)
                type1 = line1[2]     #PTM类型  ubi或sumo
                seq1 = line1[3]     #fasta序列
                id_num.append(id-1)   #位点

                del line1           #删除变量


            elif (num > 2):
                line = line.replace('  ','\t')
                line = line.replace(' ','\t')
                line1 = line.split('\t')
                name = line1[0]   #第二个序列名
                name_list.append(name)
                id_list.append(int(line1[1]))
                type = line1[2]   #PTM类型
                seq = line1[3]    #序列

                if (name == name1 and type1 =='UBI'):     #若为同一序列，加上该位置
                    id_num.append(int(line1[1])-1)
                    # print("id_num_ubi",id_num)

                elif(name != name1 and type1 =='UBI'):
                    #del aseqpssm
                    #aseqpssm = getpssmfeature(name1,pssmfilepath)
                    for i in range(len(seq1)):     #循环序列中位置
                        if (seq1[i]=='K' and find(i, id_num)):
                            wrr += 10
                            pos_1 += 1
                            subSeq = subSeqq(seq1, i, win)  #截窗
                            # print(subSeq)
                            # with open('result/sq1.txt', mode='a') as resFile:
                            #     resFile.write(str(subSeq)+'\r\n')

                            if(type == type1):    
                                final_seq = [1,0] + [AA for AA in subSeq]
                                sum1+=1
                                # print('sum1: ',sum1)
                                oneofkey_1.append(final_seq)
                                del subSeq, final_seq
                            # else:
                            #     final_seq = [1,1] + [AA for AA in subSeq]
                            #     sum3+=1
                            #     print('sum333333333333333333333:',sum3)
                            #     flag=1
                            #     oneofkey_3.append(final_seq)
                            #     del subSeq, final_seq
                                                              
                        elif (seq1[i]=='K' and not find(i, id_num)):   #负样本
                            neg += 1
                            subSeq = subSeqq(seq1, i, win)
                            final_seq = [0,0] + [AA for AA in subSeq]       #0     none
                            sum0 += 1
                            # print('sum0',sum0)
                            oneofkey_0.append(final_seq)
                            del subSeq, final_seq
  

                    wrr = 0
                    id_num = []
                    name1 = name
                    seq1 = seq
                    type1 = type
                    id_num.append(int(line1[1])-1)
                    if(flag):
                        line = line.replace(line,'')     #去掉已经被标记为3的位置和序列
                    del line1
                    del line

                elif (name == name1 and type1 =='SUMO'):     #若为同一序列，加上该位置
                    id_num.append(int(line1[1])-1)
                    # print("id_num_sumo",id_num)
                    del line1
                elif(name != name1 and type1 =='SUMO'):
                    for i in range(len(seq1)):     #循环序列中位置
                        if (seq1[i]=='K' and find(i, id_num)):
                            wrr += 10
                            pos_2 += 1
                            subSeq = subSeqq(seq1, i, win)  #截窗
                            # print(subSeq)
                            # with open('result/sq2.txt', mode='a') as resFile:
                            #     resFile.write(str(subSeq)+'\r\n')
                            final_seq = [0,1] + [AA for AA in subSeq]       #sumo
                            sum2+=1
                            # print('sum2: ',sum2)
                            oneofkey_2.append(final_seq)
                            del subSeq, final_seq

                        elif (seq1[i]=='K' and not find(i, id_num)):   #负样本
                            neg += 1
                            subSeq = subSeqq(seq1, i, win)
                            final_seq = [0,0] + [AA for AA in subSeq]
                            oneofkey_0.append(final_seq)
                            sum0+=1
                            # print('sum0: ',sum0)
                            del subSeq, final_seq


                    wrr = 0
                    id_num = []
                    name1 = name
                    seq1 = seq
                    id_num.append(int(line1[1])-1)

                    del line1
                    del line


        ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
        for i in range(len(seq1)):   #最后一个序列处理
            if (seq1[i]=='K' and find(i, id_num)):           
                pos_2 += 1
                subSeq = subSeqq(seq1, i, win)
                final_seq = [0,1] + [AA for AA in subSeq]
                sum2+=1
                # print('sum2: ',sum2)
                oneofkey_2.append(final_seq)
                del subSeq,final_seq


            elif (seq1[i]=='K' and not find(i,id_num)):              
                neg += 1
                subSeq = subSeqq(seq1,i,win)
                final_seq = [0,0] + [AA for AA in subSeq]  
                sum0+=1
                # print('sum0: ',sum0)
                oneofkey_0.append(final_seq)
                del subSeq,final_seq

    with open(fasta_file, "r") as f:
        k = 0
        i = 0
        for line in f.readlines():
            i += 1
            if i == 1:
                continue
            line = line.replace('  ','\t')
            line = line.replace(' ','\t')
            name = line.split('\t', 4)[0] #去掉列表中每一个元素的换行符
            num = line.split('\t', 4)[1] #去掉列表中每一个元素的换行符
            type = line.split('\t', 4)[2] #去掉列表中每一个元素的换行符
            seq = line.split('\t', 4)[3] #去掉列表中每一个元素的换行符
            if type =='CROSS':
                k += 1
                # print(name, num)
                subSeq = subSeqq(seq, int(num)-1, win)  #截窗
                f_seq = [1,1] + [AA for AA in subSeq]
                # flag=1
                # print(f_seq)
                oneofkey_3.append(f_seq)
                del subSeq, f_seq
        # print(k)

    # with open(fasta_file, "r") as f:
    #     k = 0
    #     i = 0
    #     for line in f.readlines():
    #         i += 1
    #         if i == 1:
    #             continue

    #         name = line.split('\t', 4)[0] #去掉列表中每一个元素的换行符
    #         num = line.split('\t', 4)[1] #去掉列表中每一个元素的换行符
    #         type = line.split('\t', 4)[2] #去掉列表中每一个元素的换行符
    #         seq = line.split('\t', 4)[3] #去掉列表中每一个元素的换行符
    #         if type =='UBI':
    #             j=0
    #             with open(fasta_file, "r") as f1:
    #                 for line1 in f1.readlines():
    #                     j += 1
    #                     if j == 1:
    #                         continue
    #                     name1 = line1.split('\t', 4)[0] #去掉列表中每一个元素的换行符
    #                     num1 = line1.split('\t', 4)[1] #去掉列表中每一个元素的换行符
    #                     type1 = line1.split('\t', 4)[2] #去掉列表中每一个元素的换行符
    #                     if type1 == 'SUMO':
    #                         if name == name1 and num == num1:
    #                             k += 1
    #                             print(name, num)
    #                             subSeq = subSeqq(seq, int(num)-1, win)  #截窗
    #                             f_seq = [1,1] + [AA for AA in subSeq]
    #                             print(f_seq)
    #                             # flag=1
    #                             oneofkey_3.append(f_seq)
    # #                             del subSeq, f_seq
    #     print(k)
        return oneofkey_0,oneofkey_1,oneofkey_2,oneofkey_3,name_list,id_list

def get_data(string,pssmfilepath,label):
    
    ##### string: 'Ubisite_test.txt' #####
    ##### pssmfilepath: '/pssmpickle/' #####
    ##### lable: train=true  test=flase #####
    
    winnum = 24
    oneofkey_0,oneofkey_1,oneofkey_2,oneofkey_3 = read_fasta(string, winnum, pssmfilepath, label)
    #oneofkey_pos,oneofkey_neg,pssm_pos,pssm_neg,physical_pos,physical_neg = read_fasta(string,winnum,pssmfilepath,label)
    
    #oneofkey_pos = pd.DataFrame(oneofkey_pos)
    #oneofkey_pos = oneofkey_pos.as_matrix()
    
    #oneofkey_neg = pd.DataFrame(oneofkey_neg)
    #oneofkey_neg = oneofkey_neg.as_matrix()
    
    return oneofkey_0,oneofkey_1,oneofkey_2,oneofkey_3

def get_data_test(string,pssmfilepath,label):
    
    ##### string: 'Ubisite_test.txt' #####
    ##### pssmfilepath: '/pssmpickle/' #####
    ##### lable: train=true  test=flase #####
    
    winnum = 24
    oneofkey_0,oneofkey_1,oneofkey_2,oneofkey_3,name_list,id_list = read_fasta_test(string, winnum, pssmfilepath, label)
    #oneofkey_pos,oneofkey_neg,pssm_pos,pssm_neg,physical_pos,physical_neg = read_fasta(string,winnum,pssmfilepath,label)
    
    #oneofkey_pos = pd.DataFrame(oneofkey_pos)
    #oneofkey_pos = oneofkey_pos.as_matrix()
    
    #oneofkey_neg = pd.DataFrame(oneofkey_neg)
    #oneofkey_neg = oneofkey_neg.as_matrix()
    
    return oneofkey_0,oneofkey_1,oneofkey_2,oneofkey_3,name_list,id_list
    #return oneofkey_pos,oneofkey_neg,pssm_pos,pssm_neg,physical_pos,physical_neg 

# def load_data(poswindows_filepath,negwindows_filepath):
#     seq = ''
#     windows_pos = []
#     windows_neg = []
#     with open (poswindows_filepath,"r") as fp:
#         for line in fp:
#             seq = line.rstrip("\n")
#             pos_sequence = [1] + [aa for aa in seq]
#             windows_pos.append(pos_sequence)
#     del pos_sequence

#     with open (negwindows_filepath, "r") as fp:
#         for line in fp:
#             seq = line.rstrip("\n")
#             neg_sequence = [0] + [aa for aa in seq]
#             windows_neg.append(neg_sequence)
#     del neg_sequence

    # print('windows_pos:', windows_pos)
    # print('windows_neg:', windows_neg)
    # return windows_pos,windows_neg

if __name__ == '__main__':

    g=getdata()

#
# all_train_windows_pos,all_train_windows_neg = get_data(r'data/multidata/realtrain0.txt',r'data/pssmpickle2/',label= True)


#print(all_train_windows_pos)
#print(all_train_windows_neg)
