#!/usr/bin/python
import re, argparse
from pathlib import Path

##############################################################
#                                                            #
#    This script removes the redundant primitives from       #
#    Dunning basis sets following the procedure outlined in  #
#    Chem. Phys. Lett., 260 (1996) 514-18                    #
#                                                            #
#    Input: The basis set file from "Basis Set Exchange"     #
#           with the headers removed along with the empty    #
#           lines above "H     0". You can use any           #
#           combination of atoms                             #
#                                                            #
#    Output: file with modified primitives (default:out.txt) #
#                                                            #
#    Note: In case of an out.txt file alread present it      #
#          will be overwritten                               #
#          Compatible up to G functions (trivial to extend)  #
#                                                            #
##############################################################

tol = 1.0E-10
thresh = 1.0E-3

def eliminate(original):
    # list of elements that can be discarded
    elim = [ [] for x in original ]
    
    # the original coefficients are copied before being modified
    modified = [row[:] for row in original]
    
    nelems = len(original[0])
    
    row_consider = True
    
    for indi,i in enumerate(modified):
        pivot = 0
        found_scale = False
        for indj,j in enumerate(i):
            if abs(j) < tol and indj not in elim[indi]:
                elim[indi].append(indj)
        for indk,k in enumerate(modified):
            if indk > indi:
                for indj,j in enumerate(k):
                    if abs(j) < tol and indj not in elim[indk]:
                        elim[indk].append(indj)
                for l in elim[indk]:
                    if l not in elim[indi]:
                        row_consider = False
                        break
                if not row_consider:
                    row_consider = True
                    continue
                if not found_scale:
                    for m in range(nelems):
                        if m not in elim[indi] and abs(i[m]) > tol and abs(k[m]) > thresh:
                            scale = k[m]/i[m]
                            pivot = m
                            found_scale = True
                            break
                else:
                    scale = k[pivot]/i[pivot]
                for m in range(nelems):
                    modified[indk][m] = modified[indk][m] - scale*modified[indi][m]
    
    elim = [ x for x in list(reversed(elim)) ]
    
    row_consider = True
    
    for indi,i in enumerate(list(reversed(modified))):
        for indj,j in enumerate(i):
            if abs(j) < tol and indj not in elim[indi]:
                elim[indi].append(indj)
        for indk,k in enumerate(list(reversed(modified))):
            if indk > indi:
                for indj,j in enumerate(k):
                    if abs(j) < tol and indj not in elim[indk]:
                        elim[indk].append(indj)
                for l in elim[indk]:
                    if l not in elim[indi]:
                        row_consider = False
                        break
                if not row_consider:
                    row_consider = True
                    continue
                for m in reversed(range(nelems)):
                    if m not in elim[indi] and abs(i[m]) > tol:
                        scale = k[m]/i[m]
                        break
                for m in reversed(range(nelems)):
                    modified[(-1)*(indk+1)][m] = modified[(-1)*(indk+1)][m] - scale*modified[(-1)*(indi+1)][m]

    elim = [ x for x in list(reversed(elim)) ]
    
    return modified, elim
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="Name of input file. Input file contains the basis set of elements (can be one or more). If you taking the file from basis set exchange, use Gaussian format and remove all the header and empty lines above H     0",type=str)
    parser.add_argument("-o","--output",help="Output file to write your general linear transformed Dunning basis set",default="out.txt",type=str)
    args=parser.parse_args()

    output_file = getattr(args,"output")

    print("printing the transformed Dunning basis set to '"+output_file+"'. "+output_file+" will be overwritten.\nPress enter if that is okay.")
    input()

    input_file = getattr(args,"input") or input("input file with dunning basis sets: ")

    alllines = open(input_file,'r').readlines()

    file = Path(output_file)

    if file.exists():
        file.write_text("")   # truncates
    else:
        file.touch()          # creates empty file

    lines = []
    for allline in alllines:
        if allline != '****\n':
            lines.append(allline)
        else:
            shellexps = {'S':[],'P':[],'D':[],'F':[],'G':[]}
            shellmaxprim = {'S':0,'P':0,'D':0,'F':0,'G':0}
            shellcoeffs = {'S':[],'P':[],'D':[],'F':[],'G':[]}
            
            for ind,line in enumerate(lines):
                if re.search('[A-Z]\s+\d+\s+1.00',line):
                    shellnew = re.search('([A-Z])\s+\d+\s+1.00',line).group(1)
                    nprim = int(re.search('[A-Z]\s+(\d+)\s+1.00',line).group(1))
                    shellexps[shellnew] = shellexps[shellnew] + [ float(re.sub('D','E',x.strip().split()[0])) for x in lines[ind+1:ind+1+nprim] ]
            
            for j in list(shellexps.keys()):
                shellexps[j] = list(set(shellexps[j]))
                shellexps[j].sort(reverse=True)
                shellmaxprim[j] = len(shellexps[j])
            
            for ind,line in enumerate(lines):
                if re.search('[A-Z]\s+\d+\s+1.00',line):
                    shellnew = re.search('([A-Z])\s+\d+\s+1.00',line).group(1)
                    nprim = int(re.search('[A-Z]\s+(\d+)\s+1.00',line).group(1))
                    if shellmaxprim[shellnew] != 0:
                        shellcoeffs[shellnew].append([ 0.0 for i in range(shellmaxprim[shellnew]) ])
                        for x in lines[ind+1:ind+1+nprim]:
                            expnew = float(re.sub('D','E',x.strip().split()[0]))
                            for ind1,exp in enumerate(shellexps[shellnew]):
                                if exp == expnew:
                                    shellcoeffs[shellnew][-1][ind1] = float(re.sub('D','E',x.strip().split()[1]))
                                    break
            
            modifiedcoeffs = {'S':[],'P':[],'D':[],'F':[],'G':[]}
            elimcoeffs = {'S':[],'P':[],'D':[],'F':[],'G':[]}
            
            with open(output_file,'a') as fh:
                fh.write(lines[0])
                for j in list(shellexps.keys()):
                    if shellmaxprim[j] != 0:
                        modifiedcoeffs[j], elimcoeffs[j] = eliminate(shellcoeffs[j])
                        for i in range(len(shellcoeffs[j])):
                            fh.write(j+'    '+str(len(modifiedcoeffs[j][i])-len(elimcoeffs[j][i]))+'   1.00\n')
                            for ind,line in enumerate(modifiedcoeffs[j][i]):
                                if ind not in elimcoeffs[j][i]:
                                    fh.write(str(f"{shellexps[j][ind]:18.6E}").replace("E", "D")+str(f"{modifiedcoeffs[j][i][ind]:23.6E}").replace("E", "D")+'\n')
                fh.write('****\n')
            lines = []

