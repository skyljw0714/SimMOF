import glob
import pubchempy as pcp
import os
import re
import shutil
import sys
import time 

files = glob.glob('./*.def')

def name2smiles(name): #name or formula
    results = pcp.get_compounds(name, 'name')
    if len(results) == 1:
        smiles =  results[0].isomeric_smiles
        print(smiles)
        return smiles
    else:
        print('more than 2 compounds')
        print(name)

def find_alkanes(text):
    pattern = r'\bC\d+\b'  # 'C' 뒤에 하나 이상의 숫자가 오는 패턴
    matches = re.findall(pattern, text)
    return matches

def copy_file(original_file, new_file):
    try:
        shutil.copyfile(original_file, new_file)
        print(f"File {original_file} copied successfully to {new_file}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except:
        print("Unexpected error:", sys.exc_info())


def get_compounds(name):
    results = pcp.get_compounds(name, 'name')
    if not results:
        try:
            print('not found in compound name, it may be formula...')
            results = pcp.get_compounds(name, 'formula')
        except Exception as e:
            return None
    return results

'''
alkanes = []
for f in files:
    name = f.split('.def')[0].split('./')[-1]
    time.sleep(5)
    if find_alkanes(name):
        continue
        num_c = int(name.split('C')[-1])
        formula = 'C{}H{}'.format(num_c, num_c*2+2)
        copy_file(f, '../new/{}.def'.format(formula))
    else:
        results = get_compounds(name)
        if not results: 
            print('no compunds')
        elif len(results) == 1:
            formula = results[0].molecular_formula
            copy_file(f, '../new/{}.def'.format(formula))
        elif len(results) > 1:
            print('more than 2 results, {}'.format(name)) 
'''


def process_files(files, processed_files):
    for f in files:
        name = f.split('.def')[0].split('./')[-1]
        if name in processed_files:
            continue  # 이미 처리된 파일은 건너뜀

        try:
            time.sleep(5)  # 요청 사이에 지연 시간
            if find_alkanes(name):
                num_c = int(name.split('C')[-1])
                formula = 'C{}H{}'.format(num_c, num_c*2+2)
                copy_file(f, '../new/{}.def'.format(formula))
            else:
                results = get_compounds(name)
                if not results:
                    print('no compounds')
                elif len(results) == 1:
                    formula = results[0].molecular_formula
                    copy_file(f, '../new/{}.def'.format(formula))
                elif len(results) > 1:
                    print('more than 2 results, {}'.format(name))
            processed_files.add(name)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            break

# 파일 목록과 처리된 파일의 집합
processed_files = set()

# 프로세스 실행
process_files(files, processed_files)

# 처리된 파일 목록 저장
with open('processed_files.txt', 'w') as file:
    for name in processed_files:
        file.write(name + '\n')
       
    
