import xml.etree.cElementTree as ET
from xml.etree.ElementTree import parse
import numpy as np

file = "../traffic-matrices/IntraTM-2005-01-01-00-30.xml"
doc = parse(file)
root = doc.getroot()
print(doc)
# tree = ET.ElementTree(file)
# root = tree.getroot()

print("root:",root)

print(root.tag, root.attrib)

num_id = 23

values = []
ids = []
srcs = []
l = 0
mat_dict = {}
for data in root.findall('IntraTM'):
    for ii,item in enumerate(data.findall("src")):
        
        srcs.append(item.attrib["id"])
        src_id = item.attrib["id"]
        # tmp_dict = {src_id:[]}
        
        
        tmp_dict = {}
        for iii,dst in enumerate(item.findall("dst")):
            value = float(dst.text)
            Id  = dst.attrib["id"]
            tmp_dict[Id] = value

        tmp_list = []
        # print(len(tmp_dict))
        for k in range(num_id):
            try:
                tmp_list.append(tmp_dict[str(k+1)])
            except:
                tmp_list.append(-1)

        mat_dict[src_id] = tmp_list
# print(mat_dict) 

TM = []
for i in range(num_id):
    TM.append(mat_dict[str(i+1)])

TM = np.array(TM)
print(TM)
print(TM.shape)
# print(tmp_list)
# print(srcs)
# print(ids)
# print(len(values))