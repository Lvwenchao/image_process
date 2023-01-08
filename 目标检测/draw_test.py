# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2022/12/28 12:23
# @FileName : draw_test.py
# @Software : PyCharm
import xml.etree.cElementTree as ET

tree = ET.ElementTree(file="./test.xml")
root = tree.getroot()
# 节点标签和属性
print(root["filename"].tag, root[1].text)
