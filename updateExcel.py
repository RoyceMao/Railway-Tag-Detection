#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import pandas as pd
import numpy as np
import os




class DealExcel:
    def __init__(self):
        self.file_dir = os.listdir('E:\list')

    def read_excel(self,from_filename,to_filename, vendor):
        dt = pd.read_excel(from_filename, sheet_name=0, index_col=None)
        # print (dt.ix[-1:])

        dt = dt[[u'条码', u'书名', u'定价', u'折扣', u'出版社', u'出版时间', u'图书编号', u'库存']]
        dt[u'商家'] = vendor
        print(dt)
        return dt

    def main(self):
        for f in self.file_dir:
            if 'xls' in f and f == '208.xls':
                # print (f)
                vendor = int(f.split('.')[0])
                from_filename = 'E:\list' + '\\' + str(f)
                to_filename = 'E:\list\csv' + '\\' + str(vendor) + '.xls'
                dt = self.read_excel(from_filename,to_filename, vendor)
                # 删除特定行并写入指定目录
                print(dt['条码'].isin(["www"]))
                dt = dt[ ~ dt['条码'].isin(['【www.taobao.com】'])]  # 删除某列包含特殊字符的行
                dt.to_excel(to_filename, index=False)  # 将数据重新写入excel


if __name__ == "__main__":
    a = DealExcel()
    a.main()
