#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   test.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/5/8 下午9:04   aleaho      1.0        

"""
文档说明：


"""
import subprocess
command = "python conlleval.py < predictprediction_0.csv"
process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
result = process.communicate()[0].decode("utf-8")
print(result)



