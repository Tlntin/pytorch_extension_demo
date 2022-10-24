### pytorch加载扩展的三种方法
- setup: 直接将扩展链接或者安装为模块，然后导入模块使用函数，速度最快
- load: 使用时再编译，编译后再导入，速度较慢
- load_inline: 基本和load没啥区别，只不过从文件变成了字符串，由pytorch负责添加一些专用头文件

### 准备工作
- 已安装python, cuda, pytorch, 且pytorch为gpu版
- 安装所需模块
```bash
pip install -r requirements.txt
```

### 使用方法：
- setup安装:
```bash
# 以链接方式编译
python3 setup.py build develop
# 直接安装为模块
python3 setup.py install # 或者pip install .
```
- setup方式运行扩展
```bash
python3 test/test.py
```

- load方式运行扩展
```bash
cd test
python3 test/test2.py
```

- load_inline方式运行扩展 
```bash
cd test
python3 test/test3.py
```


### 参考文档：
- [torch官网文档](https://pytorch.org/docs/master/cpp_extension.html)
- [pybind11文档](https://pybind11.readthedocs.io/en/stable/basics.html)
- [知乎](https://zhuanlan.zhihu.com/p/358220419)