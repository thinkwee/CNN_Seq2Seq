def show_param(net):
    """统计参数"""
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构： " + str(list(i.size())), end="   ")
        for j in i.size():
            l *= j
        print("该层参数和： " + str(l))
        k = k + l
    print("总参数数量和： " + str(k))
    print("占用总内存： " + str(float(k) * 4 / 1024 / 1024) + " MB")
