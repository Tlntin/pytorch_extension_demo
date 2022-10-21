

void launch_add2_cpu(const float * x, const float * y, float * z, int n);
void launch_add2_gpu(const float * x, const float * y, float * z, int n);
/*
求和函数，用于x与y相加
x, y，z均为float数组
其中 z = x + y
n为整数，代表数组长度
*/

