#include "add.h"

void launch_add2_cpu(const float * x, const float * y, float * z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}