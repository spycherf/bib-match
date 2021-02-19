#!/bin/bash

program="xwc/generate_labels_by_cluster.py"

p1() {
  python $program -r 0-40000
}
p2() {
  python $program -r 40000-80000
}
p3() {
  python $program -r 80000-120000
}
p4() {
  python $program -r 120000-160000
}
p5() {
  python $program -r 160000-200000
}
p6() {
  python $program -r 200000-240000
}
p7() {
  python $program -r 240000-280000
}
p8() {
  python $program -r 280000-320000
}
p9() {
  python $program -r 320000-360000
}
p10() {
  python $program -r 360000-400000
}
p11() {
  python $program -r 400000-440000
}
p12() {
  python $program -r 440000-500000
}

(trap "kill 0" SIGINT; p1 & p2 & p3 & p4 & p5 & p6 & p7 & p8 & p9 & p10 & p11 & p12)
