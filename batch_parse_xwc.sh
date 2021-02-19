#!/bin/bash

parser="xwc/retrieve_bib_by_offset.py"

p1() {
  python $program -r 0-59
}
p2() {
  python $program -r 60-119
}
p3() {
  python $program -r 120-179
}
p4() {
  python $program -r 180-239
}
p5() {
  python $program -r 240-319
}
p6() {
  python $program -r 320-379
}
p7() {
  python $program -r 380-399
}

(trap "kill 0" SIGINT; p1 & p2 & p3 & p4 & p5 & p6 & p7)
