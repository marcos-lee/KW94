import delimited "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\output\T40_1aprox_s2000.csv", clear
sort ev1val ev2val ev3val ev4val


gen a = 1 if ev1val >= 20548 & ev1val <= 20549
gen b = 1 if ev3val == -4000
keep if a == 1 & b == 1

twoway (scatter yhat ev2val, sort) ///
(scatter yf ev2val, sort)
graph export "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\stata\figure11.png", as(png) replace


twoway (scatter emax ev2val, sort) ///
(scatter maxe ev2val, sort) ///
(scatter emaxhat ev2val, sort)

import delimited "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\output\df2aprox.csv", clear

gen a = 1 if ev1val >= 22211 & ev1val <= 22212
gen b = 1 if ev3val >= -10107 & ev3val <= -10106

keep if a == 1 & b == 1

twoway (scatter yhat ev2val, sort) ///
(scatter y ev2val, sort)


twoway (scatter emaxap ev2val, sort) ///
(scatter maxeap ev2val, sort) ///
(scatter emaxhat ev2val, sort)



import delimited "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\output\df3aprox.csv", clear

gen a = 1 if ev1val >= 19053 & ev1val <= 19054
gen b = 1 if ev3val >= -15125 & ev3val <= -15124

keep if a == 1 & b == 1

twoway (scatter yhat ev2val, sort) ///
(scatter y ev2val, sort)


twoway (scatter emaxap ev2val, sort) ///
(scatter maxeap ev2val, sort) ///
(scatter emaxhat ev2val, sort)

