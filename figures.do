import delimited "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\df1aprox.csv", clear

gen a = 1 if ev1val >= 20597 & ev1val <= 20598
gen b = 1 if ev3val >= -4027 & ev3val <= -4026

keep if a == 1 & b == 1

twoway (scatter yhat ev2val, sort) ///
(scatter y ev2val, sort)


twoway (scatter emaxap ev2val, sort) ///
(scatter maxeap ev2val, sort) ///
(scatter emaxhat ev2val, sort)

import delimited "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\df2aprox.csv", clear

gen a = 1 if ev1val >= 22211 & ev1val <= 22212
gen b = 1 if ev3val >= -10107 & ev3val <= -10106

keep if a == 1 & b == 1

twoway (scatter yhat ev2val, sort) ///
(scatter y ev2val, sort)


twoway (scatter emaxap ev2val, sort) ///
(scatter maxeap ev2val, sort) ///
(scatter emaxhat ev2val, sort)



import delimited "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\df3aprox.csv", clear

gen a = 1 if ev1val >= 19053 & ev1val <= 19054
gen b = 1 if ev3val >= -15125 & ev3val <= -15124

keep if a == 1 & b == 1

twoway (scatter yhat ev2val, sort) ///
(scatter y ev2val, sort)


twoway (scatter emaxap ev2val, sort) ///
(scatter maxeap ev2val, sort) ///
(scatter emaxhat ev2val, sort)

