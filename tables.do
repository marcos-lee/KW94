cd "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\"

use keane_wolpin_1994_restat.dta, clear

gen school_c = 0
replace school_c = 1 if choice == 3

gen work1 = 0
replace work1 = 1 if choice == 1

gen work2 = 0
replace work2 = 1 if choice == 2

gen home = 0
replace home = 1 if choice == 4


preserve
collapse school_c work1 work2 home, by(period)
keep if period == 1 | period == 10 | period == 20 | period == 30 | period == 40
tabstat school_c, by(period)
tabstat work1, by(period)
tabstat work2, by(period)
tabstat home, by(period)
restore

foreach i in 100000 2000 1000 250{
	import delimited output\df1_MC`i'.csv, clear
	
preserve
collapse school_c work1 work2 home, by(period)
keep if period == 1 | period == 10 | period == 20 | period == 30 | period == 40
tabstat school_c, by(period)
tabstat work1, by(period)
tabstat work2, by(period)
tabstat home, by(period)
restore
}


foreach i in 2000 500{
	import delimited output\df1_MC2000_S`i'.csv, clear
	
preserve
collapse school_c work1 work2 home, by(period)
keep if period == 1 | period == 10 | period == 20 | period == 30 | period == 40
tabstat school_c, by(period)
tabstat work1, by(period)
tabstat work2, by(period)
tabstat home, by(period)
restore
}

